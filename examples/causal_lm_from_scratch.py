"""Train a tiny language model *from scratch* on data from a causalrl SCM — and show it
learns to tell ``do(X)`` apart from ``see(X)``.

The word "causal" means two different things, and this example deliberately uses both:

1. **Causal in the LLM sense** — the model below is an autoregressive ("causal") transformer:
   a triangular attention mask, next-token prediction. This is what GPT-style pretraining is.
   We build it from scratch in plain PyTorch (no ``transformers`` dependency) so nothing is
   hidden. That is the "treinar desde a fundação" part.

2. **Causal in the Pearl sense** — the *data* comes from a causalrl
   :class:`~causalrl.StructuralCausalModel` with an unobserved confounder ``U``. We feed the
   model both observational samples (tagged ``<see>``) and interventional samples
   (tagged ``<do>``), but never show it ``U``. The question is whether a from-scratch LM can
   internalise the interventional distribution ``P(Y | do(X))`` and keep it separate from the
   confounded observational association ``P(Y | X)``.

The SCM is rigged so the two differ a lot:

    U ~ Bernoulli(0.5)                      # confounder, hidden from the model
    P(X=1 | U) = 0.2 + 0.6*U                # X tracks U
    P(Y=1 | X,U) = 0.5 + 0.15*(2X-1) + 0.35*(2U-1)

    truth:  P(Y=1 | do(X=1)) = 0.65   P(Y=1 | do(X=0)) = 0.35   -> causal gap 0.30
    confounded: P(Y=1 | X=1)  = 0.86   P(Y=1 | X=0)  = 0.14     -> seen gap   0.72

After training, prompting the model with ``[<do>, x=1]`` should yield P(y=1) near 0.65
(the real interventional value), while ``[<see>, x=1]`` should yield near 0.86 — recovering
the gap between intervening and merely observing, straight from the SCM via the library.

Run::

    uv run --extra torch python examples/causal_lm_from_scratch.py

It trains in a few seconds on CPU. causalrl is the source of ground truth throughout.
This is a didactic demonstration, not a performance claim.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.distributions import Uniform

from causalrl import CausalGraph, FunctionalMechanism, StructuralCausalModel

# --------------------------------------------------------------------------------------------
# 1. The confounded SCM, built with causalrl. U is an explicit latent node; the model never
#    sees it. Each node's exogenous noise is Uniform(0,1), thresholded into a Bernoulli draw
#    inside the mechanism.
# --------------------------------------------------------------------------------------------


def build_confounded_scm() -> StructuralCausalModel:
    graph = CausalGraph(directed_edges=[("U", "X"), ("U", "Y"), ("X", "Y")])

    def u_mech(_parents: dict[str, Tensor], noise: Tensor) -> Tensor:
        return (noise < 0.5).float()

    def x_mech(parents: dict[str, Tensor], noise: Tensor) -> Tensor:
        prob = 0.2 + 0.6 * parents["U"]  # P(X=1|U) = 0.2 or 0.8
        return (noise < prob).float()

    def y_mech(parents: dict[str, Tensor], noise: Tensor) -> Tensor:
        prob = 0.5 + 0.15 * (2 * parents["X"] - 1) + 0.35 * (2 * parents["U"] - 1)
        return (noise < prob.clamp(0.0, 1.0)).float()

    mechanisms = {
        "U": FunctionalMechanism([], u_mech),
        "X": FunctionalMechanism(["U"], x_mech),
        "Y": FunctionalMechanism(["X", "U"], y_mech),
    }
    exogenous = {name: Uniform(0.0, 1.0) for name in ("U", "X", "Y")}
    return StructuralCausalModel(graph, mechanisms, exogenous)


# --------------------------------------------------------------------------------------------
# 2. Tokenisation. A sample becomes a length-3 sequence: [regime, x, y]. The model only ever
#    sees X and Y — U is held out, which is exactly what makes the confounding bite.
# --------------------------------------------------------------------------------------------

SEE, DO, X0, X1, Y0, Y1 = range(6)
VOCAB_SIZE = 6
SEQ_LEN = 3


def encode(regime: int, x: Tensor, y: Tensor) -> Tensor:
    xt = torch.where(x > 0.5, X1, X0)
    yt = torch.where(y > 0.5, Y1, Y0)
    reg = torch.full_like(xt, regime)
    return torch.stack([reg, xt, yt], dim=1)  # (n, 3)


def make_dataset(scm: StructuralCausalModel, n: int, seed: int) -> Tensor:
    """Half observational (<see>), half interventional (<do>), split evenly over do(X=0/1)."""
    obs = scm.see(n, seed=seed)
    see_rows = encode(SEE, obs["X"], obs["Y"])

    half = n // 2
    do0 = scm.do({"X": 0.0}).see(half, seed=seed + 1)
    do1 = scm.do({"X": 1.0}).see(n - half, seed=seed + 2)
    do_rows = torch.cat(
        [encode(DO, do0["X"], do0["Y"]), encode(DO, do1["X"], do1["Y"])], dim=0
    )

    rows = torch.cat([see_rows, do_rows], dim=0)
    perm = torch.randperm(rows.shape[0], generator=torch.Generator().manual_seed(seed))
    return rows[perm]


# --------------------------------------------------------------------------------------------
# 3. A tiny autoregressive ("causal") transformer, from scratch. Causal mask + next-token loss.
# --------------------------------------------------------------------------------------------


class TinyGPT(nn.Module):
    def __init__(self, vocab: int, d_model: int = 32, n_heads: int = 4, n_layers: int = 2) -> None:
        super().__init__()
        self.tok = nn.Embedding(vocab, d_model)
        self.pos = nn.Embedding(SEQ_LEN, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=4 * d_model, batch_first=True, dropout=0.0
        )
        self.blocks = nn.TransformerEncoder(layer, n_layers)
        self.head = nn.Linear(d_model, vocab)

    def forward(self, idx: Tensor) -> Tensor:
        t = idx.shape[1]
        pos = torch.arange(t, device=idx.device)
        h = self.tok(idx) + self.pos(pos)[None]
        mask = torch.triu(torch.ones(t, t, device=idx.device), diagonal=1).bool()  # causal mask
        h = self.blocks(h, mask=mask)
        return self.head(h)  # (n, t, vocab)


def train(model: nn.Module, data: Tensor, epochs: int = 300, lr: float = 3e-3) -> None:
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    inputs, targets = data[:, :-1], data[:, 1:]
    for _ in range(epochs):
        logits = model(inputs)
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, VOCAB_SIZE), targets.reshape(-1)
        )
        opt.zero_grad()
        loss.backward()
        opt.step()


@torch.no_grad()
def prob_y1(model: nn.Module, regime: int, x_tok: int) -> float:
    """Model's P(y=1) given a [regime, x] prompt — read off the next-token distribution."""
    prompt = torch.tensor([[regime, x_tok]])
    logits = model(prompt)[0, -1]  # distribution over the token after x
    p = torch.softmax(logits[[Y0, Y1]], dim=-1)
    return float(p[1])


# --------------------------------------------------------------------------------------------
# 4. Ground truth from causalrl, then train and compare.
# --------------------------------------------------------------------------------------------


def scm_truth(scm: StructuralCausalModel, n: int = 200_000) -> dict[str, float]:
    obs = scm.see(n, seed=7)
    x, y = obs["X"], obs["Y"]
    do1 = scm.do({"X": 1.0}).see(n, seed=8)["Y"]
    do0 = scm.do({"X": 0.0}).see(n, seed=9)["Y"]
    return {
        "see_x1": float(y[x > 0.5].mean()),
        "see_x0": float(y[x < 0.5].mean()),
        "do_x1": float(do1.mean()),
        "do_x0": float(do0.mean()),
    }


def main() -> None:
    torch.manual_seed(0)
    scm = build_confounded_scm()
    truth = scm_truth(scm)

    data = make_dataset(scm, n=20_000, seed=0)
    model = TinyGPT(VOCAB_SIZE)
    train(model, data)

    m_see_x1 = prob_y1(model, SEE, X1)
    m_do_x1 = prob_y1(model, DO, X1)
    m_do_x0 = prob_y1(model, DO, X0)

    print("                         P(Y=1)")
    print("                    SCM truth   tiny-GPT")
    print(f"  see  X=1   (confounded)  {truth['see_x1']:.3f}      {m_see_x1:.3f}")
    print(f"  do   X=1   (causal)      {truth['do_x1']:.3f}      {m_do_x1:.3f}")
    print(f"  do   X=0   (causal)      {truth['do_x0']:.3f}      {m_do_x0:.3f}")
    print()
    print(f"  causal gap  P(Y|do X=1) - P(Y|do X=0):  truth {truth['do_x1'] - truth['do_x0']:+.3f}"
          f"   model {m_do_x1 - m_do_x0:+.3f}")
    print(f"  the model keeps do(X=1)={m_do_x1:.2f} distinct from see(X=1)={m_see_x1:.2f}: it "
          "learned to intervene, not just to observe.")


if __name__ == "__main__":
    main()
