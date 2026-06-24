# STATUS: by-construction · L3 counterfactual — fixed-SCM twin network (8-state lookup)  ·  map: CAUSAL_LLM.md
"""L3 layer: a twin-network counterfactual head — abduction-action-prediction inside the network.

L1 (seeing) and L2 (doing) are about populations; L3 (counterfactual) is about a *specific unit*:
"this patient took the drug and recovered — would they have recovered WITHOUT it?" Answering it
requires **abduction** — inferring the unit's exogenous noise from its evidence — then re-running
the mechanisms under a different action holding that noise fixed. That is what distinguishes L3 from
L2, and it is what no amount of observational/interventional pattern-matching can fake (Causal
Hierarchy Theorem).

The architecture is Pearl's **twin network**: two coupled copies of the model — a *factual* world
that reads the evidence and a *counterfactual* world with do(X) applied — that **share the same
exogenous-noise representation**. Here:

* world A (factual) encodes the unit's evidence into a per-node latent — this is the abduction;
* world B (counterfactual) takes that shared latent, applies do(X=x') by **edge ablation** (severing
  X's parents, the native causal-core intervention) and clamping X, and reads the counterfactual
  outcome. The shared latent is the coupling that makes this a counterfactual, not a fresh do().

Honesty, built in from the start (the lesson of the d-separation audit). The counterfactual is only
interesting when it disagrees with the two shortcuts a model could fake it with:

* the **factual** outcome (ignore the intervention), and
* the **interventional marginal** P(Y | do(x')) (ignore the unit's evidence).

So we evaluate against an exact `causalrl`-consistent oracle AND both baselines, and we report
separately on the *counterfactual-relevant* subset where the true answer differs from both. A model
that only matches overall but not there has learned a shortcut, not L3.

The SCM is the confounded chain from `causal_ncm_reasoning.py` (Z→X→Y, Z→Y; Z observed), with
monotone threshold mechanisms so the counterfactual is exactly computable. A *controlled* demo
of the L3 architecture; learning abduction across many unseen SCMs is the open frontier
(examples/CAUSAL_LLM_RESEARCH.md).

Run::

    uv run --extra torch python examples/causal_counterfactual_twin.py
"""

from __future__ import annotations

import random

import torch
from torch import Tensor, nn

# nodes: 0=Z (confounder, observed), 1=X (treatment), 2=Y (outcome)
Z, X, Y = 0, 1, 2
N = 3
# causal relations for attention (relation of j w.r.t. i, for i attending to j)
SELF, PARENT, CHILD, NONE = 0, 1, 2, 3
N_REL = 4
# the chain Z->X, Z->Y, X->Y as a relation matrix
BASE_REL = [[SELF, NONE, NONE] for _ in range(N)]
for a, b in [(Z, X), (Z, Y), (X, Y)]:  # a -> b
    BASE_REL[b][a] = PARENT
    BASE_REL[a][b] = CHILD


# ============================================================================================
# The SCM and its exact counterfactual (the causalrl-consistent oracle)
# ============================================================================================


def p_x(z: int) -> float:
    return 0.2 + 0.6 * z


def p_y(x: int, z: int) -> float:
    return min(1.0, max(0.0, 0.5 + 0.15 * (2 * x - 1) + 0.35 * (2 * z - 1)))


def sample_unit(rng: random.Random) -> tuple[int, int, int, float, float, float]:
    """Sample a unit's exogenous noise and return (z, x, y, u_z, u_x, u_y)."""
    uz, ux, uy = rng.random(), rng.random(), rng.random()
    z = int(uz < 0.5)
    x = int(ux < p_x(z))
    y = int(uy < p_y(x, z))
    return z, x, y, uz, ux, uy


def cf_prob(z: int, x: int, y: int, x_new: int) -> float:
    """Exact P(Y_{do(X=x_new)} = 1 | evidence z, x, y) — monotone-noise abduction.

    Only Y is downstream of X (Z is observed and upstream), so abduct u_Y from the factual outcome
    and re-evaluate Y's mechanism under the new treatment. This matches causalrl's
    StructuralCausalModel.counterfactual on this SCM.
    """
    pf, pc = p_y(x, z), p_y(x_new, z)
    if y == 1:  # u_Y ~ U(0, pf);  P(u_Y < pc)
        return min(pc, pf) / pf if pf > 0 else 0.0
    return max(0.0, pc - pf) / (1.0 - pf) if pf < 1 else 0.0  # u_Y ~ U(pf, 1)


def interventional_marginal(x_new: int) -> float:
    """P(Y=1 | do(X=x_new)) — the evidence-free shortcut baseline."""
    return 0.5 * p_y(x_new, 0) + 0.5 * p_y(x_new, 1)


# ============================================================================================
# Data: each example is a unit's evidence + an intervention, labelled by the exact CF probability
# ============================================================================================


def make_batch(rng: random.Random, n: int, device: str) -> dict[str, Tensor]:
    vals, x_new, target, factual, interv = [], [], [], [], []
    for _ in range(n):
        z, x, y, *_ = sample_unit(rng)
        xn = 1 - x  # counterfactual: flip the treatment
        vals.append([z, x, y])
        x_new.append(xn)
        target.append(cf_prob(z, x, y, xn))
        factual.append(float(y))
        interv.append(interventional_marginal(xn))
    return {
        "values": torch.tensor(vals, device=device),  # (n, 3) factual node values
        "x_new": torch.tensor(x_new, device=device),  # (n,) counterfactual treatment value
        "target": torch.tensor(target, device=device),  # (n,) exact P(Y_cf=1 | evidence)
        "factual": torch.tensor(factual, device=device),
        "interv": torch.tensor(interv, device=device),
    }


# ============================================================================================
# Twin-network graph transformer
# ============================================================================================


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps, self.weight = eps, nn.Parameter(torch.ones(d))

    def forward(self, x: Tensor) -> Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class GraphAttention(nn.Module):
    def __init__(self, d: int, heads: int) -> None:
        super().__init__()
        self.h, self.dh = heads, d // heads
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.rel_bias = nn.Parameter(torch.zeros(heads, N_REL))

    def forward(self, h: Tensor, rel: Tensor) -> Tensor:
        b, n, _ = h.shape
        q, k, v = self.qkv(h).split(h.shape[-1], dim=2)
        q = q.view(b, n, self.h, self.dh).transpose(1, 2)
        k = k.view(b, n, self.h, self.dh).transpose(1, 2)
        v = v.view(b, n, self.h, self.dh).transpose(1, 2)
        bias = self.rel_bias[:, rel].permute(1, 0, 2, 3)  # (B,H,N,N)
        out = nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        return self.proj(out.transpose(1, 2).reshape(b, n, self.h * self.dh))


class Block(nn.Module):
    def __init__(self, d: int, heads: int) -> None:
        super().__init__()
        self.n1, self.attn = RMSNorm(d), GraphAttention(d, heads)
        hidden = ((8 * d // 3 + 31) // 32) * 32
        self.n2 = RMSNorm(d)
        self.w1, self.w3, self.w2 = (
            nn.Linear(d, hidden, bias=False),
            nn.Linear(d, hidden, bias=False),
            nn.Linear(hidden, d, bias=False),
        )

    def forward(self, h: Tensor, rel: Tensor) -> Tensor:
        h = h + self.attn(self.n1(h), rel)
        z = self.n2(h)
        return h + self.w2(nn.functional.silu(self.w1(z)) * self.w3(z))


class TwinCounterfactualNet(nn.Module):
    def __init__(self, d: int = 96, heads: int = 6, layers: int = 3) -> None:
        super().__init__()
        self.node_id = nn.Embedding(N, d)  # which variable (Z/X/Y)
        self.val_emb = nn.Embedding(3, d)  # value 0, 1, or 2 = "set by intervention / unknown"
        self.world = nn.Embedding(2, d)  # factual vs counterfactual world tag
        self.encoder = nn.ModuleList([Block(d, heads) for _ in range(layers)])
        self.decoder = nn.ModuleList([Block(d, heads) for _ in range(layers)])
        self.read = nn.Sequential(RMSNorm(d), nn.Linear(d, 1, bias=False))

    def forward(self, values: Tensor, x_new: Tensor) -> Tensor:
        b = values.shape[0]
        dev = values.device
        ids = torch.arange(N, device=dev)
        base_rel = torch.tensor(BASE_REL, device=dev)[None].expand(b, N, N)

        # World A (factual): read the evidence -> latent "noise" per node = ABDUCTION.
        ha = self.node_id(ids)[None] + self.val_emb(values) + self.world(torch.zeros_like(values))
        for blk in self.encoder:
            ha = blk(ha, base_rel)

        # World B (counterfactual): share the abducted latent ha; apply do(X=x_new) by edge ablation
        # (sever X's incoming PARENT edge from Z) and clamp X to its new value; Y becomes "unknown".
        cf_vals = values.clone()
        cf_vals[:, X] = x_new
        cf_vals[:, Y] = 2  # unknown: to be predicted
        rel_do = base_rel.clone()
        rel_do[:, X, Z] = NONE  # graph surgery: X no longer depends on its parent Z
        hb = (
            ha + self.node_id(ids)[None] + self.val_emb(cf_vals)
            + self.world(torch.ones_like(values))
        )
        for blk in self.decoder:
            hb = blk(hb, rel_do)
        return self.read(hb[:, Y]).squeeze(-1)  # logit for P(Y_cf = 1)


# ============================================================================================
# Train + honest, shortcut-aware evaluation
# ============================================================================================


@torch.no_grad()
def report(model: nn.Module, rng: random.Random, device: str) -> None:
    b = make_batch(rng, 20000, device)
    p_model = torch.sigmoid(model(b["values"], b["x_new"]))
    tgt, fac, inv = b["target"], b["factual"], b["interv"]

    def brier(p: Tensor) -> float:
        return float((p - tgt).pow(2).mean())

    def acc(p: Tensor) -> float:
        return float(((p > 0.5) == (tgt > 0.5)).float().mean())

    # counterfactual-relevant subset: true answer differs from BOTH shortcuts
    differs = ((tgt - fac).abs() > 0.25) & ((tgt - inv).abs() > 0.1)
    d = differs

    def brier_on(p: Tensor, m: Tensor) -> float:
        return float((p[m] - tgt[m]).pow(2).mean()) if int(m.sum()) else float("nan")

    print("\nBrier score vs exact causalrl-consistent oracle (lower is better):")
    print(f"  twin network        : {brier(p_model):.4f}   (acc {acc(p_model):.3f})")
    print(f"  factual baseline    : {brier(fac):.4f}   (predict the factual outcome)")
    print(f"  interventional base : {brier(inv):.4f}   (predict P(Y|do x'), ignore the unit)")
    print(f"\non the counterfactual-relevant subset ({int(d.sum())} of 20000, "
          "where the truth differs from BOTH shortcuts):")
    print(f"  twin network        : {brier_on(p_model, d):.4f}")
    print(f"  factual baseline    : {brier_on(fac, d):.4f}")
    print(f"  interventional base : {brier_on(inv, d):.4f}")
    print("\nThe twin network is doing L3 only if it beats BOTH baselines on that subset — i.e. it "
          "used the unit's evidence (abduction) AND the intervention, not one or the other.")
    print(
        "\nHONESTY CHECK: this SCM has only 8 distinct evidence states (z,x,y in {0,1}^3), so a "
        "Brier of ~0 means the architecture *correctly represents and learns* the exact "
        "counterfactual function (an 8-entry table) and does not collapse to a shortcut — it does "
        "NOT demonstrate generalisation. Learning abduction over richer/continuous SCMs and unseen "
        "units is the open frontier (see examples/CAUSAL_LLM_RESEARCH.md)."
    )


def main() -> None:
    torch.manual_seed(0)
    rng = random.Random(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TwinCounterfactualNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"training twin-network counterfactual head ({n_params / 1e3:.0f}K params) ...")
    for step in range(3000):
        b = make_batch(rng, 256, device)
        logit = model(b["values"], b["x_new"])
        loss = nn.functional.binary_cross_entropy_with_logits(logit, b["target"])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (step + 1) % 500 == 0:
            print(f"  step {step + 1}/3000  loss {loss.item():.4f}")
    report(model, random.Random(1), device)


if __name__ == "__main__":
    main()
