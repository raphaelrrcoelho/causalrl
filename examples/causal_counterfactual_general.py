# STATUS: canonical · L3 counterfactual — random parameters; genuine abduction-action-prediction generalization  ·  map: CAUSAL_LLM.md
"""Generalising L3: a twin-network counterfactual head over *random-parameter* SCMs.

`causal_counterfactual_twin.py` proved the twin-network architecture computes counterfactuals
correctly — but on one fixed SCM with only 8 evidence states, so "learning" was an 8-entry lookup,
not generalisation. This file removes that crutch: the mechanism parameters (edge weights, node
biases) are drawn at random per example and fed to the model as features, so the evidence/parameter
space is continuous and infinite. The model cannot memorise; it must learn the *abduction-action-
prediction algorithm* itself and apply it to SCMs it has never seen.

Topology is the fixed confounded chain (Z→X→Y, Z→Y; Z observed) so the counterfactual has an exact
closed form (monotone-noise interval abduction, matching causalrl). What varies, and what the model
must generalise over, are the continuous mechanism parameters — including learning to *ignore* the
parameters that are irrelevant to the query (Z's and X's own mechanisms) and use only Y's.

Honesty is built in (the d-separation lesson). We score against the exact oracle AND against the two
shortcuts a model could fake L3 with — predict the factual outcome, predict the interventional
marginal — and report separately on the counterfactual-relevant subset where the truth differs from
*both*. We also test **out-of-distribution generalisation**: parameters from a wider range than
training. A model doing real L3 beats both baselines on the relevant subset and degrades gracefully
OOD; a lookup/shortcut does not.

Run::

    uv run --extra torch python examples/causal_counterfactual_general.py

Scope: random parameters over a fixed topology. Random *topologies* (exact-MC oracle) are the next
step — see examples/CAUSAL_LLM_RESEARCH.md. Didactic research scaffold, not a perf guarantee.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

# fixed confounded chain: Z=0 (confounder, observed), X=1 (treatment), Y=2 (outcome)
Z, X, Y = 0, 1, 2
N = 3
SELF, PARENT, CHILD, NONE = 0, 1, 2, 3
N_REL = 4
BASE_REL = torch.full((N, N), NONE)
for i in range(N):
    BASE_REL[i, i] = SELF
for a, b in [(Z, X), (Z, Y), (X, Y)]:  # a -> b
    BASE_REL[b, a] = PARENT
    BASE_REL[a, b] = CHILD


# ============================================================================================
# Random-parameter SCM batch + exact counterfactual oracle (vectorised)
# ============================================================================================


def make_batch(b: int, sigma: float, device: str, gen: torch.Generator) -> dict[str, Tensor]:
    def randn(*shape: int) -> Tensor:
        return torch.randn(*shape, generator=gen, device=device) * sigma

    def rand(*shape: int) -> Tensor:
        return torch.rand(*shape, generator=gen, device=device)

    bZ, bX, bY = randn(b), randn(b), randn(b)
    wXZ, wYX, wYZ = randn(b), randn(b), randn(b)

    # sample a unit (exogenous noise -> factual values)
    z = (rand(b) < torch.sigmoid(bZ)).float()
    pX = torch.sigmoid(bX + wXZ * z)
    x = (rand(b) < pX).float()
    pY = torch.sigmoid(bY + wYX * x + wYZ * z)
    y = (rand(b) < pY).float()

    x_new = 1.0 - x  # counterfactual treatment
    pY_cf = torch.sigmoid(bY + wYX * x_new + wYZ * z)

    # exact P(Y_{do(X=x')} = 1 | z, x, y): monotone-noise abduction of u_Y (matches causalrl)
    target = torch.where(
        y > 0.5,
        torch.clamp(torch.minimum(pY_cf, pY) / pY.clamp(min=1e-6), 0, 1),
        torch.clamp((pY_cf - pY).clamp(min=0) / (1 - pY).clamp(min=1e-6), 0, 1),
    )
    # shortcut baselines
    factual = y
    pz = torch.sigmoid(bZ)  # interventional marginal P(Y|do x') = E_Z[ sigmoid(bY+wYX x'+wYZ Z) ]
    interv = pz * torch.sigmoid(bY + wYX * x_new + wYZ) + (1 - pz) * torch.sigmoid(bY + wYX * x_new)

    # per-node bias feature and symmetric edge-weight feature (coefficient on each present edge)
    node_bias = torch.stack([bZ, bX, bY], dim=1)  # (b, 3)
    wfeat = torch.zeros(b, N, N, device=device)
    for i, j, w in [(Z, X, wXZ), (Z, Y, wYZ), (X, Y, wYX)]:
        wfeat[:, i, j] = w
        wfeat[:, j, i] = w
    return {
        "z": z, "x": x, "y": y, "x_new": x_new, "node_bias": node_bias, "wfeat": wfeat,
        "target": target, "factual": factual, "interv": interv,
    }


# ============================================================================================
# Twin-network graph transformer with continuous parameter features
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
        self.w_scale = nn.Parameter(torch.zeros(heads))  # how much edge weight bends attention

    def forward(self, h: Tensor, rel: Tensor, wfeat: Tensor) -> Tensor:
        b, n, _ = h.shape
        q, k, v = self.qkv(h).split(h.shape[-1], dim=2)
        q = q.view(b, n, self.h, self.dh).transpose(1, 2)
        k = k.view(b, n, self.h, self.dh).transpose(1, 2)
        v = v.view(b, n, self.h, self.dh).transpose(1, 2)
        bias = self.rel_bias[:, rel].permute(1, 0, 2, 3)  # (b,H,n,n) from discrete relation
        bias = bias + wfeat[:, None] * self.w_scale[None, :, None, None]  # + continuous weight
        out = nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        return self.proj(out.transpose(1, 2).reshape(b, n, self.h * self.dh))


class Block(nn.Module):
    def __init__(self, d: int, heads: int) -> None:
        super().__init__()
        self.n1, self.attn = RMSNorm(d), GraphAttention(d, heads)
        hid = ((8 * d // 3 + 31) // 32) * 32
        self.n2 = RMSNorm(d)
        self.w1 = nn.Linear(d, hid, bias=False)
        self.w3 = nn.Linear(d, hid, bias=False)
        self.w2 = nn.Linear(hid, d, bias=False)

    def forward(self, h: Tensor, rel: Tensor, wfeat: Tensor) -> Tensor:
        h = h + self.attn(self.n1(h), rel, wfeat)
        z = self.n2(h)
        return h + self.w2(nn.functional.silu(self.w1(z)) * self.w3(z))


class TwinCFGeneral(nn.Module):
    def __init__(self, d: int = 96, heads: int = 6, layers: int = 3) -> None:
        super().__init__()
        self.node_id = nn.Embedding(N, d)
        self.val_emb = nn.Embedding(3, d)  # 0, 1, or 2 = unknown (to predict)
        self.bias_proj = nn.Linear(1, d, bias=False)  # continuous node-bias feature
        self.world = nn.Embedding(2, d)
        self.encoder = nn.ModuleList([Block(d, heads) for _ in range(layers)])
        self.decoder = nn.ModuleList([Block(d, heads) for _ in range(layers)])
        self.read = nn.Sequential(RMSNorm(d), nn.Linear(d, 1, bias=False))

    def _embed(self, vals: Tensor, node_bias: Tensor, world: int) -> Tensor:
        b = vals.shape[0]
        ids = torch.arange(N, device=vals.device)
        w = torch.full((b, N), world, device=vals.device)
        return (
            self.node_id(ids)[None]
            + self.val_emb(vals)
            + self.bias_proj(node_bias[..., None])
            + self.world(w)
        )

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        b = batch["z"].shape[0]
        dev = batch["z"].device
        rel = BASE_REL.to(dev)[None].expand(b, N, N)
        wfeat = batch["wfeat"]
        nb = batch["node_bias"]

        # World A (factual): abduct from full evidence (z, x, y).
        fz, fx, fy = batch["z"].long(), batch["x"].long(), batch["y"].long()
        fa = torch.stack([fz, fx, fy], dim=1)
        ha = self._embed(fa, nb, world=0)
        for blk in self.encoder:
            ha = blk(ha, rel, wfeat)

        # World B (counterfactual): share ha; do(X=x') by edge ablation (sever Z->X) + clamp X.
        cb = fa.clone()
        cb[:, X] = batch["x_new"].long()
        cb[:, Y] = 2  # unknown
        rel_do = rel.clone()
        rel_do[:, X, Z] = NONE  # X no longer depends on its parent Z
        wfeat_do = wfeat.clone()
        wfeat_do[:, X, Z] = 0.0
        hb = ha + self._embed(cb, nb, world=1)
        for blk in self.decoder:
            hb = blk(hb, rel_do, wfeat_do)
        return self.read(hb[:, Y]).squeeze(-1)


# ============================================================================================
# Train (fresh random SCMs every step -> generalisation by construction) + honest audit
# ============================================================================================


def brier(p: Tensor, t: Tensor) -> float:
    return float((p - t).pow(2).mean())


@torch.no_grad()
def audit(model: nn.Module, sigma: float, label: str, device: str, gen: torch.Generator) -> None:
    b = make_batch(40000, sigma, device, gen)
    p = torch.sigmoid(model(b))
    tgt, fac, inv = b["target"], b["factual"], b["interv"]
    differs = ((tgt - fac).abs() > 0.25) & ((tgt - inv).abs() > 0.1)
    d = differs

    def on(x: Tensor) -> float:
        return float((x[d] - tgt[d]).pow(2).mean()) if int(d.sum()) else float("nan")

    print(f"\n[{label}]  Brier vs exact oracle (lower better):")
    print(f"  twin network        : {brier(p, tgt):.4f}")
    print(f"  factual baseline    : {brier(fac, tgt):.4f}")
    print(f"  interventional base : {brier(inv, tgt):.4f}")
    print(f"  counterfactual-relevant subset ({int(d.sum())}/40000, truth differs from both):")
    print(f"    twin network      : {on(p):.4f}")
    print(f"    factual baseline  : {on(fac):.4f}")
    print(f"    interventional    : {on(inv):.4f}")


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    gen = torch.Generator(device=device).manual_seed(1)
    model = TwinCFGeneral().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    sigma = 1.5  # training parameter scale
    n_params = sum(p.numel() for p in model.parameters())
    print(f"training generalising twin-network CF head ({n_params / 1e3:.0f}K params), "
          "fresh random SCMs each step ...")
    for step in range(6000):
        b = make_batch(256, sigma, device, gen)
        loss = nn.functional.binary_cross_entropy_with_logits(model(b), b["target"])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (step + 1) % 1000 == 0:
            print(f"  step {step + 1}/6000  loss {loss.item():.4f}")

    model.eval()
    audit(model, sigma, "in-distribution params (sigma=1.5)", device, gen)
    audit(model, 2.5, "OOD params (sigma=2.5, wider than training)", device, gen)
    print(
        "\nReal L3 (not a shortcut) = the twin network beats BOTH baselines on the "
        "counterfactual-relevant subset, in-distribution AND out-of-distribution. Because "
        "parameters are continuous and SCMs are fresh every step, this cannot be memorisation — it "
        "is the abduction-action-prediction algorithm generalising over unseen SCMs. (Fixed "
        "topology; random topologies are the next step.)"
    )


if __name__ == "__main__":
    main()
