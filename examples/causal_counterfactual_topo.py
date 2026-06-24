# STATUS: research · L3 counterfactual — random topologies; latest step (verify results before citing)  ·  map: CAUSAL_LLM.md
"""Generalising L3, next step: twin-network counterfactuals over random TOPOLOGIES *and* parameters.

`causal_counterfactual_general.py` generalised over random parameters of a fixed chain. This removes
the last crutch — the fixed graph — by drawing a random DAG *and* random mechanism parameters per
example, with a treatment/outcome pair chosen on the fly. The model sees only structure (relations),
parameters (node biases, edge weights), the unit's evidence, and the intervention; it must run
abduction-action-prediction over a graph it has never seen, at sizes it was never trained on.

Because a node downstream of X can itself be random, the counterfactual has no single closed form,
so the oracle is **exact Monte-Carlo abduction**: each exogenous u_i is sampled from the interval
its factual value pins it to (no rejection — full acceptance, unbiased), then the do-mutilated
mechanisms are propagated. This is exactly causalrl's abduction-action-prediction, vectorised.

Honesty is built in (the d-separation lesson): score against the exact oracle AND the two shortcuts
(predict the factual outcome; predict the interventional marginal), report on the counterfactual-
relevant subset where the truth differs from both, and test **size extrapolation** (train 3-4 vars,
test 5) and **OOD parameters**.

PREPARED, NOT YET TUNED. Launch with::

    uv run --extra torch python examples/causal_counterfactual_topo.py            # full
    uv run --extra torch python examples/causal_counterfactual_topo.py --smoke    # seconds, CPU

Didactic research scaffold, not a performance guarantee.
"""

from __future__ import annotations

import argparse

import torch
from torch import Tensor, nn

PLAIN, ROLE_X, ROLE_Y = 0, 1, 2
SELF, PARENT, CHILD, NONE = 0, 1, 2, 3
N_REL = 4


# ============================================================================================
# Random DAG + random-parameter SCM, with an exact Monte-Carlo counterfactual oracle (vectorised)
# ============================================================================================


def _reachability(adj: Tensor) -> Tensor:
    """Boolean strict-descendant matrix from adjacency ``adj`` (B,n,n), edges i->j."""
    n = adj.shape[-1]
    reach = adj.clone()
    power = adj.clone()
    for _ in range(n - 1):
        power = (power.float() @ adj.float() > 0).to(adj.dtype)
        reach = reach | power
    return reach


def _propagate(weff: Tensor, b: Tensor, u: Tensor, x_node: Tensor, x_val: Tensor) -> Tensor:
    """Evaluate the (do-mutilated) SCM. weff,(B,n,n) effective weights; b,(B,n); u,(B,K,n).

    Node order 0..n-1 is topological (edges only i<j). At node x_node the value is clamped to x_val
    (the intervention) and its parents are ignored — graph surgery.
    """
    n = u.shape[-1]
    v = torch.zeros_like(u)
    for j in range(n):
        contrib = torch.einsum("bi,bki->bk", weff[:, :, j], v)
        p = torch.sigmoid(b[:, j][:, None] + contrib)
        vj = (u[:, :, j] < p).float()
        clamp = (x_node == j)[:, None]
        v[:, :, j] = torch.where(clamp, x_val[:, None], vj)
    return v


def _factual(weff: Tensor, b: Tensor, u: Tensor) -> tuple[Tensor, Tensor]:
    """Factual values (B,n) and per-node probabilities (B,n)."""
    bsz, n = b.shape
    v = torch.zeros(bsz, n, device=b.device)
    p = torch.zeros(bsz, n, device=b.device)
    for j in range(n):
        p[:, j] = torch.sigmoid(b[:, j] + torch.einsum("bi,bi->b", weff[:, :, j], v))
        v[:, j] = (u[:, j] < p[:, j]).float()
    return v, p


def make_batch(bsz: int, n: int, sigma: float, edge_prob: float, k: int, device: str,
               gen: torch.Generator) -> dict[str, Tensor] | None:
    """A batch of random SCMs of size n, each with a unit, an intervention, and the exact CF target.

    Returns None if too few examples have a valid (X -> ... -> Y) pair (e.g. very sparse graphs).
    """
    tri = torch.triu(torch.ones(n, n, device=device), 1).bool()
    adj = (torch.rand(bsz, n, n, generator=gen, device=device) < edge_prob) & tri
    weff = torch.randn(bsz, n, n, generator=gen, device=device) * sigma * adj.float()
    b = torch.randn(bsz, n, generator=gen, device=device) * sigma

    u = torch.rand(bsz, n, generator=gen, device=device)
    v, pf = _factual(weff, b, u)

    # choose X (has a descendant) and Y (a descendant of X) per example
    reach = _reachability(adj)
    x_idx = torch.zeros(bsz, dtype=torch.long, device=device)
    y_idx = torch.zeros(bsz, dtype=torch.long, device=device)
    valid = torch.zeros(bsz, dtype=torch.bool, device=device)
    for e in range(bsz):
        pairs = reach[e].nonzero(as_tuple=False)
        if pairs.shape[0] == 0:
            continue
        pick = pairs[int(torch.randint(pairs.shape[0], (1,), generator=gen, device=device))]
        x_idx[e], y_idx[e], valid[e] = pick[0], pick[1], True
    if int(valid.sum()) < bsz // 2:
        return None

    ar = torch.arange(bsz, device=device)
    x_fac = v[ar, x_idx]
    x_new = 1.0 - x_fac
    roles = torch.full((bsz, n), PLAIN, device=device, dtype=torch.long)
    roles[ar, x_idx] = ROLE_X
    roles[ar, y_idx] = ROLE_Y

    # exact-MC abduction: sample u_i from the interval its factual value pins it to (full accept)
    r = torch.rand(bsz, k, n, generator=gen, device=device)
    pf_k, v_k = pf[:, None, :], v[:, None, :]
    u_cf = torch.where(v_k > 0.5, r * pf_k, pf_k + r * (1 - pf_k))
    v_cf = _propagate(weff, b, u_cf, x_idx, x_new)
    target = v_cf[ar, :, y_idx].mean(1)

    # interventional marginal (fresh noise; ignore the unit's evidence)
    u_int = torch.rand(bsz, k, n, generator=gen, device=device)
    v_int = _propagate(weff, b, u_int, x_idx, x_new)
    interv = v_int[ar, :, y_idx].mean(1)

    # symmetric edge-weight feature + relation matrix
    wfeat = weff + weff.transpose(1, 2)
    rel = torch.full((bsz, n, n), NONE, device=device, dtype=torch.long)
    eye = torch.eye(n, device=device, dtype=torch.bool)[None].expand(bsz, n, n)
    rel = rel.masked_fill(eye, SELF)
    rel = rel.masked_fill(adj.transpose(1, 2), PARENT)  # j is a parent of i
    rel = rel.masked_fill(adj, CHILD)
    return {
        "values": v.long(), "roles": roles, "rel": rel, "wfeat": wfeat, "node_bias": b,
        "x_idx": x_idx, "y_idx": y_idx, "x_new": x_new, "target": target,
        "factual": v[ar, y_idx], "interv": interv, "valid": valid,
    }


# ============================================================================================
# Twin-network graph transformer (variable size, continuous parameters)  — Llama-recipe internals
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
        self.w_scale = nn.Parameter(torch.zeros(heads))

    def forward(self, h: Tensor, rel: Tensor, wfeat: Tensor) -> Tensor:
        b, n, _ = h.shape
        q, k, v = self.qkv(h).split(h.shape[-1], dim=2)
        q = q.view(b, n, self.h, self.dh).transpose(1, 2)
        k = k.view(b, n, self.h, self.dh).transpose(1, 2)
        v = v.view(b, n, self.h, self.dh).transpose(1, 2)
        bias = self.rel_bias[:, rel].permute(1, 0, 2, 3)
        bias = bias + wfeat[:, None] * self.w_scale[None, :, None, None]
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


class TwinCFTopo(nn.Module):
    def __init__(self, d: int = 96, heads: int = 6, layers: int = 4) -> None:
        super().__init__()
        self.role_emb = nn.Embedding(3, d)
        self.val_emb = nn.Embedding(3, d)  # 0, 1, or 2 = unknown
        self.bias_proj = nn.Linear(1, d, bias=False)
        self.world = nn.Embedding(2, d)
        self.encoder = nn.ModuleList([Block(d, heads) for _ in range(layers)])
        self.decoder = nn.ModuleList([Block(d, heads) for _ in range(layers)])
        self.read = nn.Sequential(RMSNorm(d), nn.Linear(d, 1, bias=False))

    def _embed(self, roles: Tensor, vals: Tensor, nb: Tensor, world: int) -> Tensor:
        w = torch.full_like(roles, world)
        return (
            self.role_emb(roles) + self.val_emb(vals)
            + self.bias_proj(nb[..., None]) + self.world(w)
        )

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        rel, wfeat, nb = batch["rel"], batch["wfeat"], batch["node_bias"]
        roles, vals = batch["roles"], batch["values"]
        ar = torch.arange(roles.shape[0], device=roles.device)

        ha = self._embed(roles, vals, nb, 0)  # factual world: abduction from evidence
        for blk in self.encoder:
            ha = blk(ha, rel, wfeat)

        cf_vals = vals.clone()
        cf_vals[ar, batch["x_idx"]] = batch["x_new"].long()
        cf_vals[ar, batch["y_idx"]] = 2  # unknown
        rel_do = rel.clone()
        wfeat_do = wfeat.clone()
        # do(X): sever X's incoming parent edges
        for e in range(roles.shape[0]):
            xi = int(batch["x_idx"][e])
            parents = rel[e, xi] == PARENT
            rel_do[e, xi][parents] = NONE
            wfeat_do[e, xi][parents] = 0.0
        hb = ha + self._embed(roles, cf_vals, nb, 1)
        for blk in self.decoder:
            hb = blk(hb, rel_do, wfeat_do)
        return self.read(hb[ar, batch["y_idx"]]).squeeze(-1)


# ============================================================================================
# Train (fresh random SCMs every step) + honest, shortcut-aware audit
# ============================================================================================


def _masked_brier(p: Tensor, t: Tensor, m: Tensor) -> float:
    return float((p[m] - t[m]).pow(2).mean()) if int(m.sum()) else float("nan")


@torch.no_grad()
def audit(model: nn.Module, n: int, sigma: float, edge_prob: float, label: str, device: str,
          gen: torch.Generator) -> None:
    b = None
    while b is None:
        b = make_batch(20000, n, sigma, edge_prob, 256, device, gen)
    p = torch.sigmoid(model(b))
    tgt, fac, inv, valid = b["target"], b["factual"], b["interv"], b["valid"]
    differs = valid & ((tgt - fac).abs() > 0.25) & ((tgt - inv).abs() > 0.1)
    print(f"\n[{label}]  Brier vs exact-MC oracle on the counterfactual-relevant subset "
          f"({int(differs.sum())} examples):")
    print(f"  twin network        : {_masked_brier(p, tgt, differs):.4f}")
    print(f"  factual baseline    : {_masked_brier(fac, tgt, differs):.4f}")
    print(f"  interventional base : {_masked_brier(inv, tgt, differs):.4f}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Generalising L3 over random topologies + parameters.")
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--mc", type=int, default=256, help="Monte-Carlo samples for the oracle")
    ap.add_argument("--sigma", type=float, default=1.5)
    ap.add_argument("--edge-prob", type=float, default=0.5)
    ap.add_argument("--train-sizes", type=int, nargs="+", default=[3, 4])
    ap.add_argument("--extrap-size", type=int, default=5)
    ap.add_argument("--d-model", type=int, default=96)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        a.steps, a.batch, a.mc, a.d_model, a.layers = 60, 64, 64, 48, 2

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    gen = torch.Generator(device=device).manual_seed(1)
    model = TwinCFTopo(d=a.d_model, layers=a.layers).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"training twin-network CF over random topologies ({n_params / 1e3:.0f}K params), "
          f"train sizes {a.train_sizes}, fresh SCMs each step ...")
    rng_sizes = a.train_sizes
    for step in range(a.steps):
        n = rng_sizes[step % len(rng_sizes)]
        batch = make_batch(a.batch, n, a.sigma, a.edge_prob, a.mc, device, gen)
        if batch is None:
            continue
        logit = model(batch)
        loss_full = nn.functional.binary_cross_entropy_with_logits(
            logit, batch["target"], reduction="none"
        )
        loss = (loss_full * batch["valid"]).sum() / batch["valid"].sum().clamp(min=1)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (step + 1) % max(1, a.steps // 8) == 0:
            print(f"  step {step + 1}/{a.steps}  loss {loss.item():.4f}")

    model.eval()
    audit(model, a.train_sizes[-1], a.sigma, a.edge_prob, "in-distribution", device, gen)
    audit(model, a.extrap_size, a.sigma, a.edge_prob,
          f"size extrapolation (n={a.extrap_size}, unseen)", device, gen)
    audit(model, a.train_sizes[-1], a.sigma + 1.0, a.edge_prob, "OOD parameters", device, gen)
    print("\nReal, generalising L3 = the twin network beats BOTH baselines on the counterfactual-"
          "relevant subset across unseen topologies, unseen sizes, and OOD parameters.")


if __name__ == "__main__":
    main()
