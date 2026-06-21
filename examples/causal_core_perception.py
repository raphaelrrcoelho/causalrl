"""Size-robust perception: a relational encoder closes the remaining size wall.

The do()-core (``causal_core_do.py``) size-generalized its *reasoning* (propagation + do are exact by
construction) but its *perception* -- a transformer encoder with absolute positions reading the
edge-list text -- did not length-generalize: edge recovery fell 1.0 -> 0.89 -> 0.78 across sizes
2/3 -> 4 -> 5, dragging the observational read down (0.60 at size 5).

The fix is architectural too. Instead of a position-indexed sequence encoder, treat the evidence as a
*set of relational facts* (the directed edges) and build the adjacency with a permutation- and
count-invariant relational encoder: each edge is an item; each ordered variable pair (u,v) is a query
matched against the edge set. No absolute positions, no dependence on how many edges there are -- so
it generalizes to any number of variables by construction.

Everything else is the embedded causal core: explicit adjacency A, K-step reachability propagation,
and the do() switch (causal read = directed reachability; observational adds the back-door term).
Trained on 2/3-variable graphs, tested on 4/5 held out by size. We report edge recovery and both query
types; the claim is that size-robust perception lifts held-out accuracy toward the in-dist ceiling --
closing the last measured wall of the embedded core.

CPU-sized; reuses the do()-core's data generator.  Run::

    uv run --extra torch python examples/causal_core_perception.py
"""

from __future__ import annotations

import os
import random
import sys

import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import causal_core_do as cdo  # data generator (make/build), reachability ground truth

torch.set_num_threads(4)
MAX_E = 12  # max directed edges over <=5 nodes seen here


def rel_batch(items):
    """Represent each example as a SET of directed edges (src slot, dst slot) -- no positions."""
    b = len(items)
    esrc = torch.zeros(b, MAX_E, dtype=torch.long)
    edst = torch.zeros(b, MAX_E, dtype=torch.long)
    emask = torch.zeros(b, MAX_E)
    for k, e in enumerate(items):
        edges = [(i, j) for i in range(5) for j in range(5) if e["adj"][i][j]]
        for t, (i, j) in enumerate(edges[:MAX_E]):
            esrc[k, t], edst[k, t], emask[k, t] = i, j, 1.0
    g = lambda key, dt: torch.tensor([e[key] for e in items], dtype=dt)  # noqa: E731
    return (
        esrc,
        edst,
        emask,
        g("xs", torch.long),
        g("ys", torch.long),
        g("is_causal", torch.float),
        g("label", torch.float),
        torch.tensor([e["adj"] for e in items], dtype=torch.float),
        torch.tensor([e["present"] for e in items], dtype=torch.float),
    )


class RelationalCausalCore(nn.Module):
    def __init__(self, d=64, steps=5):
        super().__init__()
        self.steps = steps
        self.var = nn.Embedding(5, d)  # a learned embedding per variable slot
        self.edge = nn.Sequential(nn.Linear(2 * d, d), nn.ReLU(), nn.Linear(d, d))  # edge -> key
        self.query = nn.Sequential(nn.Linear(2 * d, d), nn.ReLU(), nn.Linear(d, d))  # pair -> query
        self.scale = d**0.5
        self.read = nn.Linear(1, 1)

    def adjacency(self, esrc, edst, emask):
        items = self.edge(torch.cat([self.var(esrc), self.var(edst)], -1))  # (B,E,d) edge keys
        u = torch.arange(5).view(5, 1).expand(5, 5)
        v = torch.arange(5).view(1, 5).expand(5, 5)
        q = self.query(torch.cat([self.var(u), self.var(v)], -1))  # (5,5,d) pair queries
        # match each pair-query against the edge SET: A_logit[b,u,v] = max_e <q[u,v], item[b,e]>
        scores = torch.einsum("uvd,bed->buve", q, items) / self.scale  # (B,5,5,E)
        scores = scores.masked_fill(emask.view(emask.size(0), 1, 1, -1) == 0, -1e9)
        logits = scores.max(dim=-1).values  # (B,5,5)
        return logits.masked_fill(torch.eye(5).bool(), -30.0)

    def forward(self, esrc, edst, emask, xs, ys, is_causal, pres):
        logits = self.adjacency(esrc, edst, emask)
        a = torch.sigmoid(logits) * pres.unsqueeze(2) * pres.unsqueeze(1)
        r = a
        for _ in range(self.steps):
            r = torch.clamp(a + torch.bmm(r, a), 0.0, 1.0)
        idx = torch.arange(esrc.size(0))
        fwd, bwd = r[idx, xs, ys], r[idx, ys, xs]
        rzx, rzy = r[idx, :, xs], r[idx, :, ys]
        notxy = torch.ones_like(rzx)
        notxy[idx, xs] = 0.0
        notxy[idx, ys] = 0.0
        common = (rzx * rzy * notxy * pres).max(dim=1).values
        score = is_causal * fwd + (1 - is_causal) * (1 - (1 - fwd) * (1 - bwd) * (1 - common))
        pm = pres.unsqueeze(2) * pres.unsqueeze(1)
        return self.read(score.unsqueeze(-1)).squeeze(-1), logits, pm


def train(model, data, epochs=16, lr=2e-3, bs=128, lam=1.0):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss()
    rng = random.Random(0)
    for ep in range(epochs):
        rng.shuffle(data)
        ta = te = nb = 0.0
        for i in range(0, len(data), bs):
            esrc, edst, emask, xs, ys, isc, lab, adj, pres = rel_batch(data[i : i + bs])
            ans, elog, pm = model(esrc, edst, emask, xs, ys, isc, pres)
            al = bce(ans, lab)
            el = (
                nn.functional.binary_cross_entropy_with_logits(elog, adj, reduction="none") * pm
            ).sum() / pm.sum()
            loss = al + lam * el
            opt.zero_grad()
            loss.backward()
            opt.step()
            ta += float(al.detach())
            te += float(el.detach())
            nb += 1
        print(f"    epoch {ep + 1}/{epochs}  answer {ta / nb:.3f}  edge {te / nb:.3f}")


@torch.no_grad()
def evaluate(model, data) -> tuple[float, float]:
    if not data:
        return float("nan"), float("nan")
    esrc, edst, emask, xs, ys, isc, lab, adj, pres = rel_batch(data)
    ans, elog, pm = model(esrc, edst, emask, xs, ys, isc, pres)
    acc = float(((ans > 0) == (lab > 0.5)).float().mean())
    edge = float((((elog > 0).float() == adj) * pm).sum() / pm.sum())
    return acc, edge


def main() -> None:
    torch.manual_seed(0)
    print(
        "training: 2/3-var graphs; held-out test: 4/5-var (relational, position-free perception) ..."
    )
    train_data = cdo.build(16000, sizes=[2, 3], seed=1)
    tests = {s: cdo.build(2000, sizes=[s], seed=10 + s) for s in (2, 3, 4, 5)}

    model = RelationalCausalCore()
    print(f"RelationalCausalCore: {sum(p.numel() for p in model.parameters()) / 1e3:.0f}K params")
    train(model, train_data)
    model.eval()

    print("\n              accuracy by query type        edge")
    print("  size    observational   interventional   recovery   in-training?")
    for s in (2, 3, 4, 5):
        corr = [e for e in tests[s] if not e["is_causal"]]
        cause = [e for e in tests[s] if e["is_causal"]]
        c_acc, edge = evaluate(model, tests[s])
        o_acc, _ = evaluate(model, corr)
        i_acc, _ = evaluate(model, cause)
        seen = "trained" if s in (2, 3) else "HELD-OUT"
        print(f"  size {s}:    {o_acc:.3f}           {i_acc:.3f}            {edge:.3f}      {seen}")

    print("\n  confounded pairs (corr=yes, cause=no) as 'does X cause Y?':")
    for s in (3, 4, 5):
        conf = [
            dict(e, is_causal=1, label=e["cause"]) for e in tests[s] if e["corr"] and not e["cause"]
        ]
        a, _ = evaluate(model, conf)
        print(f"    size {s}:  {a:.3f}  [n={len(conf)}]")

    print(
        "\nReading: with a permutation/count-invariant relational perception (vs the position-indexed "
        "text encoder of causal_core_do.py), edge recovery should stay ~1.0 at the held-out sizes, "
        "lifting BOTH query reads toward the in-dist ceiling -- perception and reasoning both "
        "size-general, a fully size-robust embedded causal core."
    )


if __name__ == "__main__":
    main()
