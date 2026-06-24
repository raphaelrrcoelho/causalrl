# STATUS: oracle-fed · Act 3 Learnability — embedded discovery (interventional facts are the true edges)  ·  map: CAUSAL_LLM.md
"""Closing the loop: embedded discovery -- statistical evidence -> structure -> causal reasoning.

So far the embedded core was *given* the structure (a set of edges). The last brick is discovery:
infer the adjacency from *statistical evidence*, in the weights, then reason + do() over it. This makes
the whole pipeline -- evidence -> structure -> observational/interventional answer -- a single model.

Two evidence regimes feed the SAME discovery+reasoning core:

  * OBSERVATIONAL  conditional-(in)dependence facts (correlations): for pairs (i,j) and subsets Z,
    whether i _||_ j | Z. These identify the skeleton + colliders but NOT the orientation of other
    edges -- the Markov-equivalence limit. Supervised toward the true DAG, the discoverer therefore
    cannot learn those orientations (no signal in the evidence) -> partial structure -> MEC-capped.
  * INTERVENTIONAL the direct effects do(i)->j (an experiment's immediate result) = oriented edges.
    These identify the full DAG -> full structure -> answers toward 1.0.

Discovery is a relational cross-attention: each ordered pair (u,v) is a query that attends over the
evidence facts and emits an edge logit -- permutation/count-invariant, so it size-generalizes. The
rest is the embedded core: K-step reachability propagation + the do() switch (causal read = directed
reach, observational read adds the back-door). Trained on 2/3-variable graphs, tested on 4 held out.

The result to look for: interventional evidence -> high structure recovery and ~1.0 answers (incl.
confounded), observational evidence -> lower structure recovery and MEC-capped answers -- the
*discovery-level* version of "beyond correlations", end-to-end in one architecture.

CPU-sized.  Run::

    uv run --extra torch python examples/causal_core_discovery.py
"""

from __future__ import annotations

import itertools
import os
import random
import sys

import torch
from torch import nn

import causalrl as C
from causalrl.identification._separation import d_separated

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

LETTERS = "ABCDE"
MAX_F = 80  # max evidence facts
torch.set_num_threads(4)


def reachable(adj, i, j) -> bool:
    seen, stack = set(), [i]
    while stack:
        u = stack.pop()
        for v in range(5):
            if adj[u][v] and v not in seen:
                if v == j:
                    return True
                seen.add(v)
                stack.append(v)
    return False


def make(sizes, rng) -> dict:
    k = rng.choice(sizes)
    letters = rng.sample(LETTERS, k)
    idx = [LETTERS.index(c) for c in letters]
    order = letters[:]
    rng.shuffle(order)
    adj = [[0] * 5 for _ in range(5)]
    edges = []
    for a in range(k):
        for b in range(a + 1, k):
            if rng.random() < 0.5:
                u, v = order[a], order[b]
                adj[LETTERS.index(u)][LETTERS.index(v)] = 1
                edges.append((u, v))
    g = C.CausalGraph(directed_edges=edges, nodes=letters)
    interventional = rng.random() < 0.5
    facts = []  # each: (i, j, Zmask(5 floats), value, type) ; type 1.0 = interventional
    if interventional:
        for i in idx:
            for j in idx:
                if i != j:
                    facts.append((i, j, [0.0] * 5, float(adj[i][j]), 1.0))
    else:
        for a, b in itertools.combinations(idx, 2):
            others = [z for z in idx if z not in (a, b)]
            for r in range(len(others) + 1):
                for Z in itertools.combinations(others, r):
                    zmask = [1.0 if z in Z else 0.0 for z in range(5)]
                    indep = d_separated(g, {LETTERS[a]}, {LETTERS[b]}, {LETTERS[z] for z in Z})
                    facts.append((a, b, zmask, float(indep), 0.0))
    x, y = rng.sample(idx, 2)
    cause = reachable(adj, x, y)
    corr = not d_separated(g, {LETTERS[x]}, {LETTERS[y]}, set())
    is_causal = rng.random() < 0.5
    return {
        "facts": facts[:MAX_F],
        "xs": x,
        "ys": y,
        "is_causal": int(is_causal),
        "cause": int(cause),
        "corr": int(corr),
        "label": int(cause if is_causal else corr),
        "adj": adj,
        "present": [chr(65 + i) in letters for i in range(5)],
        "interventional": int(interventional),
    }


def build(n, sizes, seed) -> list[dict]:
    rng = random.Random(seed)
    out, tries = [], 0
    cnt = {(t, lab): 0 for t in (0, 1) for lab in (0, 1)}
    cap = n // 4
    while len(out) < 4 * cap and tries < n * 800:
        tries += 1
        e = make(sizes, rng)
        key = (e["is_causal"], e["label"])
        if cnt[key] >= cap:
            continue
        out.append(e)
        cnt[key] += 1
    rng.shuffle(out)
    return out


def batch(items):
    b = len(items)
    fi = torch.zeros(b, MAX_F, dtype=torch.long)
    fj = torch.zeros(b, MAX_F, dtype=torch.long)
    fz = torch.zeros(b, MAX_F, 5)
    fval = torch.zeros(b, MAX_F)
    ftype = torch.zeros(b, MAX_F)
    fmask = torch.zeros(b, MAX_F)
    for k, e in enumerate(items):
        for t, (i, j, zmask, val, typ) in enumerate(e["facts"]):
            fi[k, t], fj[k, t] = i, j
            fz[k, t] = torch.tensor(zmask)
            fval[k, t], ftype[k, t], fmask[k, t] = val, typ, 1.0
    g = lambda key, dt: torch.tensor([e[key] for e in items], dtype=dt)  # noqa: E731
    return (
        fi,
        fj,
        fz,
        fval,
        ftype,
        fmask,
        g("xs", torch.long),
        g("ys", torch.long),
        g("is_causal", torch.float),
        g("label", torch.float),
        torch.tensor([e["adj"] for e in items], dtype=torch.float),
        torch.tensor([e["present"] for e in items], dtype=torch.float),
    )


class DiscoveryCausalCore(nn.Module):
    def __init__(self, d=64, steps=5):
        super().__init__()
        self.steps = steps
        self.var = nn.Embedding(5, d)
        self.fact = nn.Sequential(nn.Linear(3 * d + 2, d), nn.ReLU(), nn.Linear(d, d))
        self.query = nn.Sequential(nn.Linear(2 * d, d), nn.ReLU(), nn.Linear(d, d))
        self.combine = nn.Sequential(nn.Linear(2 * d, d), nn.ReLU(), nn.Linear(d, 1))
        self.scale = d**0.5
        self.read = nn.Linear(1, 1)

    def adjacency(self, fi, fj, fz, fval, ftype, fmask):
        zpool = fz @ self.var.weight  # (B,F,d) pooled conditioning-set embedding
        feat = torch.cat(
            [self.var(fi), self.var(fj), zpool, fval.unsqueeze(-1), ftype.unsqueeze(-1)], dim=-1
        )
        items = self.fact(feat)  # (B,F,d)
        u = torch.arange(5).view(5, 1).expand(5, 5)
        v = torch.arange(5).view(1, 5).expand(5, 5)
        q = self.query(torch.cat([self.var(u), self.var(v)], -1))  # (5,5,d)
        att = torch.einsum("uvd,bfd->buvf", q, items) / self.scale  # (B,5,5,F)
        att = att.masked_fill(fmask.view(fmask.size(0), 1, 1, -1) == 0, -1e9)
        w = torch.softmax(att, dim=-1)
        agg = torch.einsum("buvf,bfd->buvd", w, items)  # (B,5,5,d)
        qb = q.unsqueeze(0).expand(agg.size(0), -1, -1, -1)
        logits = self.combine(torch.cat([qb, agg], -1)).squeeze(-1)  # (B,5,5)
        return logits.masked_fill(torch.eye(5).bool(), -30.0)

    def forward(self, fi, fj, fz, fval, ftype, fmask, xs, ys, is_causal, pres):
        logits = self.adjacency(fi, fj, fz, fval, ftype, fmask)
        a = torch.sigmoid(logits) * pres.unsqueeze(2) * pres.unsqueeze(1)
        r = a
        for _ in range(self.steps):
            r = torch.clamp(a + torch.bmm(r, a), 0.0, 1.0)
        idx = torch.arange(a.size(0))
        fwd, bwd = r[idx, xs, ys], r[idx, ys, xs]
        rzx, rzy = r[idx, :, xs], r[idx, :, ys]
        notxy = torch.ones_like(rzx)
        notxy[idx, xs] = 0.0
        notxy[idx, ys] = 0.0
        common = (rzx * rzy * notxy * pres).max(dim=1).values
        score = is_causal * fwd + (1 - is_causal) * (1 - (1 - fwd) * (1 - bwd) * (1 - common))
        pm = pres.unsqueeze(2) * pres.unsqueeze(1)
        return self.read(score.unsqueeze(-1)).squeeze(-1), logits, pm


def train(model, data, epochs=18, lr=2e-3, bs=128, lam=1.0):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss()
    rng = random.Random(0)
    for ep in range(epochs):
        rng.shuffle(data)
        ta = te = nb = 0.0
        for i in range(0, len(data), bs):
            *ev, xs, ys, isc, lab, adj, pres = batch(data[i : i + bs])
            ans, elog, pm = model(*ev, xs, ys, isc, pres)
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
    *ev, xs, ys, isc, lab, adj, pres = batch(data)
    ans, elog, pm = model(*ev, xs, ys, isc, pres)
    acc = float(((ans > 0) == (lab > 0.5)).float().mean())
    edge = float((((elog > 0).float() == adj) * pm).sum() / pm.sum())
    return acc, edge


def main() -> None:
    torch.manual_seed(0)
    print(
        "building discovery data (evidence -> structure -> answer); train 2/3-var, test 4 held-out"
    )
    train_data = build(12000, sizes=[2, 3], seed=1)
    tests = {s: build(2400, sizes=[s], seed=10 + s) for s in (3, 4)}

    model = DiscoveryCausalCore()
    print(f"DiscoveryCausalCore: {sum(p.numel() for p in model.parameters()) / 1e3:.0f}K params")
    train(model, train_data)
    model.eval()

    print("\n           structure recovery & answer accuracy, by evidence type")
    print("  size   evidence        edge-recovery   answer-acc   confounded-cause   seen?")
    for s in (3, 4):
        for typ, name in [(0, "observational"), (1, "interventional")]:
            sub = [e for e in tests[s] if e["interventional"] == typ]
            acc, edge = evaluate(model, sub)
            conf = [
                dict(e, is_causal=1, label=e["cause"]) for e in sub if e["corr"] and not e["cause"]
            ]
            ca, _ = evaluate(model, conf)
            seen = "trained" if s == 3 else "HELD-OUT"
            print(
                f"  size {s}  {name:14s}  {edge:.3f}           {acc:.3f}        {ca:.3f} "
                f"[n={len(conf)}]   {seen}"
            )

    print(
        "\nReading: one embedded model discovers the structure from statistical evidence, then "
        "reasons + do() over it. Interventional evidence recovers the oriented DAG -> high edge "
        "recovery and answers (incl. confounded); observational evidence recovers the skeleton but "
        "not the unidentifiable orientations -> lower edge recovery and MEC-capped answers. That is "
        "discovery-level 'beyond correlations', end-to-end in one architecture, and size-general."
    )


if __name__ == "__main__":
    main()
