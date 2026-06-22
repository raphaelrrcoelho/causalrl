"""Corrected experiment: is the causal REASONING actually learnable, or only hand-coded?

The independent audit (AUDIT.md) found the decisive caveat: in the earlier "embedded core" scripts
the reachability + do() + back-door computation is a HARD-CODED differentiable formula -- the models
only learn edge perception, so the size-generalization was by construction, not learned. This script
finishes the exercise honestly: it gives the model the TRUE structure (so perception is not the
variable) and makes the REASONING itself learned, then measures whether a learned reasoner can do
causal reasoning (cause = directed reachability; corr = marginal d-connection incl. common cause)
and generalize to graph sizes never seen in training.

Three reasoners, same data:
  * HARDWIRED   the fixed reachability+do() formula (NOT learned) -- by-construction reference (~1.0).
  * MLP         flatten the adjacency + query -> MLP -> answer (learned; not algorithmically aligned).
  * GNN         K rounds of learned message passing over the adjacency -> readout (learned; aligned
                to reachability, so it *might* size-generalize).

Variable slots are randomized so every slot is exercised at every size (a fair size-extrapolation
test, no new-token confound). Trained on 2/3-variable graphs, tested on 3 (in-dist), 4 and 5 (held).

The honest question: does a *learned* causal reasoner match the hand-coded one, and does it extrapolate?

CPU-sized.  Run::

    uv run --extra torch python examples/causal_core_learned_reasoning.py
"""

from __future__ import annotations

import random

import torch
from torch import nn

NMAX = 5
torch.set_num_threads(4)


def reachable(adj, i, j) -> bool:
    seen, stack = set(), [i]
    while stack:
        u = stack.pop()
        for v in range(NMAX):
            if adj[u][v] and v not in seen:
                if v == j:
                    return True
                seen.add(v)
                stack.append(v)
    return False


def make(sizes, rng) -> dict:
    k = rng.choice(sizes)
    slots = sorted(rng.sample(range(NMAX), k))  # random slot subset -> fair size-extrapolation
    order = slots[:]
    rng.shuffle(order)
    adj = [[0] * NMAX for _ in range(NMAX)]
    for a in range(k):
        for b in range(a + 1, k):
            if rng.random() < 0.5:
                adj[order[a]][order[b]] = 1
    x, y = rng.sample(slots, 2)
    cause = reachable(adj, x, y)
    desc = {s: {t for t in slots if t != s and reachable(adj, s, t)} for s in slots}
    common = any(z not in (x, y) and x in desc[z] and y in desc[z] for z in slots)
    corr = cause or reachable(adj, y, x) or common
    is_causal = rng.random() < 0.5
    return {
        "adj": adj,
        "xs": x,
        "ys": y,
        "is_causal": int(is_causal),
        "cause": int(cause),
        "corr": int(corr),
        "label": int(cause if is_causal else corr),
        "present": [s in slots for s in range(NMAX)],
    }


def build(n, sizes, seed) -> list[dict]:
    rng = random.Random(seed)
    out, tries = [], 0
    cnt = {(t, lab): 0 for t in (0, 1) for lab in (0, 1)}
    cap = n // 4
    while len(out) < 4 * cap and tries < n * 200:
        tries += 1
        e = make(sizes, rng)
        key = (e["is_causal"], e["label"])
        if cnt[key] >= cap:
            continue
        out.append(e)
        cnt[key] += 1
    rng.shuffle(out)
    return out


def pack(items):
    adj = torch.tensor([e["adj"] for e in items], dtype=torch.float)
    g = lambda key, dt: torch.tensor([e[key] for e in items], dtype=dt)  # noqa: E731
    return (
        adj,
        g("xs", torch.long),
        g("ys", torch.long),
        g("is_causal", torch.float),
        g("label", torch.float),
        torch.tensor([e["present"] for e in items], dtype=torch.float),
    )


class Hardwired(nn.Module):
    """The earlier hand-coded core: exact reachability + fixed do()/back-door formula. NOT learned."""

    def forward(self, adj, xs, ys, isc, pres):
        a = adj * pres.unsqueeze(2) * pres.unsqueeze(1)
        r = a
        for _ in range(NMAX):
            r = torch.clamp(a + torch.bmm(r, a), 0.0, 1.0)
        idx = torch.arange(adj.size(0))
        fwd, bwd = r[idx, xs, ys], r[idx, ys, xs]
        rzx, rzy = r[idx, :, xs], r[idx, :, ys]
        notxy = torch.ones_like(rzx)
        notxy[idx, xs] = 0.0
        notxy[idx, ys] = 0.0
        common = (rzx * rzy * notxy * pres).max(1).values
        score = isc * fwd + (1 - isc) * (1 - (1 - fwd) * (1 - bwd) * (1 - common))
        return (score - 0.5) * 20  # logit-ish; threshold at 0.5


class MLPReasoner(nn.Module):
    def __init__(self, d=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(NMAX * NMAX + 2 * NMAX + 1 + NMAX, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, 1),
        )

    def forward(self, adj, xs, ys, isc, pres):
        b = adj.size(0)
        xoh = torch.zeros(b, NMAX).scatter_(1, xs.unsqueeze(1), 1.0)
        yoh = torch.zeros(b, NMAX).scatter_(1, ys.unsqueeze(1), 1.0)
        feat = torch.cat([adj.reshape(b, -1), xoh, yoh, isc.unsqueeze(1), pres], dim=1)
        return self.net(feat).squeeze(-1)


class GNNReasoner(nn.Module):
    """Learned K-step message passing over the (given) adjacency -> readout for the query."""

    def __init__(self, d=48, steps=5):
        super().__init__()
        self.steps = steps
        self.init = nn.Parameter(torch.randn(d) * 0.1)
        self.win = nn.Linear(d, d)
        self.wout = nn.Linear(d, d)
        self.w0 = nn.Linear(d, d)
        self.read = nn.Sequential(nn.Linear(2 * d + 1, d), nn.ReLU(), nn.Linear(d, 1))

    def forward(self, adj, xs, ys, isc, pres):
        b = adj.size(0)
        h = self.init.view(1, 1, -1).expand(b, NMAX, -1) * pres.unsqueeze(-1)
        for _ in range(self.steps):
            m_in = torch.bmm(adj.transpose(1, 2), self.win(h))  # messages from parents (ancestry)
            m_out = torch.bmm(adj, self.wout(h))  # messages from children
            h = torch.relu(self.w0(h) + m_in + m_out) * pres.unsqueeze(-1)
        idx = torch.arange(b)
        hx, hy = h[idx, xs], h[idx, ys]
        return self.read(torch.cat([hx, hy, isc.unsqueeze(1)], dim=1)).squeeze(-1)


def train(model, data, epochs=25, lr=2e-3, bs=128):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss()
    rng = random.Random(0)
    for _ in range(epochs):
        rng.shuffle(data)
        for i in range(0, len(data), bs):
            adj, xs, ys, isc, lab, pres = pack(data[i : i + bs])
            loss = bce(model(adj, xs, ys, isc, pres), lab)
            opt.zero_grad()
            loss.backward()
            opt.step()


@torch.no_grad()
def acc(model, data) -> float:
    if not data:
        return float("nan")
    adj, xs, ys, isc, lab, pres = pack(data)
    return float(((model(adj, xs, ys, isc, pres) > 0) == (lab > 0.5)).float().mean())


def report(name, model, tests):
    print(f"\n  {name}")
    print("  size   corr   cause   confounded   (in training?)")
    for s in (3, 4, 5):
        d = tests[s]
        corr = [e for e in d if not e["is_causal"]]
        cause = [e for e in d if e["is_causal"]]
        conf = [dict(e, is_causal=1, label=e["cause"]) for e in d if e["corr"] and not e["cause"]]
        seen = "trained" if s == 3 else "HELD-OUT"
        print(
            f"  {s}     {acc(model, corr):.3f}  {acc(model, cause):.3f}   {acc(model, conf):.3f}"
            f"       {seen}"
        )


def main() -> None:
    torch.manual_seed(0)
    print("Is causal REASONING learnable & size-general? (structure given; reasoning learned)")
    train_data = build(12000, sizes=[2, 3], seed=1)
    tests = {s: build(2000, sizes=[s], seed=10 + s) for s in (3, 4, 5)}

    report("HARDWIRED formula (NOT learned -- by-construction reference)", Hardwired(), tests)

    mlp = MLPReasoner()
    train(mlp, train_data)
    report("MLP reasoner (learned)", mlp, tests)

    gnn = GNNReasoner()
    train(gnn, train_data)
    report("GNN reasoner (learned message passing)", gnn, tests)

    print(
        "\nReading: HARDWIRED is ~1.0 everywhere BY CONSTRUCTION (it is the algorithm, not learned). "
        "The honest question is the learned reasoners: if MLP/GNN match it in-dist (size 3) but drop "
        "out of size (4/5), a learned causal reasoner does NOT extrapolate -- the earlier 'size-general "
        "causal core' was the hand-coded formula doing the work, not learning. If the GNN holds out of "
        "size, algorithmic alignment lets a learned reasoner generalize; if not, size-general causal "
        "reasoning stays unlearned (only hand-codable)."
    )


if __name__ == "__main__":
    main()
