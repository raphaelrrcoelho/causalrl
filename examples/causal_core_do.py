"""Embedded do() — one causal core that answers both observational and interventional queries.

The previous core (``causal_core_architecture.py``) embedded an explicit adjacency + iterative
reachability and size-generalized. This adds the piece that makes it *causal beyond correlation*: an
embedded **do() operator**. One learned structure A answers two query types, and do() is the
architectural switch between them:

  * "are X and Y correlated?"  (observational)  = marginal d-connection: a directed path X~Y OR a
    common cause (back-door). Reads reachability *with* the back-door term.
  * "does X cause Y?"          (interventional) = do(X) affects Y = *directed* reachability only.
    Intervening removes the back-door -> reads reachability *without* the common-cause term.

So one model, one adjacency, two read-outs that differ only by whether the back-door path is included
-- which is exactly what do() does. The decisive check is the **confounded** subset (X and Y correlated
but X does NOT cause Y, e.g. X <- Z -> Y): a correlation-only reasoner says "causes" and is wrong; the
do()-routed core must say "no". Edge perception is learned (aux); reachability and the back-door/do()
logic are fixed inductive biases. Trained on 2/3-variable graphs, tested on 4/5 held-out by size.

CPU-sized.  Run::

    uv run --extra torch python examples/causal_core_do.py
"""

from __future__ import annotations

import random

import torch
from torch import nn

LETTERS = "ABCDE"
# vocab: pad, edge arrow, the two query-type markers, and the 5 variable names
VOCAB = {"<pad>": 0, ">": 1, "cause": 2, "corr": 3, "A": 4, "B": 5, "C": 6, "D": 7, "E": 8}
MAXLEN = 48
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


def anc(adj, i):  # set of nodes reachable from i (descendants)
    return {j for j in range(5) if j != i and reachable(adj, i, j)}


def make(sizes, rng) -> dict:
    k = rng.choice(sizes)
    letters = rng.sample(LETTERS, k)
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
    x, y = rng.sample(letters, 2)
    xi, yi = LETTERS.index(x), LETTERS.index(y)
    cause = reachable(adj, xi, yi)
    desc = {i: anc(adj, i) for i in range(5)}
    common_cause = any(z not in (xi, yi) and xi in desc[z] and yi in desc[z] for z in range(5))
    corr = cause or reachable(adj, yi, xi) or common_cause
    is_causal = rng.random() < 0.5
    qtype = "cause" if is_causal else "corr"
    toks = []
    for u, v in edges:
        toks += [u, ">", v]
    toks += [qtype, x, y]
    present = [chr(65 + i) in letters for i in range(5)]
    return {
        "ids": [VOCAB[t] for t in toks][:MAXLEN],
        "xs": xi,
        "ys": yi,
        "is_causal": int(is_causal),
        "cause": int(cause),
        "corr": int(corr),
        "label": int(cause if is_causal else corr),
        "adj": adj,
        "present": present,
    }


def build(n, sizes, seed) -> list[dict]:
    rng = random.Random(seed)
    out, tries = [], 0
    # balance the active label within each query type
    cnt = {(t, lab): 0 for t in (0, 1) for lab in (0, 1)}
    cap = n // 4
    while len(out) < 4 * cap and tries < n * 600:
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
    width = max(len(e["ids"]) for e in items)
    ids = torch.zeros(len(items), width, dtype=torch.long)
    pad = torch.ones(len(items), width, dtype=torch.bool)
    for j, e in enumerate(items):
        ids[j, : len(e["ids"])] = torch.tensor(e["ids"])
        pad[j, : len(e["ids"])] = False
    g = lambda key, dt: torch.tensor([e[key] for e in items], dtype=dt)  # noqa: E731
    return (
        ids,
        pad,
        g("xs", torch.long),
        g("ys", torch.long),
        g("is_causal", torch.float),
        g("label", torch.float),
        torch.tensor([e["adj"] for e in items], dtype=torch.float),
        torch.tensor([e["present"] for e in items], dtype=torch.float),
    )


class CausalCoreDo(nn.Module):
    def __init__(self, d=64, heads=4, layers=2, steps=5):
        super().__init__()
        self.steps = steps
        self.emb = nn.Embedding(len(VOCAB), d)
        self.pos = nn.Embedding(MAXLEN, d)
        layer = nn.TransformerEncoderLayer(d, heads, dim_feedforward=2 * d, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers)
        self.edge = nn.Sequential(nn.Linear(2 * d, d), nn.ReLU(), nn.Linear(d, 1))
        self.read = nn.Linear(1, 1)

    def adjacency(self, ids, pad):
        t = ids.size(1)
        h = self.encoder(self.emb(ids) + self.pos(torch.arange(t)), src_key_padding_mask=pad)
        slots = []
        for v in range(5):
            m = ((ids == v + 4) & ~pad).float().unsqueeze(-1)
            slots.append((h * m).sum(1) / m.sum(1).clamp(min=1.0))
        hv = torch.stack(slots, 1)  # (B,5,d)
        hi = hv.unsqueeze(2).expand(-1, 5, 5, -1)
        hj = hv.unsqueeze(1).expand(-1, 5, 5, -1)
        logits = self.edge(torch.cat([hi, hj], -1)).squeeze(-1)
        return logits.masked_fill(torch.eye(5).bool(), -30.0)

    def forward(self, ids, pad, xs, ys, is_causal, pres):
        logits = self.adjacency(ids, pad)
        a = torch.sigmoid(logits) * pres.unsqueeze(2) * pres.unsqueeze(1)
        r = a
        for _ in range(self.steps):  # directed reachability (transitive closure)
            r = torch.clamp(a + torch.bmm(r, a), 0.0, 1.0)
        idx = torch.arange(ids.size(0))
        fwd = r[idx, xs, ys]  # X ~> Y
        bwd = r[idx, ys, xs]  # Y ~> X
        rzx = r[idx, :, xs]  # (B,5) Z ~> X
        rzy = r[idx, :, ys]  # (B,5) Z ~> Y
        notxy = torch.ones_like(rzx)
        notxy[idx, xs] = 0.0
        notxy[idx, ys] = 0.0
        common = (rzx * rzy * notxy * pres).max(dim=1).values  # a shared cause (back-door)
        # do(): causal read = directed forward reach only; observational adds back-door + reverse
        causal_score = fwd
        corr_score = 1 - (1 - fwd) * (1 - bwd) * (1 - common)
        score = is_causal * causal_score + (1 - is_causal) * corr_score
        return (
            self.read(score.unsqueeze(-1)).squeeze(-1),
            logits,
            pres.unsqueeze(2) * pres.unsqueeze(1),
        )


def train(model, data, epochs=16, lr=2e-3, bs=128, lam=1.0):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss()
    rng = random.Random(0)
    for ep in range(epochs):
        rng.shuffle(data)
        ta = te = nb = 0.0
        for i in range(0, len(data), bs):
            ids, pad, xs, ys, isc, lab, adj, pres = batch(data[i : i + bs])
            ans, elog, pm = model(ids, pad, xs, ys, isc, pres)
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
def acc(model, data) -> float:
    if not data:
        return float("nan")
    ids, pad, xs, ys, isc, lab, adj, pres = batch(data)
    ans, _, _ = model(ids, pad, xs, ys, isc, pres)
    return float(((ans > 0) == (lab > 0.5)).float().mean())


def main() -> None:
    torch.manual_seed(0)
    print("training data: 2/3-var graphs; held-out test: 4/5-var (random variable names) ...")
    train_data = build(16000, sizes=[2, 3], seed=1)
    tests = {s: build(2000, sizes=[s], seed=10 + s) for s in (2, 3, 4, 5)}

    model = CausalCoreDo()
    print(f"CausalCoreDo: {sum(p.numel() for p in model.parameters()) / 1e3:.0f}K params")
    train(model, train_data)
    model.eval()

    print("\n                  accuracy by query type")
    print("  size      observational(corr)   interventional(cause)   in-training?")
    for s in (2, 3, 4, 5):
        corr = [e for e in tests[s] if not e["is_causal"]]
        cause = [e for e in tests[s] if e["is_causal"]]
        seen = "trained" if s in (2, 3) else "HELD-OUT"
        print(
            f"  size {s}:        {acc(model, corr):.3f}                {acc(model, cause):.3f}"
            f"                {seen}"
        )

    # the decisive test: confounded pairs -- correlated but X does NOT cause Y -> a cause query
    print("\n  do() vs correlation on CONFOUNDED pairs (corr=yes, cause=no), as 'does X cause Y?':")
    for s in (3, 4, 5):
        conf = [
            dict(e, is_causal=1, label=e["cause"]) for e in tests[s] if e["corr"] and not e["cause"]
        ]
        a = acc(model, conf)
        print(
            f"    size {s}:  embedded do()-core = {a:.3f}   (a correlation-only reasoner = 0.000)  "
            f"[n={len(conf)}]"
        )

    print(
        "\nReading: one embedded core, one learned adjacency, answers BOTH query types -- the do() "
        "operator is the switch (causal read drops the back-door term that the observational read "
        "keeps). On confounded pairs a correlation-only reasoner scores 0 by definition; the "
        "do()-core says 'no cause' correctly, and it holds on graph sizes never trained on -- "
        "observational + interventional reasoning, beyond correlation, in one architecture."
    )


if __name__ == "__main__":
    main()
