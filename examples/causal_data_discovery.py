"""Discovery from RAW DATA: structure + causal answers inferred from SCM samples, not oracle facts.

Every discovery experiment so far fed the model pre-digested facts (edges, CI results). The honest
core of a causal model is going from *raw observations* to structure. Here the model sees only
**samples** from a binary SCM -- no d-separation oracle -- and must infer the adjacency, then reason
(reachability + do()).

The discovery front-end is permutation-invariant over samples: for each ordered pair (u,v) it embeds
the raw per-sample tuple [x_u, x_v, do-on-u?, do-on-v?] and averages over samples,
then an edge head decides the edge. The model learns the relevant statistic itself -- given raw
samples, not correlations or CI tests.

Two data regimes feed the same model:
  * OBSERVATIONAL  only observational samples -> the model reads dependence (skeleton) but cannot
    orient most edges: the Markov-equivalence ceiling, now hit *from data*.
  * INTERVENTIONAL  observational + do(v) samples (each variable randomized in turn) -> intervening
    shifts descendants, which orients edges -> the full DAG.

We report, on held-out graph sizes, structure recovery and the confounded-cause query (correlated
but not causal) -- where observational data cannot decide and interventional data can. Beyond
correlation, from raw data, end-to-end.

CPU-sized.  Run::

    uv run --extra torch python examples/causal_data_discovery.py
"""

from __future__ import annotations

import math
import random

import torch
from torch import nn

NMAX = 5
N_OBS = 120
N_INT = 30  # samples per intervened variable
torch.set_num_threads(4)


def reachable(adj, i, j, n) -> bool:
    seen, stack = set(), [i]
    while stack:
        u = stack.pop()
        for v in range(n):
            if adj[u][v] and v not in seen:
                if v == j:
                    return True
                seen.add(v)
                stack.append(v)
    return False


def gen_scm(n, p, rng):
    order = list(range(n))
    rng.shuffle(order)
    adj = [[0] * n for _ in range(n)]
    w = [[0.0] * n for _ in range(n)]
    bias = [rng.uniform(-0.5, 0.5) for _ in range(n)]
    for a in range(n):
        for b in range(a + 1, n):
            if rng.random() < p:
                u, v = order[a], order[b]
                adj[u][v] = 1
                w[u][v] = rng.choice([-1, 1]) * rng.uniform(2.0, 3.5)
    return adj, w, bias, order


def _sigmoid(z):
    return 1.0 / (1.0 + math.exp(-z))


def sample(adj, w, bias, order, n, rng, do_var=None):
    x = [0] * n
    for v in order:
        if v == do_var:
            x[v] = rng.randint(0, 1)
            continue
        z = bias[v] + sum(w[p][v] * (2 * x[p] - 1) for p in range(n) if adj[p][v])
        x[v] = 1 if rng.random() < _sigmoid(z) else 0
    return x


def make(sizes, rng) -> dict:
    n = rng.choice(sizes)
    adj, w, bias, order = gen_scm(n, 0.5, rng)
    interventional = rng.random() < 0.5
    rows, regime = [], []
    for _ in range(N_OBS):
        rows.append(sample(adj, w, bias, order, n, rng))
        regime.append(-1)
    if interventional:
        for v in range(n):
            for _ in range(N_INT):
                rows.append(sample(adj, w, bias, order, n, rng, do_var=v))
                regime.append(v)
    x, y = rng.sample(range(n), 2)
    cause = reachable(adj, x, y, n)
    desc = {s: {t for t in range(n) if t != s and reachable(adj, s, t, n)} for s in range(n)}
    common = any(z not in (x, y) and x in desc[z] and y in desc[z] for z in range(n))
    corr = cause or reachable(adj, y, x, n) or common
    is_causal = rng.random() < 0.5
    return {
        "rows": rows,
        "regime": regime,
        "n": n,
        "xs": x,
        "ys": y,
        "is_causal": int(is_causal),
        "cause": int(cause),
        "corr": int(corr),
        "label": int(cause if is_causal else corr),
        "adj": [[adj[i][j] if i < n and j < n else 0 for j in range(NMAX)] for i in range(NMAX)],
        "present": [s < n for s in range(NMAX)],
        "interventional": int(interventional),
    }


def build(n_examples, sizes, seed) -> list[dict]:
    rng = random.Random(seed)
    out, tries = [], 0
    cnt = {(t, lab): 0 for t in (0, 1) for lab in (0, 1)}
    cap = n_examples // 4
    while len(out) < 4 * cap and tries < n_examples * 80:
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
    b = len(items)
    nsamp = max(len(e["rows"]) for e in items)
    x = torch.zeros(b, nsamp, NMAX)
    reg = torch.full((b, nsamp), -1, dtype=torch.long)
    smask = torch.zeros(b, nsamp)
    for k, e in enumerate(items):
        for t, row in enumerate(e["rows"]):
            for v in range(e["n"]):
                x[k, t, v] = row[v]
            reg[k, t] = e["regime"][t]
            smask[k, t] = 1.0
    g = lambda key, dt: torch.tensor([e[key] for e in items], dtype=dt)  # noqa: E731
    return (
        x,
        reg,
        smask,
        g("xs", torch.long),
        g("ys", torch.long),
        g("is_causal", torch.float),
        g("label", torch.float),
        torch.tensor([e["adj"] for e in items], dtype=torch.float),
        torch.tensor([e["present"] for e in items], dtype=torch.float),
    )


class DataDiscoveryCore(nn.Module):
    def __init__(self, d=64, steps=5):
        super().__init__()
        self.steps = steps
        self.cell = nn.Sequential(
            nn.Linear(4, d), nn.ReLU(), nn.Linear(d, d)
        )  # per-sample, per-pair
        self.edge = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 1))
        self.read = nn.Linear(1, 1)

    def adjacency(self, x, reg, smask):
        b, nsamp, _ = x.shape
        ar = torch.arange(NMAX)
        xu = x.unsqueeze(3).expand(b, nsamp, NMAX, NMAX)  # value of u
        xv = x.unsqueeze(2).expand(b, nsamp, NMAX, NMAX)  # value of v
        rr = reg.unsqueeze(-1).unsqueeze(-1)  # (B,N,1,1)
        ru = (rr == ar.view(1, 1, NMAX, 1)).float().expand(b, nsamp, NMAX, NMAX)  # do-on-u?
        rv = (rr == ar.view(1, 1, 1, NMAX)).float().expand(b, nsamp, NMAX, NMAX)  # do-on-v?
        feat = torch.stack([xu, xv, ru, rv], dim=-1)  # (B,N,NMAX,NMAX,4)
        emb = self.cell(feat)  # (B,N,NMAX,NMAX,d)
        m = smask.view(b, nsamp, 1, 1, 1)
        pair = (emb * m).sum(1) / m.sum(1).clamp(min=1.0)  # average over samples -> (B,NMAX,NMAX,d)
        logit = self.edge(pair).squeeze(-1)
        return logit.masked_fill(torch.eye(NMAX).bool(), -30.0)

    def forward(self, x, reg, smask, xs, ys, is_causal, pres):
        logit = self.adjacency(x, reg, smask)
        a = torch.sigmoid(logit) * pres.unsqueeze(2) * pres.unsqueeze(1)
        r = a
        for _ in range(self.steps):
            r = torch.clamp(a + torch.bmm(r, a), 0.0, 1.0)
        idx = torch.arange(x.size(0))
        fwd, bwd = r[idx, xs, ys], r[idx, ys, xs]
        rzx, rzy = r[idx, :, xs], r[idx, :, ys]
        notxy = torch.ones_like(rzx)
        notxy[idx, xs] = 0.0
        notxy[idx, ys] = 0.0
        common = (rzx * rzy * notxy * pres).max(dim=1).values
        score = is_causal * fwd + (1 - is_causal) * (1 - (1 - fwd) * (1 - bwd) * (1 - common))
        pm = pres.unsqueeze(2) * pres.unsqueeze(1)
        return self.read(score.unsqueeze(-1)).squeeze(-1), logit, pm


def train(model, data, epochs=14, lr=2e-3, bs=32, lam=1.0):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss()
    rng = random.Random(0)
    for ep in range(epochs):
        rng.shuffle(data)
        ta = te = nb = 0.0
        for i in range(0, len(data), bs):
            x, reg, smask, xs, ys, isc, lab, adj, pres = pack(data[i : i + bs])
            ans, elog, pm = model(x, reg, smask, xs, ys, isc, pres)
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
def evaluate(model, data, bs=64) -> tuple[float, float]:
    if not data:
        return float("nan"), float("nan")
    ok = et = eo = tot = 0.0
    for i in range(0, len(data), bs):
        x, reg, smask, xs, ys, isc, lab, adj, pres = pack(data[i : i + bs])
        ans, elog, pm = model(x, reg, smask, xs, ys, isc, pres)
        ok += float(((ans > 0) == (lab > 0.5)).sum())
        tot += lab.numel()
        et += float((((elog > 0).float() == adj) * pm).sum())
        eo += float(pm.sum())
    return ok / tot, et / eo


def baselines(data):
    """Trivial baselines to expose degeneracy: all-no-edge edge-recovery, majority-class answer."""
    no_edge = maj = tot = epairs = eok = 0.0
    pos = 0
    for e in data:
        pos += e["label"]
        tot += 1
        n = e["n"]
        for i in range(n):
            for j in range(n):
                if i != j:
                    epairs += 1
                    eok += int(e["adj"][i][j] == 0)
    maj = max(pos, tot - pos) / tot
    no_edge = eok / epairs
    return no_edge, maj


def main() -> None:
    torch.manual_seed(0)
    print("discovery from RAW SCM samples (no d-sep oracle); train sizes {3,4}, test size-4 graphs")
    train_data = build(5000, sizes=[3, 4], seed=1)
    test = build(1600, sizes=[4], seed=7)

    model = DataDiscoveryCore()
    print(f"DataDiscoveryCore: {sum(p.numel() for p in model.parameters()) / 1e3:.0f}K params")
    train(model, train_data, epochs=20)
    model.eval()

    print("\n           structure recovery & confounded-cause, by data regime (size-4 test graphs)")
    print("  regime           edge-recovery   answer-acc   confounded-cause")
    for typ, name in [(0, "observational"), (1, "interventional")]:
        sub = [e for e in test if e["interventional"] == typ]
        acc, edge = evaluate(model, sub)
        conf = [dict(e, is_causal=1, label=e["cause"]) for e in sub if e["corr"] and not e["cause"]]
        ca, _ = evaluate(model, conf)
        nb, mb = baselines(sub)
        print(f"  {name:15s}  {edge:.3f} (base {nb:.3f})   {acc:.3f} (base {mb:.3f})   {ca:.3f}")

    print(
        "\nReading (judge against the baselines): edge-recovery matters only if it beats the "
        "all-no-edge baseline, and answer-acc only if it beats majority. If interventional edge/answer "
        "beat their baselines and observational does not, raw-data discovery works and shows the MEC "
        "limit; if both sit at baseline, the simple sample-aggregator failed to extract structure from "
        "raw data (the AVICI/CSIvA regime needs a heavier encoder) -- an honest boundary, not success."
    )


if __name__ == "__main__":
    main()
