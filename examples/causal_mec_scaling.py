# STATUS: canonical · Act 6 Frontier — Phase 2c/B2 size-extrapolation on a causalrl-generated benchmark: the size-agnostic GNN trained on N<=5 extrapolates to N=6..9, tested against a FAIR size-agnostic graph-transformer baseline (B2) and a fixed-size MLP strawman; dogfoods causalrl discovery (Meek rules + d-separation)  ·  map: CAUSAL_LLM.md
"""Phase 2c — a second, controlled testbed for the decoupling thesis: SIZE EXTRAPOLATION.

Corr2Cause caps at 6 variables (a complete-CI premise is exponential, so the benchmark itself stops
there). To probe breadth -- does the decoupled reasoner generalize across graph SIZES it never saw? --
we generate our own structures with ``causalrl`` and dogfood the library for ground truth:

  * Random DAGs at N = 4..9 (controlled size).
  * The reasoner's INPUT is the PC-style structure (skeleton S + unshielded-collider evidence D),
    exactly as in the Corr2Cause experiment.
  * The LABEL is a Markov-equivalence-INVARIANT query -- "is X a DEFINITE ancestor of Y?" (a directed
    path in the true CPDAG) -- so it is well-posed from the structure. Ground truth comes from
    causalrl's own Meek-rule orientation (``causalrl.discovery``); we also assert causalrl's
    ``d_separated`` matches networkx (dogfooding the library as the authority).

Three reasoners, trained ONLY on N in {4,5}, evaluated per size 4..9:
  * size-agnostic GNN (message passing -- the Phase-2 reasoner, ``make_gnn``);
  * (B2) a FAIR size-agnostic graph transformer (``make_transformer``): self-attention over the
    variable tokens with the structure injected as attention bias -- it also handles any N, so unlike
    the MLP it *can* extrapolate; the honest baseline the earlier MLP-only comparison lacked;
  * a fixed-size MLP on the padded (S,D,X,Y) tensor (the size-tied strawman lower bound).

Thesis (the size leg of decoupling): the GNN's local message-passing extrapolates to larger graphs
better than generic global attention, and far better than the size-tied MLP. This complements the
Corr2Cause size-shift (train 96% N=6).

Honest scope: synthetic structures, MEC-invariant query, small models, CPU. The point is the
size-extrapolation *gap*, not an absolute SOTA number.

Run::

    SMOKE=1 uv run --extra torch python examples/causal_mec_scaling.py
    uv run --extra torch python examples/causal_mec_scaling.py
"""

from __future__ import annotations

import os
import random
import sys
from itertools import combinations

import networkx as nx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from causal_corr2cause_learned import f1, make_gnn

SEED = int(os.environ.get("SEED", "0"))
SMOKE = os.environ.get("SMOKE", "") not in ("", "0", "false")
NMAX = 9
TRAIN_SIZES = [4, 5]
TEST_SIZES = [4, 5, 6, 7, 8, 9]
GRAPHS_PER_SIZE = int(os.environ.get("GRAPHS_PER_SIZE", "1500"))
EDGE_P = float(os.environ.get("EDGE_P", "0.4"))
EPOCHS = int(os.environ.get("EPOCHS", "40"))
if SMOKE:
    GRAPHS_PER_SIZE, EPOCHS = 300, 12


# --------------------------------------------------------------------------- generation + causalrl ground truth
def gen_dag(n, rng):
    perm = list(range(n))
    rng.shuffle(perm)
    g = nx.DiGraph()
    g.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < EDGE_P:
                g.add_edge(perm[i], perm[j])  # edge follows a random topological order -> acyclic
    return g


def true_cpdag_directed(dag):
    """Directed edges of the true CPDAG, via causalrl's own collider-orient + Meek rules (dogfood)."""
    from causalrl.discovery import _apply_meek_rules, _orient

    nodes = list(dag.nodes)
    skel = {frozenset(e) for e in dag.edges}
    directed: set[tuple[int, int]] = set()
    undirected = set(skel)
    for c in nodes:  # orient unshielded colliders a -> c <- b
        for a, b in combinations(sorted(dag.predecessors(c)), 2):
            if frozenset((a, b)) not in skel:
                _orient(a, c, directed, undirected)
                _orient(b, c, directed, undirected)
    _apply_meek_rules(nodes, directed, undirected)
    return directed


def structure(dag, n):
    """Skeleton S + unshielded-collider evidence D (NMAX-padded) — the reasoner's input."""
    import numpy as np

    S = np.zeros((NMAX, NMAX), dtype="float32")
    D = np.zeros((NMAX, NMAX), dtype="float32")
    present = np.zeros(NMAX, dtype="float32")
    present[:n] = 1.0
    skel = {frozenset(e) for e in dag.edges}
    for a, b in dag.edges:
        S[a, b] = S[b, a] = 1.0
    for c in dag.nodes:
        for a, b in combinations(sorted(dag.predecessors(c)), 2):
            if frozenset((a, b)) not in skel:
                D[a, c] = D[b, c] = 1.0
    return S, D, present


def to_tensors(rows):
    import numpy as np
    import torch

    S = torch.tensor(np.stack([r[0] for r in rows]))
    D = torch.tensor(np.stack([r[1] for r in rows]))
    present = torch.tensor(np.stack([r[2] for r in rows]))
    xi = torch.tensor([r[3] for r in rows], dtype=torch.long)
    yi = torch.tensor([r[4] for r in rows], dtype=torch.long)
    nf = torch.zeros(len(rows), NMAX, 2)
    for i, r in enumerate(rows):
        nf[i, r[3], 0] = 1.0
        nf[i, r[4], 1] = 1.0
    tpl = torch.zeros(len(rows), dtype=torch.long)
    return S, D, nf, present, tpl, xi, yi


# --------------------------------------------------------------------------- fixed-size MLP control
def make_mlp(d=128):
    import torch
    from torch import nn

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(2 * NMAX * NMAX + 2 * NMAX, d), nn.ReLU(),
                nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 1))

        def forward(self, S, D, nf, present, tpl, xi, yi):
            b = S.size(0)
            feat = torch.cat([S.reshape(b, -1), D.reshape(b, -1), nf[..., 0], nf[..., 1]], -1)
            return self.net(feat).squeeze(-1)

    return MLP()


# --------------------------------------------------------------------------- B2: fair size-capable baseline (graph transformer)
def make_transformer(d=64, layers=3, heads=4):
    """A size-agnostic baseline that is NOT a GNN: self-attention over the variable tokens, with the
    structure (S, D) injected as an additive per-head attention bias, and absent variables masked. It
    handles any N (unlike the fixed-size MLP), so it *can* extrapolate — a fair test of whether the
    GNN's local message-passing bias beats generic global attention out of distribution."""
    import torch
    from torch import nn

    class GraphTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.dh = d // heads
            self.in_proj = nn.Linear(2, d)          # node features: X / Y one-hot
            self.edge_bias = nn.Linear(3, heads)    # [S_ij, D_ij, D_ji] -> per-head attention bias
            self.q = nn.ModuleList(nn.Linear(d, d) for _ in range(layers))
            self.k = nn.ModuleList(nn.Linear(d, d) for _ in range(layers))
            self.v = nn.ModuleList(nn.Linear(d, d) for _ in range(layers))
            self.o = nn.ModuleList(nn.Linear(d, d) for _ in range(layers))
            self.ff = nn.ModuleList(
                nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d)) for _ in range(layers))
            self.n1 = nn.ModuleList(nn.LayerNorm(d) for _ in range(layers))
            self.n2 = nn.ModuleList(nn.LayerNorm(d) for _ in range(layers))
            self.head = nn.Sequential(nn.Linear(2 * d, d), nn.ReLU(), nn.Linear(d, 1))

        def forward(self, S, D, nf, present, tpl, xi, yi):
            b, n = present.shape
            h = self.in_proj(nf)                                    # (b,n,d)
            ef = torch.stack([S, D, D.transpose(1, 2)], -1)         # (b,n,n,3): skel + both v-struct dirs
            bias = self.edge_bias(ef).permute(0, 3, 1, 2)          # (b,heads,n,n)
            kmask = (present[:, None, None, :] == 0)                # mask absent variables as keys
            for li in range(layers):
                hn = self.n1[li](h)
                q = self.q[li](hn).view(b, n, heads, self.dh).transpose(1, 2)
                k = self.k[li](hn).view(b, n, heads, self.dh).transpose(1, 2)
                v = self.v[li](hn).view(b, n, heads, self.dh).transpose(1, 2)
                att = (q @ k.transpose(-1, -2)) / (self.dh ** 0.5) + bias
                att = att.masked_fill(kmask, float("-inf")).softmax(-1).nan_to_num(0.0)
                out = (att @ v).transpose(1, 2).reshape(b, n, d)
                h = h + self.o[li](out)
                h = h + self.ff[li](self.n2[li](h))
            ar = torch.arange(b)
            hx, hy = h[ar, xi], h[ar, yi]                           # read out the X / Y node embeddings
            return self.head(torch.cat([hx, hy], -1)).squeeze(-1)

    return GraphTransformer()


# --------------------------------------------------------------------------- train / eval
def train(model, rows, labels):
    import torch
    from torch import nn

    torch.manual_seed(SEED)
    S, D, nf, present, tpl, xi, yi = to_tensors(rows)
    y = torch.tensor(labels, dtype=torch.float32)
    pos = float(y.sum())
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([(len(y) - pos) / max(pos, 1.0)]))
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    bs, n = 256, len(rows)
    order = list(range(n))
    rng = random.Random(SEED)
    for ep in range(EPOCHS):
        rng.shuffle(order)
        model.train()
        for i in range(0, n, bs):
            b = order[i : i + bs]
            loss = lossf(model(S[b], D[b], nf[b], present[b], tpl[b], xi[b], yi[b]), y[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model


def evf1(model, rows, labels):
    import torch

    if not rows:
        return float("nan")
    S, D, nf, present, tpl, xi, yi = to_tensors(rows)
    model.eval()
    with torch.no_grad():
        pred = (model(S, D, nf, present, tpl, xi, yi) > 0).long().tolist()
    return f1(labels, pred)[2]


# --------------------------------------------------------------------------- main
def main() -> None:
    print(f"MEC size-extrapolation (causalrl-generated)  seed={SEED}  smoke={SMOKE}  "
          f"train sizes={TRAIN_SIZES}  test={TEST_SIZES}")

    # dogfood: causalrl.d_separated == networkx on random DAGs
    from causalrl import CausalGraph
    try:
        from causalrl import d_separated
    except ImportError:
        from causalrl.identification._separation import d_separated
    rng = random.Random(SEED)
    chk = 0
    for _ in range(40):
        dag = gen_dag(6, rng)
        cg = CausalGraph([(str(a), str(b)) for a, b in dag.edges], nodes=[str(i) for i in range(6)])
        a, b = "0", "5"
        z = {str(i) for i in rng.sample(range(1, 5), rng.randint(0, 2))}
        assert nx.is_d_separator(dag, {0}, {5}, {int(t) for t in z}) == d_separated(cg, {a}, {b}, z)
        chk += 1
    print(f"  dogfood: causalrl.d_separated == networkx on {chk} DAGs ✓")

    # build train (small sizes) + per-size test sets, with definite-ancestor labels
    def make_split(sizes, seed, per):
        r = random.Random(seed)
        pos, neg = [], []
        for n in sizes:
            for _ in range(per):
                dag = gen_dag(n, r)
                directed = true_cpdag_directed(dag)
                g2 = nx.DiGraph(list(directed))
                g2.add_nodes_from(dag.nodes)
                S, D, present = structure(dag, n)
                desc = {u: nx.descendants(g2, u) for u in dag.nodes}
                for x in range(n):
                    for y in range(n):
                        if x != y:
                            (pos if y in desc[x] else neg).append((S, D, present, x, y))
        r.shuffle(pos)
        r.shuffle(neg)
        k = min(len(pos), len(neg))
        rows = pos[:k] + neg[:k]
        labels = [1] * k + [0] * k
        idx = list(range(len(rows)))
        r.shuffle(idx)
        return [rows[i] for i in idx], [labels[i] for i in idx]

    tr_rows, tr_lab = make_split(TRAIN_SIZES, SEED, GRAPHS_PER_SIZE)
    print(f"  train: {len(tr_rows)} (x,y) pairs over sizes {TRAIN_SIZES} (50/50 balanced)")

    import torch  # noqa: F401  (ensure torch present before building models)

    gnn = train(make_gnn(), tr_rows, tr_lab)
    tfm = train(make_transformer(), tr_rows, tr_lab)  # B2: fair, size-capable, non-GNN baseline
    mlp = train(make_mlp(), tr_rows, tr_lab)

    print("\n--- F1 (definite-ancestor) by graph size — TRAIN sizes {4,5}, the rest are EXTRAPOLATION ---")
    print(f"  {'reasoner':30s}  " + "  ".join(f"N={s}" for s in TEST_SIZES))
    gnn_cells, tfm_cells, mlp_cells = [], [], []
    for s in TEST_SIZES:
        te_rows, te_lab = make_split([s], SEED + 100 + s, max(GRAPHS_PER_SIZE // 3, 60))
        gnn_cells.append(f"{evf1(gnn, te_rows, te_lab):.2f}")
        tfm_cells.append(f"{evf1(tfm, te_rows, te_lab):.2f}")
        mlp_cells.append(f"{evf1(mlp, te_rows, te_lab):.2f}")
    star = ["" if s in TRAIN_SIZES else "*" for s in TEST_SIZES]
    print(f"  {'size-agnostic GNN':30s}  " + "  ".join(f"{c}{x}" for c, x in zip(gnn_cells, star)))
    print(f"  {'graph transformer (B2, fair)':30s}  " + "  ".join(f"{c}{x}" for c, x in zip(tfm_cells, star)))
    print(f"  {'fixed-size MLP (strawman)':30s}  " + "  ".join(f"{c}{x}" for c, x in zip(mlp_cells, star)))
    print("  (* = size NEVER seen in training. The GNN and the graph transformer are BOTH size-agnostic")
    print("   — this tests whether message-passing extrapolates better than global attention. The")
    print("   fixed-size MLP, tied to the train sizes/positions, is the strawman lower bound.)")


if __name__ == "__main__":
    main()
