"""Honest audit of the causal graph transformer's d-separation accuracy.

A single headline number on a class-balanced set can hide that the model only solves the easy cases.
This stratifies a *natural* (unbalanced) test set by the difficulty of the d-separation query and
reports per-stratum accuracy, so we can see whether the model actually handles the subtle cases —
above all the **collider** case, where conditioning on a collider (or its descendant) *opens* a
path. That is the case that separates real d-separation reasoning from adjacency/connectivity
shortcuts.

Strata (computed with the causalrl oracle by comparing separation with and without Z):

* ``adjacent``     — X and Y share a direct edge: never separable. Trivial negative.
* ``sep_robust``   — separated with and without Z (usually no path at all). Easy positive.
* ``conn_robust``  — connected with and without Z (Z did not help). Easy-ish negative.
* ``blocked``      — Z *blocks* a chain/fork path (separated only given Z). Medium.
* ``collider_open``— Z *opens* a path via a collider/descendant (separated only without Z). HARD.

Run::

    uv run --extra torch python examples/causal_graph_transformer_diagnose.py \
        --ckpt /tmp/cgt_real/best.pt --d-model 128 --layers 4 --sizes 6 7
"""

from __future__ import annotations

import argparse
import random
import string
from collections import Counter, defaultdict

import torch
from causal_graph_transformer import (
    CHILD,
    NONE,
    PARENT,
    PLAIN,
    ROLE_X,
    ROLE_Y,
    ROLE_Z,
    SELF,
    CausalGraphTransformer,
    Config,
    Example,
    GraphDataset,
    collate,
)
from torch.utils.data import DataLoader

from causalrl import CausalGraph
from causalrl.identification._separation import d_separated


def generate(cfg: Config, n: int, rng: random.Random, pool: list[str]) -> tuple[Example, str]:
    names = rng.sample(pool, n)
    order = names[:]
    rng.shuffle(order)
    directed = [
        (order[i], order[j])
        for i in range(n)
        for j in range(i + 1, n)
        if rng.random() < cfg.edge_prob
    ]
    graph = CausalGraph(directed_edges=directed, nodes=names)
    x, y = rng.sample(names, 2)
    rest = [v for v in names if v not in (x, y)]
    z = rng.sample(rest, rng.randint(0, min(cfg.max_cond, len(rest))))

    sep_z = d_separated(graph, {x}, {y}, set(z))
    sep_empty = d_separated(graph, {x}, {y}, set())
    adjacent = (x, y) in directed or (y, x) in directed
    if adjacent:
        stratum = "adjacent"
    elif sep_empty and not sep_z:
        stratum = "collider_open"
    elif not sep_empty and sep_z:
        stratum = "blocked"
    elif sep_empty and sep_z:
        stratum = "sep_robust"
    else:
        stratum = "conn_robust"

    idx = {name: i for i, name in enumerate(names)}
    roles = [PLAIN] * n
    roles[idx[x]] = ROLE_X
    roles[idx[y]] = ROLE_Y
    for zz in z:
        roles[idx[zz]] = ROLE_Z
    rel = [[NONE] * n for _ in range(n)]
    for i in range(n):
        rel[i][i] = SELF
    for a, b in directed:
        rel[idx[b]][idx[a]] = PARENT
        rel[idx[a]][idx[b]] = CHILD
    return Example(roles, rel, idx[x], idx[y], sep_z), stratum


def main() -> None:
    p = argparse.ArgumentParser(description="Difficulty-stratified audit of d-separation accuracy.")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--sizes", type=int, nargs="+", default=[6, 7])
    p.add_argument("--n", type=int, default=8000)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--ablate-structure", action="store_true",
                   help="set if the checkpoint was trained structure-blind")
    a = p.parse_args()

    cfg = Config(d_model=a.d_model, n_layers=a.layers, n_heads=a.heads,
                 ablate_structure=a.ablate_structure)
    device = cfg.resolved_device()
    model = CausalGraphTransformer(cfg).to(device)
    model.load_state_dict(torch.load(a.ckpt, map_location=device))
    model.eval()

    rng = random.Random(a.seed)
    pool = list(string.ascii_uppercase[: cfg.max_nodes])
    examples: list[Example] = []
    strata: list[str] = []
    for _ in range(a.n):
        ex, st = generate(cfg, rng.choice(a.sizes), rng, pool)
        examples.append(ex)
        strata.append(st)

    counts: Counter[str] = Counter(strata)
    correct: dict[str, int] = defaultdict(int)
    loader = DataLoader(GraphDataset(examples), batch_size=512, collate_fn=collate)
    preds: list[int] = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            preds.extend(model(batch).argmax(-1).tolist())
    n_pos = 0
    overall = 0
    for ex, st, pred in zip(examples, strata, preds, strict=True):
        ok = int(pred == int(ex.label))
        correct[st] += ok
        overall += ok
        n_pos += int(ex.label)

    majority = max(n_pos, a.n - n_pos) / a.n
    print(f"\nnatural (unbalanced) test set: {a.n} queries over sizes {a.sizes}")
    print(f"base rate P(d-separated)={n_pos / a.n:.3f}  ->  majority baseline {majority:.3f}")
    print(f"overall accuracy (natural): {overall / a.n:.3f}\n")
    print(f"{'stratum':<14}{'share':>8}{'n':>7}{'accuracy':>10}   (difficulty)")
    labels = {
        "adjacent": "trivial neg",
        "sep_robust": "easy pos",
        "conn_robust": "easy-ish neg",
        "blocked": "medium",
        "collider_open": "HARD",
    }
    for st in ["adjacent", "sep_robust", "conn_robust", "blocked", "collider_open"]:
        c = counts.get(st, 0)
        acc = correct[st] / c if c else float("nan")
        print(f"{st:<14}{c / a.n:>7.1%}{c:>7}{acc:>10.3f}   ({labels[st]})")
    print(
        "\nRead the collider_open row first: if it is rare AND near 0.5, the headline accuracy is "
        "carried by easy cases and the model has not learned the subtle part of d-separation."
    )


if __name__ == "__main__":
    main()
