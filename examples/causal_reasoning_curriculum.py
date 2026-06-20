"""(2) Making causal-structure reasoning size-general with a multi-size curriculum.

The scaffold/install results showed the model can reason over a *given* causal structure (CPDAG ->
answer ~0.82) but only at the trained graph size: a struct model trained on 3-variable graphs
collapses on 4-variable graphs (~0.57). This tests whether a **curriculum of sizes** fixes it.

Two struct-only models (CPDAG -> answer; the format is size-agnostic -- variable letters and edges):

    * SINGLE   trained on 3-variable graphs only
    * CURRIC   trained on 2- and 3-variable graphs

Both are then evaluated on **4-variable graphs, which neither saw in training** (held-out, larger).
If CURRIC >> SINGLE on size 4, seeing multiple sizes teaches a size-general reasoning procedure that
*extrapolates* to a larger unseen graph -- attacking the OOD wall head-on.

Ground truth (CPDAG, label, MEC ceiling) is computed from causalrl with the *full* conditional-
independence oracle, so the CPDAG is the exact identifiable structure at every size.

CPU-sized.  Run::

    uv run --extra torch python examples/causal_reasoning_curriculum.py
"""

from __future__ import annotations

import itertools
import os
import random
import sys

import networkx as nx
import torch

import causalrl as C
from causalrl.identification._separation import d_separated

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import causal_reasoning_scaffold as scf  # sibling examples, imported after sys.path tweak
import causal_transfer_corr2cause as c2c

YES_TOK, NO_TOK = c2c.YES_TOK, c2c.NO_TOK
LETTERS = ["A", "B", "C", "D", "E"]

torch.set_num_threads(4)


def full_specs(nodes: list[str]):
    """All conditional-independence tests (every conditioning subset) -> exact Markov class."""
    specs = []
    for i, j in itertools.combinations(nodes, 2):
        others = [k for k in nodes if k not in (i, j)]
        for r in range(len(others) + 1):
            for z in itertools.combinations(others, r):
                specs.append((i, j, frozenset(z)))
    return specs


def fingerprint(graph, specs) -> tuple[bool, ...]:
    return tuple(d_separated(graph, {i}, {j}, set(z)) for i, j, z in specs)


_INDEX: dict[tuple[str, ...], dict] = {}


def fp_index(nodes: list[str]) -> dict:
    key = tuple(nodes)
    if key not in _INDEX:
        specs = full_specs(nodes)
        pairs = [(a, b) for a in nodes for b in nodes if a != b]
        idx: dict = {}
        for mask in range(2 ** len(pairs)):
            edges = [pairs[k] for k in range(len(pairs)) if mask >> k & 1]
            dg = nx.DiGraph(edges)
            dg.add_nodes_from(nodes)
            if nx.is_directed_acyclic_graph(dg):
                g = C.CausalGraph(directed_edges=edges, nodes=nodes)
                idx.setdefault(fingerprint(g, specs), []).append(g)
        _INDEX[key] = idx
    return _INDEX[key]


def build(n_examples: int, sizes: list[int], seed: int) -> list[dict]:
    """Balanced struct-only examples (CPDAG -> answer) across the given graph sizes."""
    rng = random.Random(seed)
    for s in sizes:
        fp_index(LETTERS[:s])  # warm caches
    want = n_examples // 2
    cnt = {True: 0, False: 0}
    out: list[dict] = []
    tries = 0
    while len(out) < 2 * want and tries < n_examples * 400:
        tries += 1
        # random letter subset (not LETTERS[:k]) so every variable name appears at every size --
        # removes the "new token at test size" confound, isolating pure size-extrapolation.
        nodes = sorted(rng.sample(LETTERS, rng.choice(sizes)))
        specs = full_specs(nodes)
        g = c2c.random_dag(nodes, p=0.45, rng=rng)
        x, y = rng.sample(nodes, 2)
        label = y in g.descendants(x)
        if cnt[label] >= want:
            continue
        members = fp_index(nodes)[fingerprint(g, specs)]
        out.append(
            {
                "nodes": nodes,
                "x": x,
                "y": y,
                "label": label,
                "cpdag": scf.cpdag_str(nodes, members),
                "corr": not d_separated(g, {x}, {y}, set()),
                "oracle": sum(y in m.descendants(x) for m in members) >= len(members) / 2,
            }
        )
        cnt[label] += 1
    rng.shuffle(out)
    return out


def baselines(data: list[dict]) -> dict[str, float]:
    n = len(data)
    maj = max(sum(d["label"] for d in data), n - sum(d["label"] for d in data)) / n
    corr = sum(d["corr"] == d["label"] for d in data) / n
    oracle = sum(d["oracle"] == d["label"] for d in data) / n
    return {"majority": maj, "corr": corr, "MEC": oracle}


def main() -> None:
    torch.manual_seed(0)
    print("generating struct-only data (sizes 2/3 for training, 4 held-out) ...")
    single_data = build(8000, sizes=[3], seed=1)
    curric_data = build(8000, sizes=[2, 3], seed=1)
    test3 = build(1500, sizes=[3], seed=2)
    test4 = build(1500, sizes=[4], seed=3)  # held-out larger size for BOTH models

    tok = c2c.build_tokenizer(
        [scf.text_struct_only(d) for d in curric_data + single_data + test4[:500]]
    )
    yes_id, no_id = tok.convert_tokens_to_ids(YES_TOK), tok.convert_tokens_to_ids(NO_TOK)

    print("\ntraining SINGLE-size struct model (sizes {3}) ...")
    single = scf.build_model(tok)
    c2c.train(single, tok, [scf.text_struct_only(d) for d in single_data], epochs=12)
    single.eval()

    print("\ntraining CURRICULUM struct model (sizes {2,3}) ...")
    curric = scf.build_model(tok)
    c2c.train(curric, tok, [scf.text_struct_only(d) for d in curric_data], epochs=12)
    curric.eval()

    print("\n                              answer accuracy (reason over a given CPDAG)")
    for name, data in [("size 3 (in-dist)", test3), ("size 4 (HELD-OUT)", test4)]:
        b = baselines(data)
        s = scf.acc_teacher_forced(single, tok, data, scf.p_struct_only, yes_id, no_id)
        c = scf.acc_teacher_forced(curric, tok, data, scf.p_struct_only, yes_id, no_id)
        print(
            f"  {name:18s}  single(3)={s:.3f}   curric(2,3)={c:.3f}   "
            f"corr={b['corr']:.3f}   MEC={b['MEC']:.3f}"
        )

    print(
        "\nReading: both models are tested on size-4 graphs neither saw. If curric(2,3) >> single(3) "
        "on the held-out size 4 (toward the MEC ceiling), training on multiple sizes teaches a "
        "size-general reasoning procedure that extrapolates to a larger unseen graph -- closing the "
        "OOD wall for the reason-over-structure half."
    )


if __name__ == "__main__":
    main()
