# STATUS: support · cross-cutting — multi-seed robustness of load-bearing claims  ·  map: CAUSAL_LLM.md
"""Robustness: multi-seed mean +/- std for the load-bearing claims of the causal-LM arc.

Every headline so far was a single seed. This re-runs the three most load-bearing (and most
surprising) results across several seeds and reports mean +/- std, so the claims are not artefacts:

  (a) HYBRID vs VANILLA GPT-2 on confounded prose (correlated but not causal) -- the capstone.
  (b) ACTIVE vs RANDOM intervention selection at a fixed budget -- active discovery.
  (c) DISCOVERY from evidence, observational vs interventional, on the confounded subset -- the
      Markov-equivalence ceiling and how interventions break it.

Configs are reduced vs the standalone scripts (fewer epochs / smaller data) so the multi-seed sweep
fits on CPU; absolute numbers may be a touch lower, but the gaps -- which are the claims -- are what
we test for stability.

CPU-sized (slow: it trains several models per seed).  Run::

    uv run --extra torch python examples/causal_robustness.py
"""

from __future__ import annotations

import os
import random
import statistics
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import causal_active_discovery as ad
import causal_core_discovery as disc
import causal_hybrid_lm as hy

SEEDS = [0, 1, 2]


def confounded(data):
    return [dict(e, is_causal=1, label=e["cause"]) for e in data if e["corr"] and not e["cause"]]


def hybrid_seed(seed):
    torch.manual_seed(seed)
    train = hy.build(8000, sizes=[2, 3], seed=seed)
    t3 = hy.build(1500, sizes=[3], seed=seed + 50)
    t4 = hy.build(1500, sizes=[4], seed=seed + 60)
    van = hy.VanillaLM()
    hy.train(van, train, hybrid=False, epochs=12)
    van.eval()
    hyb = hy.HybridLM()
    hy.train(hyb, train, hybrid=True, epochs=12)
    hyb.eval()
    return {
        "vanilla_conf_s3": hy.acc(van, confounded(t3), False),
        "hybrid_conf_s3": hy.acc(hyb, confounded(t3), True),
        "vanilla_conf_s4": hy.acc(van, confounded(t4), False),
        "hybrid_conf_s4": hy.acc(hyb, confounded(t4), True),
    }


def active_seed(seed, n=6, n_graphs=300, budget=2):
    rng = random.Random(seed)
    scorer = ad.train_scorer(n, random.Random(seed + 7), steps=1500)
    agg = {p: 0.0 for p in ("random", "active", "learned")}
    floor = used = 0.0
    for _ in range(n_graphs):
        adj = ad.gen_dag(n, 0.4, rng)
        _, und = ad.cpdag(adj, n)
        if len(und) < 2:
            continue
        used += 1
        for p in ("random", "active", "learned"):
            accs = ad.rollout(adj, n, n, p, rng, scorer)
            agg[p] += accs[budget]
        floor += ad.rollout(adj, n, 0, "random", rng)[0]
    return {
        "active_b2": agg["active"] / used,
        "random_b2": agg["random"] / used,
        "learned_b2": agg["learned"] / used,
        "obs_floor": floor / used,
    }


def discovery_seed(seed):
    torch.manual_seed(seed)
    train = disc.build(12000, sizes=[2, 3], seed=seed)
    t4 = disc.build(2000, sizes=[4], seed=seed + 60)
    model = disc.DiscoveryCausalCore()
    disc.train(model, train, epochs=18)
    model.eval()
    obs = confounded([e for e in t4 if e["interventional"] == 0])
    inv = confounded([e for e in t4 if e["interventional"] == 1])
    o, _ = disc.evaluate(model, obs)
    i, _ = disc.evaluate(model, inv)
    return {"obs_conf_s4": o, "int_conf_s4": i}


def summarize(name, rows):
    print(f"\n{name}  (mean +/- std over {len(rows)} seeds)")
    for key in rows[0]:
        vals = [r[key] for r in rows]
        sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
        print(
            f"  {key:18s}  {statistics.mean(vals):.3f} +/- {sd:.3f}   {[round(v, 3) for v in vals]}"
        )


def main() -> None:
    print(f"multi-seed robustness over seeds {SEEDS}\n")

    print("=== (b) active discovery: active vs random vs learned @ budget 2 ===")
    summarize("active discovery", [active_seed(s) for s in SEEDS])

    print(
        "\n=== (c) discovery from evidence: confounded-cause, observational vs interventional ==="
    )
    summarize("discovery (held-out size 4)", [discovery_seed(s) for s in SEEDS])

    print("\n=== (a) hybrid vs vanilla GPT-2: confounded-cause accuracy ===")
    summarize("hybrid vs vanilla", [hybrid_seed(s) for s in SEEDS])

    print(
        "\nReading: the gaps (hybrid >> vanilla on confounded; active >> random; interventional >> "
        "observational discovery) are what matter -- if they hold with small std across seeds, the "
        "claims of the arc are robust, not single-seed artefacts."
    )


if __name__ == "__main__":
    main()
