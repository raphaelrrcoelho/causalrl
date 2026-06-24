# STATUS: canonical · Act 2 Mechanism — interventions break the observational ceiling (0.82->0.99)  ·  map: CAUSAL_LLM.md
"""Beyond correlations: interventional evidence breaks the observational (MEC) ceiling.

Everything so far was observational -- premises were correlations / conditional independencies, so the
best possible accuracy was the Markov-equivalence-class ceiling (~0.82): causal *direction* is only
identifiable up to the MEC from observation alone. The defining move of a *causal* model -- what takes
it beyond correlation -- is using interventions (do): an experiment that sets a variable and watches
what changes orients edges observation cannot.

This is the decisive test. Same question ("does X cause Y?"), two kinds of evidence, same tiny model:

  * OBSERVATIONAL  the CPDAG (skeleton + only MEC-invariant orientations) -> answer   ceiling = MEC
  * INTERVENTIONAL the fully-oriented DAG (what do()-experiments reveal)   -> answer   ceiling = 1.0

Both require reachability reasoning over the structure. The point is the EVIDENCE: on the subset of
queries that are **MEC-ambiguous** (observation provably cannot decide the direction), the
observational model is stuck near chance while the interventional model answers them -- exceeding the
observational ceiling. That gap is "going beyond correlations", and the architectural mandate for a
causal LM: ingest / seek interventional evidence, do not reason from correlation alone.

Ground truth (CPDAG, DAG, MEC membership, ambiguity) is computed from causalrl.

CPU-sized.  Run::

    uv run --extra torch python examples/causal_beyond_correlation.py
"""

from __future__ import annotations

import os
import random
import sys

import torch

import causalrl as C
from causalrl.identification._separation import d_separated

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import causal_reasoning_curriculum as cur  # full-MEC index + specs
import causal_reasoning_scaffold as scf  # cpdag_str, build_model, acc_teacher_forced
import causal_transfer_corr2cause as c2c  # tokenizer, train, baselines

YES_TOK, NO_TOK = c2c.YES_TOK, c2c.NO_TOK

torch.set_num_threads(4)


def dag_str(g: C.CausalGraph, nodes: list[str]) -> str:
    """The fully-oriented edge set an intervention regime reveals: 'i > j' per directed edge."""
    edges = [f"{i} > {j}" for i in nodes for j in sorted(g.children(i))]
    return " ".join(edges) if edges else "none"


def build(n_examples: int, n_vars: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    nodes = c2c.LETTERS[:n_vars]
    specs = cur.full_specs(nodes)
    index = cur.fp_index(nodes)
    want = n_examples // 2
    cnt = {True: 0, False: 0}
    out: list[dict] = []
    tries = 0
    while len(out) < 2 * want and tries < n_examples * 400:
        tries += 1
        g = c2c.random_dag(nodes, p=0.5, rng=rng)
        x, y = rng.sample(nodes, 2)
        label = y in g.descendants(x)
        if cnt[label] >= want:
            continue
        members = index[cur.fingerprint(g, specs)]
        votes = {y in m.descendants(x) for m in members}
        out.append(
            {
                "nodes": nodes,
                "x": x,
                "y": y,
                "label": label,
                "cpdag": scf.cpdag_str(nodes, members),  # observational identification (MEC)
                "dag": dag_str(g, nodes),  # interventional identification (oriented)
                "ambiguous": len(votes) > 1,  # observation cannot decide this query
                "oracle": sum(m.descendants(x).__contains__(y) for m in members)
                >= len(members) / 2,
            }
        )
        cnt[label] += 1
    rng.shuffle(out)
    return out


def p_obs(e: dict) -> str:
    return f"observed {e['cpdag']} . does {e['x']} cause {e['y']} ?"


def p_int(e: dict) -> str:
    return f"experiments {e['dag']} . does {e['x']} cause {e['y']} ?"


def text_obs(e: dict) -> str:
    return p_obs(e) + (YES_TOK if e["label"] else NO_TOK)


def text_int(e: dict) -> str:
    return p_int(e) + (YES_TOK if e["label"] else NO_TOK)


def main() -> None:
    torch.manual_seed(0)
    print("generating observational (CPDAG) and interventional (DAG) views of the same queries ...")
    train_data = build(9000, n_vars=3, seed=1)
    val_data = build(2000, n_vars=3, seed=2)
    amb = [e for e in val_data if e["ambiguous"]]
    det = [e for e in val_data if not e["ambiguous"]]
    print(f"  val: {len(val_data)} ({len(amb)} MEC-ambiguous, {len(det)} MEC-determined)")
    print(f"  e.g. obs: {p_obs(train_data[0])!r}\n       int: {p_int(train_data[0])!r}")

    tok = c2c.build_tokenizer([text_obs(d) for d in train_data] + [text_int(d) for d in train_data])
    yes_id, no_id = tok.convert_tokens_to_ids(YES_TOK), tok.convert_tokens_to_ids(NO_TOK)

    print("\ntraining OBSERVATIONAL model (CPDAG -> answer) ...")
    obs = scf.build_model(tok)
    c2c.train(obs, tok, [text_obs(d) for d in train_data], epochs=12)
    obs.eval()

    print("\ntraining INTERVENTIONAL model (oriented DAG -> answer) ...")
    inv = scf.build_model(tok)
    c2c.train(inv, tok, [text_int(d) for d in train_data], epochs=12)
    inv.eval()

    mec = sum(e["oracle"] == e["label"] for e in val_data) / len(val_data)
    print("\n                                  answer accuracy")
    rows = [("all queries", val_data), ("MEC-ambiguous (obs cannot)", amb), ("MEC-determined", det)]
    for name, data in rows:
        o = scf.acc_teacher_forced(obs, tok, data, p_obs, yes_id, no_id)
        i = scf.acc_teacher_forced(inv, tok, data, p_int, yes_id, no_id)
        print(f"  {name:28s}  observational={o:.3f}   interventional={i:.3f}")
    print(f"\n  observational ceiling (MEC oracle, all queries): {mec:.3f}")
    print("  interventional ceiling (full orientation): 1.000")

    print(
        "\nReading: the observational model is capped at the MEC ceiling and, on MEC-AMBIGUOUS "
        "queries, sits near chance -- correlation literally cannot orient those edges. The "
        "interventional model answers them (toward 1.0), exceeding the observational ceiling. That "
        "gap is causal reasoning *beyond correlation*: a causal LM must use interventional "
        "evidence, not reason from correlations alone."
    )


if __name__ == "__main__":
    main()
