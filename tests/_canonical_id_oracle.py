"""A brute-force, structure-only oracle for non-identifiability (issue #16).

Independent of :mod:`causalrl.identification.id_algorithm`: it searches the *canonical
response-function* SCM class (Balke & Pearl 1994; Tian & Pearl 2002) compatible with a given
ADMG for two structurally-consistent instances that induce the exact same observational law
``P(V)`` but a different interventional mean ``E[outcome | do(treatment=v)]``. Finding such a
*witness* pair is a constructive, airtight proof that the effect is not identifiable from
``P(V)`` alone — no valid formula could give the right answer for both instances.

Every bidirected EDGE gets its own shared latent feeding exactly its two endpoints (not one
latent per whole c-component -- a c-component that is a chain rather than a bidirected clique,
e.g. A<->B<->C without A<->C, is not faithfully realized by a single latent touching all three;
that over-confounds the endpoints and manufactures spurious "witnesses" for effects that are
genuinely identifiable). Every node also gets a private per-node noise bit as an extra
response-function input, so the search only counts a witness when the matching ``P(V)`` has full
support — degenerate (zero-probability-cell) matches are excluded, since several identification
formulas (e.g. front-door) involve conditionals that are undefined off a degenerate
distribution's support and would otherwise look like spurious counterexamples.

The search is necessarily incomplete on the "no witness found" side (absence of a witness is not
proof of identifiability — the response-function/probability grid is finite) but always *sound*
on the "witness found" side. Restricted to small ADMGs (<=4 nodes, <=3 total parents per node
including latents) to keep the response-function enumeration tractable; larger graphs return
``"TOO_BIG"`` and should be skipped by the caller.
"""

from __future__ import annotations

import random
from collections import defaultdict
from itertools import product
from typing import Any

from causalrl.scm.graph import CausalGraph

_LATENT_GRID = (0.3, 0.7)
_PRIV_GRID = (0.3, 0.6)
_MAX_SAMPLED_COMBOS = 3000
_FULL_ENUM_CUTOFF = 4000


def _rf_apply(rf: int, parent_bits: tuple[int, ...]) -> int:
    """Evaluate truth table `rf` (encoded as an int) on a binary parent-value tuple."""
    idx = 0
    for b in parent_bits:
        idx = (idx << 1) | b
    return (rf >> idx) & 1


def find_witness(
    graph: CausalGraph, treatment: str, outcome: str, *, treated_value: int = 1, seed: int = 0
) -> dict[str, Any] | str | None:
    """Search for a non-identifiability witness for ``P(outcome | do(treatment))`` on `graph`.

    Returns a dict with the matching observational law and the disagreeing do-values if a witness
    is found, ``"TOO_BIG"`` if the graph exceeds the tractable size for this search, else ``None``
    (no witness found in this bounded search -- inconclusive, not evidence of identifiability).
    """
    nodes = graph.topological_order()
    if len(nodes) > 4:
        return "TOO_BIG"
    bidirected = graph.bidirected_edges
    n_latents = len(bidirected)
    latents_of: dict[str, list[int]] = {n: [] for n in nodes}
    for i, (u, w) in enumerate(bidirected):
        latents_of[u].append(i)
        latents_of[w].append(i)

    node_parents: dict[str, list[str]] = {}
    for n in nodes:
        pa = list(graph.parents(n))
        pa += [f"__L{i}" for i in latents_of[n]]
        pa.append(f"__P{n}")  # every node gets its own private-noise input
        node_parents[n] = pa
        if len(pa) > 3:
            return "TOO_BIG"

    rf_sizes = {n: 1 << (1 << len(node_parents[n])) for n in nodes}
    total_rf = 1
    for n in nodes:
        total_rf *= rf_sizes[n]
    total_space = total_rf * (len(_LATENT_GRID) ** max(n_latents, 1)) * len(_PRIV_GRID)

    rng = random.Random(seed)
    groups: dict[tuple, set[float]] = defaultdict(set)
    bit_names = [f"__L{i}" for i in range(n_latents)] + [f"__P{n}" for n in nodes]

    def evaluate(rf: dict[str, int], p_latent: list[float], q: float) -> None:
        pv: dict[tuple, float] = defaultdict(float)
        do_y = 0.0
        for bits in product([0, 1], repeat=len(bit_names)):
            bit_val = dict(zip(bit_names, bits, strict=True))
            weight = 1.0
            for i in range(n_latents):
                weight *= p_latent[i] if bit_val[f"__L{i}"] else (1 - p_latent[i])
            for n in nodes:
                weight *= q if bit_val[f"__P{n}"] else (1 - q)
            if weight <= 0.0:
                continue

            values: dict[str, int] = {}
            for n in nodes:
                pa_vals = tuple(
                    bit_val[pp] if pp.startswith("__") else values[pp] for pp in node_parents[n]
                )
                values[n] = _rf_apply(rf[n], pa_vals)
            pv[tuple(values[n] for n in nodes)] += weight

            values2: dict[str, int] = {}
            for n in nodes:
                if n == treatment:
                    values2[n] = treated_value
                    continue
                pa_vals = tuple(
                    bit_val[pp] if pp.startswith("__") else values2[pp] for pp in node_parents[n]
                )
                values2[n] = _rf_apply(rf[n], pa_vals)
            do_y += weight * values2[outcome]

        full_support = all(
            pv.get(combo, 0.0) > 1e-6 for combo in product([0, 1], repeat=len(nodes))
        )
        if not full_support:
            return
        key = tuple(sorted((k, round(v, 6)) for k, v in pv.items()))
        groups[key].add(round(do_y, 6))

    if total_space <= _FULL_ENUM_CUTOFF:
        rf_axes = [range(rf_sizes[n]) for n in nodes]
        latent_axes = product(_LATENT_GRID, repeat=n_latents) if n_latents else [()]
        for rf_combo in product(*rf_axes):
            rf = dict(zip(nodes, rf_combo, strict=True))
            for p_latent in latent_axes:
                for q in _PRIV_GRID:
                    evaluate(rf, list(p_latent), q)
    else:
        for _ in range(_MAX_SAMPLED_COMBOS):
            rf = {n: rng.randrange(rf_sizes[n]) for n in nodes}
            p_latent = [rng.choice(_LATENT_GRID) for _ in range(n_latents)]
            q = rng.choice(_PRIV_GRID)
            evaluate(rf, p_latent, q)

    for key, vals in groups.items():
        if len(vals) > 1:
            return {"group_key": key, "do_values": sorted(vals)}
    return None
