"""Constraint-based causal discovery (taxonomy Task 5).

Learn the causal structure (a CPDAG) from discrete observational data with the PC algorithm, then
hand the result to the rest of the library for causal planning. Conditional independence is tested
by thresholded conditional mutual information (numpy only — the project has no SciPy dependency).

Faithful to:

- P. Spirtes, C. Glymour, R. Scheines, *Causation, Prediction, and Search* (2nd ed., 2000) — the
  PC algorithm (skeleton via CI tests, then collider orientation).
- C. Meek, *Causal Inference and Causal Explanation with Background Knowledge*, UAI 1995 — the
  orientation rules R1-R3.

When interventional data is also available, :func:`discover_interventional` combines the
observational (L1) skeleton with experiments (L2) and orients further toward the *interventional*
essential graph, faithful to:

- A. Hauser, P. Buehlmann, *Characterization and Greedy Learning of Interventional Markov
  Equivalence Classes of DAGs*, JMLR 2012 (interventional Markov equivalence / I-essential graphs).
- J. Peters, P. Buehlmann, N. Meinshausen, *Causal Inference using Invariant Prediction*, JRSS-B
  2016 (the invariance principle that orients edges incident to an intervention target).

Assumes causal sufficiency (no latent confounders) and faithfulness; returns the (interventional)
Markov-equivalence class (CPDAG), which may leave some edges unoriented. No external code is ported.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations

import numpy as np

from causalrl.exceptions import CausalGraphError
from causalrl.scm.graph import CausalGraph

__all__ = ["CPDAG", "conditional_mutual_information", "discover", "discover_interventional"]


def conditional_mutual_information(
    data: Mapping[str, np.ndarray], x: str, y: str, z: Sequence[str]
) -> float:
    """Empirical ``I(X; Y | Z)`` in nats (discrete columns; ``0`` iff ``X ⊥ Y | Z``)."""
    xs, ys = data[x], data[y]
    n = len(xs)
    zcols = [data[zi] for zi in z]
    joint: dict[tuple[int, int, tuple[int, ...]], int] = defaultdict(int)
    xz: dict[tuple[int, tuple[int, ...]], int] = defaultdict(int)
    yz: dict[tuple[int, tuple[int, ...]], int] = defaultdict(int)
    zc: dict[tuple[int, ...], int] = defaultdict(int)
    for i in range(n):
        xv, yv = int(xs[i]), int(ys[i])
        zk = tuple(int(c[i]) for c in zcols)
        joint[(xv, yv, zk)] += 1
        xz[(xv, zk)] += 1
        yz[(yv, zk)] += 1
        zc[zk] += 1
    cmi = 0.0
    for (xv, yv, zk), count in joint.items():
        ratio = (count * zc[zk]) / (xz[(xv, zk)] * yz[(yv, zk)])
        cmi += (count / n) * math.log(ratio)
    return max(cmi, 0.0)


def _independent(
    data: Mapping[str, np.ndarray], x: str, y: str, z: Sequence[str], *, threshold: float
) -> bool:
    return conditional_mutual_information(data, x, y, z) < threshold


@dataclass(frozen=True)
class CPDAG:
    """A completed partially directed acyclic graph (a Markov equivalence class)."""

    variables: tuple[str, ...]
    directed_edges: frozenset[tuple[str, str]]
    undirected_edges: frozenset[frozenset[str]]

    def to_causal_graph(self) -> CausalGraph:
        """Convert to a :class:`CausalGraph`; raises if any edge is still unoriented."""
        if self.undirected_edges:
            remaining = sorted(tuple(sorted(e)) for e in self.undirected_edges)
            raise CausalGraphError(
                f"CPDAG is not fully oriented; undirected edges remain: {remaining}"
            )
        return CausalGraph(directed_edges=sorted(self.directed_edges), nodes=list(self.variables))


def _adjacent(
    directed: set[tuple[str, str]], undirected: set[frozenset[str]], a: str, b: str
) -> bool:
    return (a, b) in directed or (b, a) in directed or frozenset((a, b)) in undirected


def _orient(
    a: str, b: str, directed: set[tuple[str, str]], undirected: set[frozenset[str]]
) -> None:
    """Orient an undirected edge ``a - b`` as ``a -> b`` (no-op if already oriented)."""
    edge = frozenset((a, b))
    if edge in undirected:
        undirected.discard(edge)
        directed.add((a, b))


def _meek_forces(
    nodes: Sequence[str],
    directed: set[tuple[str, str]],
    undirected: set[frozenset[str]],
    u: str,
    v: str,
) -> bool:
    """Whether Meek rules R1-R3 force orienting the undirected edge ``u - v`` as ``u -> v``."""
    # R1: w -> u, with w not adjacent to v, forces u -> v (avoid creating a new collider at u).
    for w in nodes:
        if (w, u) in directed and w != v and not _adjacent(directed, undirected, w, v):
            return True
    # R2: a directed path u -> w -> v forces u -> v (avoid a cycle).
    for w in nodes:
        if (u, w) in directed and (w, v) in directed:
            return True
    # R3: two non-adjacent w1, w2 with u - w1, u - w2 and w1 -> v, w2 -> v force u -> v.
    for w1, w2 in combinations(nodes, 2):
        if (
            frozenset((u, w1)) in undirected
            and frozenset((u, w2)) in undirected
            and (w1, v) in directed
            and (w2, v) in directed
            and not _adjacent(directed, undirected, w1, w2)
        ):
            return True
    return False


def _apply_meek_rules(
    nodes: Sequence[str], directed: set[tuple[str, str]], undirected: set[frozenset[str]]
) -> None:
    """Repeatedly orient edges forced by Meek rules R1-R3 until no more change."""
    changed = True
    while changed:
        changed = False
        for edge in list(undirected):
            a, b = sorted(edge)
            for u, v in ((a, b), (b, a)):
                if _meek_forces(nodes, directed, undirected, u, v):
                    undirected.discard(edge)
                    directed.add((u, v))
                    changed = True
                    break


def discover(
    data: Mapping[str, np.ndarray],
    variables: Sequence[str],
    *,
    threshold: float = 0.01,
    max_conditioning_size: int = 3,
) -> CPDAG:
    """Discover the CPDAG over `variables` from `data` via the PC algorithm.

    ``threshold`` is the conditional-mutual-information cutoff below which two variables are judged
    independent; ``max_conditioning_size`` caps the separating-set search.
    """
    nodes = list(variables)
    for v in nodes:
        if v not in data:
            raise CausalGraphError(f"variable not in data: {v!r}")

    adj: dict[str, set[str]] = {v: set(nodes) - {v} for v in nodes}
    sepset: dict[frozenset[str], tuple[str, ...]] = {}

    # Skeleton phase: drop a - b when some neighbor subset renders them independent.
    for size in range(max_conditioning_size + 1):
        testable = False
        for a in nodes:
            for b in sorted(adj[a]):
                rest = sorted(adj[a] - {b})
                if len(rest) < size:
                    continue
                testable = True
                for candidate in combinations(rest, size):
                    if _independent(data, a, b, candidate, threshold=threshold):
                        adj[a].discard(b)
                        adj[b].discard(a)
                        sepset[frozenset((a, b))] = candidate
                        break
        if not testable:
            break

    # Orient unshielded colliders a -> c <- b (a, b non-adjacent and c not in their separating set).
    directed: set[tuple[str, str]] = set()
    undirected: set[frozenset[str]] = {frozenset((a, b)) for a in nodes for b in adj[a] if a < b}
    for c in nodes:
        for a, b in combinations(sorted(adj[c]), 2):
            if b in adj[a]:
                continue  # shielded triple
            if c not in sepset.get(frozenset((a, b)), ()):
                _orient(a, c, directed, undirected)
                _orient(b, c, directed, undirected)

    _apply_meek_rules(nodes, directed, undirected)
    return CPDAG(tuple(nodes), frozenset(directed), frozenset(undirected))


def _empirical_pmf(column: np.ndarray) -> dict[int, float]:
    counts: dict[int, int] = defaultdict(int)
    for value in column:
        counts[int(value)] += 1
    n = len(column)
    return {k: c / n for k, c in counts.items()}


def _total_variation(a: np.ndarray, b: np.ndarray) -> float:
    """Total-variation distance between the empirical marginals of two integer samples."""
    pa, pb = _empirical_pmf(a), _empirical_pmf(b)
    return 0.5 * sum(abs(pa.get(v, 0.0) - pb.get(v, 0.0)) for v in set(pa) | set(pb))


def discover_interventional(
    observational: Mapping[str, np.ndarray],
    interventions: Mapping[str, Mapping[str, np.ndarray]],
    variables: Sequence[str],
    *,
    threshold: float = 0.01,
    shift_threshold: float = 0.05,
    max_conditioning_size: int = 3,
) -> CPDAG:
    """Discover the interventional essential graph from observational and experimental data.

    Runs the observational PC algorithm (:func:`discover`), then orients the edges incident to each
    intervention target by the invariance principle: under a perfect intervention ``do(T)`` a
    *child* of ``T`` shifts its marginal, while a *parent* (a non-descendant) stays invariant. Each
    incident edge ``T - B`` is oriented ``T -> B`` if ``B`` shifts and ``B -> T`` if not; Meek's
    rules R1-R3 then propagate the new orientations, so the result refines the observational CPDAG
    toward the true DAG as more targets are experimented on.

    ``interventions`` maps each intervened target ``T`` to a dataset drawn from ``do(T)`` (a perfect
    intervention covering every variable in ``variables``). ``shift_threshold`` is the
    total-variation cutoff above which an endpoint's marginal is judged to have shifted.
    """
    nodes = list(variables)
    cpdag = discover(
        observational, nodes, threshold=threshold, max_conditioning_size=max_conditioning_size
    )
    directed = set(cpdag.directed_edges)
    undirected = set(cpdag.undirected_edges)

    for target, idata in interventions.items():
        if target not in nodes:
            raise CausalGraphError(f"intervention target not in variables: {target!r}")
        for edge in list(undirected):
            if target not in edge:
                continue
            other = next(iter(edge - {target}))
            if other not in observational or other not in idata:
                raise CausalGraphError(f"variable not in data: {other!r}")
            shifted = _total_variation(observational[other], idata[other]) >= shift_threshold
            undirected.discard(edge)
            directed.add((target, other) if shifted else (other, target))

    _apply_meek_rules(nodes, directed, undirected)
    return CPDAG(tuple(nodes), frozenset(directed), frozenset(undirected))
