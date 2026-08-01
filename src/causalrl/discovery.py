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
from itertools import combinations, pairwise

import numpy as np

from causalrl.exceptions import CausalGraphError
from causalrl.scm.graph import CausalGraph

__all__ = ["CPDAG", "PAG", "conditional_mutual_information", "discover", "discover_interventional", "orient"]


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


_CIRCLE, _ARROW, _TAIL = "o", ">", "-"


@dataclass(frozen=True)
class PAG:
    """A partial ancestral graph: the complete Markov-equivalence class of MAGs (the FCI output).

    ``marks[(a, b)]`` is the mark on the ``b`` end of edge ``a—b`` — a circle ``o`` (undetermined by
    the equivalence class), an arrowhead ``>``, or a tail ``-``. An edge exists iff both ``(a, b)``
    and ``(b, a)`` are present. ``a -> b`` is tail-at-``a`` / arrow-at-``b``; ``a <-> b``
    (arrowheads at both ends) witnesses a latent confounder; ``a o-o b`` is fully unoriented.
    """

    variables: tuple[str, ...]
    marks: Mapping[tuple[str, str], str]

    def __post_init__(self) -> None:
        for (a, b), mark in self.marks.items():
            if mark not in (_CIRCLE, _ARROW, _TAIL):
                raise CausalGraphError(f"invalid PAG mark {mark!r} on edge {a}-{b}")
            if (b, a) not in self.marks:
                raise CausalGraphError(f"PAG edge {a}-{b} is missing endpoint ({b!r}, {a!r})")

    def adjacent(self, a: str, b: str) -> bool:
        return (a, b) in self.marks

    def is_directed(self, a: str, b: str) -> bool:
        """Whether ``a -> b`` (tail at ``a``, arrowhead at ``b``)."""
        return self.marks.get((a, b)) == _ARROW and self.marks.get((b, a)) == _TAIL

    def is_bidirected(self, a: str, b: str) -> bool:
        """Whether ``a <-> b`` (arrowheads at both ends — a latent confounder)."""
        return self.marks.get((a, b)) == _ARROW and self.marks.get((b, a)) == _ARROW

    def edges(self) -> list[tuple[str, str, str, str]]:
        """``(a, b, mark_at_a, mark_at_b)`` for each edge, with ``a < b``."""
        out: list[tuple[str, str, str, str]] = []
        seen: set[frozenset[str]] = set()
        for a, b in self.marks:
            key = frozenset((a, b))
            if key in seen:
                continue
            seen.add(key)
            x, y = sorted((a, b))
            out.append((x, y, self.marks[(y, x)], self.marks[(x, y)]))
        return sorted(out)

    def render(self) -> str:
        left = {_CIRCLE: "o", _ARROW: "<", _TAIL: "-"}
        right = {_CIRCLE: "o", _ARROW: ">", _TAIL: "-"}
        return ", ".join(f"{x} {left[ma]}-{right[mb]} {y}" for x, y, ma, mb in self.edges())


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


def _pc_skeleton(
    data: Mapping[str, np.ndarray],
    nodes: Sequence[str],
    *,
    threshold: float,
    max_conditioning_size: int,
) -> tuple[dict[str, set[str]], dict[frozenset[str], tuple[str, ...]]]:
    """The PC skeleton: drop ``a - b`` when some neighbour subset renders them independent.

    Returns the adjacency sets and the separating set recorded for each removed pair. Shared by
    :func:`discover` (PC) and :func:`discover_latent` (FCI).
    """
    for v in nodes:
        if v not in data:
            raise CausalGraphError(f"variable not in data: {v!r}")
    adj: dict[str, set[str]] = {v: set(nodes) - {v} for v in nodes}
    sepset: dict[frozenset[str], tuple[str, ...]] = {}
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
    return adj, sepset


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
    adj, sepset = _pc_skeleton(
        data, nodes, threshold=threshold, max_conditioning_size=max_conditioning_size
    )

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


def orient(cpdag: CPDAG, *, tiers: Sequence[Sequence[str]] | None = None) -> CausalGraph:
    """Orient a CPDAG's remaining undirected edges into a DAG, or refuse.

    Resolution order per undirected edge: (1) ``tiers`` — an edge between different tiers points
    from the earlier tier to the later one; (2) acyclicity — if one direction would close a cycle,
    take the other; (3) refuse. Silently picking an orientation would commit to a choice the data
    does not identify, so the third case raises and names :func:`causalrl.fit_scm_mec`, which fits
    every member of the equivalence class instead.
    """
    rank: dict[str, int] = {}
    if tiers is not None:
        for level, tier in enumerate(tiers):
            for name in tier:
                rank[name] = level
        uncovered = set(cpdag.variables) - set(rank)
        if uncovered:
            raise CausalGraphError(
                f"variable(s) {sorted(uncovered)} not covered by tiers {list(tiers)}"
            )

    directed = set(cpdag.directed_edges)
    pending = sorted([tuple(sorted(edge)) for edge in cpdag.undirected_edges])
    unresolved: list[tuple[str, str]] = []
    # Repeat: each orientation can unlock another edge through the acyclicity rule.
    while pending:
        progressed = False
        deferred: list[tuple[str, str]] = []
        for a, b in pending:
            if rank.get(a) is not None and rank.get(b) is not None and rank[a] != rank[b]:
                # Orient according to tiers, but first check that it doesn't create a cycle.
                tail, head = (a, b) if rank[a] < rank[b] else (b, a)
                cycle_path = _creates_cycle(directed, tail, head)
                if cycle_path is not None:
                    # cycle_path is [head, ...nodes..., tail], forming path head -> ... -> tail.
                    # The new edge tail -> head would close cycle: tail -> head -> ... -> tail.
                    cycle_edges = " -> ".join([tail] + cycle_path)
                    raise CausalGraphError(
                        f"tier-implied edge {tail} -> {head} would create a cycle: {cycle_edges}"
                    )
                directed.add((tail, head))
                progressed = True
                continue
            ab_cycle = _creates_cycle(directed, a, b)
            ba_cycle = _creates_cycle(directed, b, a)
            ab_ok = ab_cycle is None
            ba_ok = ba_cycle is None
            if ab_ok and not ba_ok:
                directed.add((a, b))
                progressed = True
            elif ba_ok and not ab_ok:
                directed.add((b, a))
                progressed = True
            else:
                deferred.append((a, b))
        if not progressed:
            unresolved = deferred
            break
        pending = deferred

    if unresolved:
        raise CausalGraphError(
            f"cannot orient edge(s) {sorted(unresolved)}: neither tiers nor acyclicity decides "
            "them. Pass tiers=..., or use fit_scm_mec to fit every member of the equivalence class."
        )
    return CausalGraph(directed_edges=sorted(directed), nodes=list(cpdag.variables))


def _creates_cycle(directed: set[tuple[str, str]], tail: str, head: str) -> list[str] | None:
    """Check if adding ``tail -> head`` would close a cycle in ``directed``.

    Returns the cycle path (nodes from ``head`` back to ``tail``) if a cycle would be created,
    or ``None`` if no cycle would result.
    """
    stack: list[list[str]] = [[head]]
    visited: set[str] = {head}
    while stack:
        path = stack.pop()
        node = path[-1]
        if node == tail:
            return path  # Path from head to tail exists; cycle is path + [tail]
        for u, v in directed:
            if u == node and v not in visited:
                visited.add(v)
                stack.append(path + [v])
    return None


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


# --------------------------------------------------------------------------------------------------
# FCI: causal discovery allowing latent confounders (and selection bias). Returns a PAG.
# --------------------------------------------------------------------------------------------------
def _pag_neighbors(marks: Mapping[tuple[str, str], str], a: str) -> set[str]:
    return {b for (x, b) in marks if x == a}


def _orient_colliders(
    marks: dict[tuple[str, str], str],
    nodes: Sequence[str],
    sepset: Mapping[frozenset[str], tuple[str, ...]],
) -> None:
    """Orient every unshielded collider ``a *-> b <-* c`` (arrowheads at ``b``)."""
    for b in nodes:
        for a, c in combinations(sorted(_pag_neighbors(marks, b)), 2):
            if (a, c) in marks:  # shielded triple
                continue
            if b not in sepset.get(frozenset((a, c)), ()):
                marks[(a, b)] = _ARROW
                marks[(c, b)] = _ARROW


def _possible_d_sep(a: str, marks: Mapping[tuple[str, str], str]) -> set[str]:
    """Possible-D-SEP(``a``): every ``V`` reachable from ``a`` by a path on which each consecutive
    triple ``<X, Y, Z>`` has ``Y`` a collider *or* forms a triangle (Spirtes et al.)."""
    pds: set[str] = set(_pag_neighbors(marks, a))
    stack: list[tuple[str, str]] = [(a, b) for b in _pag_neighbors(marks, a)]
    visited: set[tuple[str, str]] = set()
    while stack:
        prev, cur = stack.pop()
        if (prev, cur) in visited:
            continue
        visited.add((prev, cur))
        for nxt in _pag_neighbors(marks, cur):
            if nxt == prev:
                continue
            collider = marks.get((prev, cur)) == _ARROW and marks.get((nxt, cur)) == _ARROW
            triangle = (prev, nxt) in marks
            if collider or triangle:
                if nxt != a:
                    pds.add(nxt)
                stack.append((cur, nxt))
    pds.discard(a)
    return pds


def _refine_skeleton_pds(
    data: Mapping[str, np.ndarray],
    marks: dict[tuple[str, str], str],
    sepset: dict[frozenset[str], tuple[str, ...]],
    nodes: Sequence[str],
    *,
    threshold: float,
    max_conditioning_size: int,
) -> None:
    """FCI Phase II: drop ``a - b`` when a subset of Possible-D-SEP renders them independent."""
    for a in list(nodes):
        for b in sorted(_pag_neighbors(marks, a)):
            if (a, b) not in marks:
                continue
            pds = sorted(_possible_d_sep(a, marks) - {a, b})
            removed = False
            for size in range(min(max_conditioning_size, len(pds)) + 1):
                for cond in combinations(pds, size):
                    if _independent(data, a, b, cond, threshold=threshold):
                        marks.pop((a, b), None)
                        marks.pop((b, a), None)
                        sepset[frozenset((a, b))] = cond
                        removed = True
                        break
                if removed:
                    break


def _rule1(marks: dict[tuple[str, str], str]) -> bool:
    """R1: ``a *-> b o-* c`` with ``a, c`` non-adjacent forces ``b -> c``."""
    changed = False
    for a, b in list(marks):
        if marks.get((a, b)) != _ARROW:
            continue
        for c in _pag_neighbors(marks, b):
            if c == a or (a, c) in marks or marks.get((c, b)) != _CIRCLE:
                continue
            marks[(c, b)] = _TAIL
            marks[(b, c)] = _ARROW
            changed = True
    return changed


def _rule2(marks: dict[tuple[str, str], str]) -> bool:
    """R2: ``a -> b *-> c`` or ``a *-> b -> c`` with ``a *-o c`` forces an arrowhead at ``c``."""
    changed = False
    for (a, c), mark in list(marks.items()):
        if mark != _CIRCLE:
            continue
        for b in _pag_neighbors(marks, a):
            if b == c or (b, c) not in marks:
                continue
            into_c = marks.get((b, c)) == _ARROW
            chain1 = marks.get((a, b)) == _ARROW and marks.get((b, a)) == _TAIL and into_c
            chain2 = marks.get((a, b)) == _ARROW and into_c and marks.get((c, b)) == _TAIL
            if chain1 or chain2:
                marks[(a, c)] = _ARROW
                changed = True
                break
    return changed


def _rule3(marks: dict[tuple[str, str], str]) -> bool:
    """R3: ``a *-> b <-* c`` with ``a *-o d o-* c`` (``a,c`` non-adjacent) and ``d *-o b`` forces
    ``d *-> b``."""
    changed = False
    for b in {b for _, b in marks}:
        into_b = [x for x in _pag_neighbors(marks, b) if marks.get((x, b)) == _ARROW]
        for d in _pag_neighbors(marks, b):
            if marks.get((d, b)) != _CIRCLE:
                continue
            for a, c in combinations(sorted(into_b), 2):
                if (a, c) in marks or a == d or c == d:
                    continue
                if marks.get((a, d)) == _CIRCLE and marks.get((c, d)) == _CIRCLE:
                    marks[(d, b)] = _ARROW
                    changed = True
                    break
    return changed


def _discriminating_path(marks: dict[tuple[str, str], str], b: str, c: str) -> list[str] | None:
    """A discriminating path ``<theta, ..., a, b, c>`` for ``b``, or ``None``.

    Vertices strictly between ``theta`` and ``b`` are colliders on the path and parents of ``c``;
    ``theta`` is not adjacent to ``c``.
    """
    for a in _pag_neighbors(marks, b):
        if a == c or marks.get((b, a)) != _ARROW:  # need b *-> a
            continue
        if not (marks.get((a, c)) == _ARROW and marks.get((c, a)) == _TAIL):  # need a -> c
            continue
        stack: list[list[str]] = [[a, b, c]]
        while stack:
            path = stack.pop()
            head = path[0]
            for theta in _pag_neighbors(marks, head):
                if theta in path or marks.get((theta, head)) != _ARROW:  # need theta *-> head
                    continue
                if (theta, c) not in marks:
                    return [theta, *path]  # theta non-adjacent to c: discriminating
                if marks.get((theta, c)) == _ARROW and marks.get((c, theta)) == _TAIL:
                    stack.append([theta, *path])  # theta -> c: keep extending
    return None


def _uncovered_paths(
    marks: dict[tuple[str, str], str], start: str, end: str, *, circle: bool
) -> list[list[str]]:
    """Uncovered paths ``start … end`` that are circle paths (every edge ``o-o``) when ``circle``,
    else potentially-directed (no arrowhead at the tail-side node of each step)."""
    results: list[list[str]] = []

    def step_ok(u: str, w: str) -> bool:
        if circle:
            return marks.get((u, w)) == _CIRCLE and marks.get((w, u)) == _CIRCLE
        return marks.get((w, u)) != _ARROW  # no arrowhead at u: orientable u -> w

    def dfs(path: list[str]) -> None:
        u = path[-1]
        for w in _pag_neighbors(marks, u):
            if w in path or not step_ok(u, w):
                continue
            if len(path) >= 2 and (path[-2], w) in marks:  # uncovered triple
                continue
            extended = [*path, w]
            if w == end:
                results.append(extended)
            else:
                dfs(extended)

    dfs([start])
    return results


def _rule4(
    marks: dict[tuple[str, str], str], sepset: Mapping[frozenset[str], tuple[str, ...]]
) -> bool:
    """R4: a discriminating path for ``b`` ending ``a, b, c`` with ``b o-* c`` orients ``b -> c`` if
    ``b`` is in the separating set, else ``a <-> b <-> c``."""
    changed = False
    for (c, b), mark in list(marks.items()):
        if mark != _CIRCLE:
            continue
        path = _discriminating_path(marks, b, c)
        if path is None:
            continue
        theta, a = path[0], path[-3]
        if b in sepset.get(frozenset((theta, c)), ()):
            marks[(c, b)], marks[(b, c)] = _TAIL, _ARROW
        else:
            marks[(a, b)] = marks[(b, a)] = marks[(b, c)] = marks[(c, b)] = _ARROW
        changed = True
    return changed


def _rule5(marks: dict[tuple[str, str], str]) -> bool:
    """R5: ``a o-o b`` with an uncovered circle path between them (suitable endpoints non-adjacent)
    makes ``a - b`` and every edge on the path undirected (selection bias)."""
    changed = False
    for (a, b), mark in list(marks.items()):
        if a >= b or not (mark == _CIRCLE and marks.get((b, a)) == _CIRCLE):
            continue
        for path in _uncovered_paths(marks, a, b, circle=True):
            if len(path) < 4:
                continue
            gamma, theta = path[1], path[-2]
            if (a, theta) in marks or (b, gamma) in marks:
                continue
            marks[(a, b)] = marks[(b, a)] = _TAIL
            for u, v in pairwise(path):
                marks[(u, v)] = marks[(v, u)] = _TAIL
            changed = True
            break
    return changed


def _rule6(marks: dict[tuple[str, str], str]) -> bool:
    """R6: ``a - b o-* c`` (``a - b`` undirected) forces a tail at ``b`` on ``b - c``."""
    changed = False
    for (c, b), mark in list(marks.items()):
        if mark != _CIRCLE:
            continue
        for a in _pag_neighbors(marks, b):
            if a != c and marks.get((a, b)) == _TAIL and marks.get((b, a)) == _TAIL:
                marks[(c, b)] = _TAIL
                changed = True
                break
    return changed


def _rule7(marks: dict[tuple[str, str], str]) -> bool:
    """R7: ``a -o b o-* c`` with ``a, c`` non-adjacent forces a tail at ``b`` on ``b - c``."""
    changed = False
    for (g, b), mark in list(marks.items()):
        if mark != _CIRCLE:
            continue
        for a in _pag_neighbors(marks, b):
            if a == g or (a, g) in marks:
                continue
            if marks.get((b, a)) == _TAIL and marks.get((a, b)) == _CIRCLE:  # a -o b
                marks[(g, b)] = _TAIL
                changed = True
                break
    return changed


def _rule8(marks: dict[tuple[str, str], str]) -> bool:
    """R8: ``a -> b -> c`` or ``a -o b -> c`` with ``a o-> c`` forces a tail at ``a``."""
    changed = False
    for (a, c), mark in list(marks.items()):
        if not (mark == _ARROW and marks.get((c, a)) == _CIRCLE):  # a o-> c
            continue
        for b in _pag_neighbors(marks, a):
            if b == c or (b, c) not in marks:
                continue
            a_to_b = marks.get((a, b)) == _ARROW and marks.get((b, a)) == _TAIL
            a_circ_b = marks.get((b, a)) == _TAIL and marks.get((a, b)) == _CIRCLE
            b_to_c = marks.get((b, c)) == _ARROW and marks.get((c, b)) == _TAIL
            if (a_to_b or a_circ_b) and b_to_c:
                marks[(c, a)] = _TAIL
                changed = True
                break
    return changed


def _rule9(marks: dict[tuple[str, str], str]) -> bool:
    """R9: ``a o-> c`` with an uncovered potentially-directed path ``a, b, ..., c`` where ``b`` is
    non-adjacent to ``c`` forces a tail at ``a``."""
    changed = False
    for (a, c), mark in list(marks.items()):
        if not (mark == _ARROW and marks.get((c, a)) == _CIRCLE):
            continue
        for path in _uncovered_paths(marks, a, c, circle=False):
            if len(path) >= 3 and (path[1], c) not in marks:
                marks[(c, a)] = _TAIL
                changed = True
                break
    return changed


def _rule10(marks: dict[tuple[str, str], str]) -> bool:
    """R10: ``a o-> c`` with parents ``b -> c <- d`` and uncovered p.d. paths ``a..b``, ``a..d``
    whose first steps are distinct and non-adjacent forces a tail at ``a`` (``a -> c``)."""
    changed = False
    for (a, c), mark in list(marks.items()):
        if not (mark == _ARROW and marks.get((c, a)) == _CIRCLE):
            continue
        parents = [
            p
            for p in _pag_neighbors(marks, c)
            if marks.get((p, c)) == _ARROW and marks.get((c, p)) == _TAIL
        ]
        oriented = False
        for beta, delta in combinations(parents, 2):
            paths_b = [p for p in _uncovered_paths(marks, a, beta, circle=False) if len(p) >= 2]
            paths_d = [p for p in _uncovered_paths(marks, a, delta, circle=False) if len(p) >= 2]
            for p1 in paths_b:
                for p2 in paths_d:
                    if p1[1] != p2[1] and (p1[1], p2[1]) not in marks:
                        marks[(c, a)] = _TAIL
                        oriented = True
                        break
                if oriented:
                    break
            if oriented:
                break
        changed = changed or oriented
    return changed


def _apply_fci_rules(
    marks: dict[tuple[str, str], str], sepset: Mapping[frozenset[str], tuple[str, ...]]
) -> None:
    """Apply the complete FCI orientation rules R1-R10 (Zhang 2008) to a fixpoint."""
    rules = (
        lambda: _rule1(marks),
        lambda: _rule2(marks),
        lambda: _rule3(marks),
        lambda: _rule4(marks, sepset),
        lambda: _rule5(marks),
        lambda: _rule6(marks),
        lambda: _rule7(marks),
        lambda: _rule8(marks),
        lambda: _rule9(marks),
        lambda: _rule10(marks),
    )
    changed = True
    while changed:
        changed = False
        for rule in rules:
            if rule():
                changed = True


def discover_latent(
    data: Mapping[str, np.ndarray],
    variables: Sequence[str],
    *,
    threshold: float = 0.01,
    max_conditioning_size: int = 3,
) -> PAG:
    """Discover a PAG over ``variables`` from ``data`` via the FCI algorithm (allows latents).

    Unlike :func:`discover`, FCI does not assume causal sufficiency: it learns the PC skeleton, then
    refines it with the Possible-D-SEP step (sound under latent confounders), re-orients unshielded
    colliders, and applies the complete orientation rules R1-R10 (Zhang 2008 — sound and complete
    for latent confounders and selection bias). The result is a :class:`PAG`: ``a <-> b``
    witnesses a latent confounder; a circle endpoint is undetermined by the equivalence class.

    ``threshold`` and ``max_conditioning_size`` mirror :func:`discover`.
    """
    nodes = list(variables)
    adj, sepset = _pc_skeleton(
        data, nodes, threshold=threshold, max_conditioning_size=max_conditioning_size
    )
    marks: dict[tuple[str, str], str] = {}
    for a in nodes:
        for b in adj[a]:
            marks[(a, b)] = _CIRCLE
    _orient_colliders(marks, nodes, sepset)
    _refine_skeleton_pds(
        data, marks, sepset, nodes, threshold=threshold, max_conditioning_size=max_conditioning_size
    )
    for key in marks:
        marks[key] = _CIRCLE
    _orient_colliders(marks, nodes, sepset)
    _apply_fci_rules(marks, sepset)
    return PAG(tuple(nodes), marks)
