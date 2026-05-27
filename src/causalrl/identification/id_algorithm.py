"""General causal-effect identification: the ID algorithm (taxonomy Task 4).

Decide whether an interventional distribution ``P(y | do(x))`` is identifiable from observational
data in an ADMG, and if so return a do-free *estimand* — a symbolic expression over the
observational distribution that can be rendered as a formula and evaluated numerically on data.
When the effect is not identifiable the algorithm raises :class:`NotIdentifiableError`, attaching
the hedge that witnesses it.

This is the sound and complete non-parametric identification algorithm, superseding the conservative
adjustment-only slice in :mod:`causalrl.identification.criteria` for the observational case.

Faithful to:

- I. Shpitser, J. Pearl, *Identification of Joint Interventional Distributions in Recursive
  Semi-Markovian Causal Models*, AAAI 2006 (the ID algorithm and the hedge criterion).
- J. Tian, J. Pearl, *A General Identification Condition for Causal Effects*, AAAI 2002 (the
  C-component / Q-decomposition the recursion rests on).

No external code is ported; the recursion runs on our own :class:`~causalrl.scm.graph.CausalGraph`.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from causalrl.exceptions import CausalGraphError, NotIdentifiableError
from causalrl.scm.graph import CausalGraph

__all__ = [
    "Estimand",
    "estimate_effect",
    "identify_effect",
    "is_identifiable_effect",
]


# --------------------------------------------------------------------------------------------------
# Numeric factors: discrete distributions that support the operations an estimand evaluates to.
# --------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class _Factor:
    """A discrete function over ``variables``: assignment (in ``variables`` order) -> value."""

    variables: tuple[str, ...]
    table: Mapping[tuple[int, ...], float]

    def marginalize(self, over: Iterable[str]) -> _Factor:
        drop = set(over) & set(self.variables)
        if not drop:
            return self
        keep = tuple(v for v in self.variables if v not in drop)
        keep_idx = [self.variables.index(v) for v in keep]
        acc: dict[tuple[int, ...], float] = defaultdict(float)
        for assignment, value in self.table.items():
            acc[tuple(assignment[i] for i in keep_idx)] += value
        return _Factor(keep, dict(acc))

    def product(self, other: _Factor) -> _Factor:
        shared = [v for v in self.variables if v in other.variables]
        s_idx = [self.variables.index(v) for v in shared]
        o_idx = [other.variables.index(v) for v in shared]
        extra = [j for j, v in enumerate(other.variables) if v not in self.variables]
        out_vars = self.variables + tuple(other.variables[j] for j in extra)
        buckets: dict[tuple[int, ...], list[tuple[tuple[int, ...], float]]] = defaultdict(list)
        for oa, ov in other.table.items():
            buckets[tuple(oa[i] for i in o_idx)].append((oa, ov))
        acc: dict[tuple[int, ...], float] = {}
        for sa, sv in self.table.items():
            for oa, ov in buckets.get(tuple(sa[i] for i in s_idx), ()):
                acc[sa + tuple(oa[j] for j in extra)] = sv * ov
        return _Factor(out_vars, acc)

    def divide(self, denominator: _Factor) -> _Factor:
        d_idx = [self.variables.index(v) for v in denominator.variables]
        acc: dict[tuple[int, ...], float] = {}
        for assignment, value in self.table.items():
            denom = denominator.table.get(tuple(assignment[i] for i in d_idx), 0.0)
            acc[assignment] = value / denom if denom > 0 else 0.0
        return _Factor(self.variables, acc)

    def condition(self, assignment: Mapping[str, int]) -> _Factor:
        fixed = {v: assignment[v] for v in self.variables if v in assignment}
        if not fixed:
            return self
        fixed_idx = {self.variables.index(v): val for v, val in fixed.items()}
        keep = tuple(v for v in self.variables if v not in fixed)
        keep_idx = [self.variables.index(v) for v in keep]
        acc: dict[tuple[int, ...], float] = {}
        for a, value in self.table.items():
            if all(a[i] == val for i, val in fixed_idx.items()):
                acc[tuple(a[i] for i in keep_idx)] = value
        return _Factor(keep, acc)

    def reorder(self, variables: Sequence[str]) -> _Factor:
        idx = [self.variables.index(v) for v in variables]
        return _Factor(
            tuple(variables), {tuple(a[i] for i in idx): val for a, val in self.table.items()}
        )


# --------------------------------------------------------------------------------------------------
# Estimands: a symbolic expression over the observational distribution P(V).
# --------------------------------------------------------------------------------------------------
class Estimand:
    """A do-free expression for an interventional distribution over the observational law ``P(V)``.

    Render it with :meth:`render`; evaluate it on data with :func:`estimate_effect`.
    """

    def render(self) -> str:
        raise NotImplementedError


@dataclass(frozen=True)
class _Joint(Estimand):
    """The observational marginal ``P(variables)``."""

    variables: frozenset[str]

    def render(self) -> str:
        return f"P({','.join(sorted(self.variables))})"


@dataclass(frozen=True)
class _Marginal(Estimand):
    """``sum_{over} child``."""

    over: frozenset[str]
    child: Estimand

    def render(self) -> str:
        return f"sum_{{{','.join(sorted(self.over))}}} {self.child.render()}"


@dataclass(frozen=True)
class _Product(Estimand):
    """A product of sub-expressions."""

    terms: tuple[Estimand, ...]

    def render(self) -> str:
        return " ".join(f"[{t.render()}]" for t in self.terms)


@dataclass(frozen=True)
class _Quotient(Estimand):
    """``numerator / denominator`` (a conditional formed from a joint and its marginal)."""

    numerator: Estimand
    denominator: Estimand

    def render(self) -> str:
        return f"[{self.numerator.render()} / {self.denominator.render()}]"


def _evaluate(estimand: Estimand, joint: _Factor) -> _Factor:
    """Evaluate an estimand on the observational ``joint`` to a numeric factor."""
    if isinstance(estimand, _Joint):
        return joint.marginalize(set(joint.variables) - estimand.variables)
    if isinstance(estimand, _Marginal):
        return _evaluate(estimand.child, joint).marginalize(estimand.over)
    if isinstance(estimand, _Product):
        result = _evaluate(estimand.terms[0], joint)
        for term in estimand.terms[1:]:
            result = result.product(_evaluate(term, joint))
        return result
    if isinstance(estimand, _Quotient):
        return _evaluate(estimand.numerator, joint).divide(_evaluate(estimand.denominator, joint))
    raise TypeError(f"unknown estimand node: {type(estimand).__name__}")


def _marginal(over: Iterable[str], child: Estimand) -> Estimand:
    over = frozenset(over)
    return child if not over else _Marginal(over, child)


def _product(terms: Sequence[Estimand]) -> Estimand:
    return terms[0] if len(terms) == 1 else _Product(tuple(terms))


def _topo_conditionals(
    p: Estimand, order: Sequence[str], restrict: frozenset[str]
) -> list[Estimand]:
    """``[P(V_i | predecessors) for V_i in restrict]`` from ``p``, in topological ``order``.

    Each conditional is the quotient of two marginals of ``p``: summing out the nodes after ``V_i``
    leaves ``P(V^{(i)})``; summing out from ``V_i`` onward leaves ``P(V^{(i-1)})``.
    """
    conditionals: list[Estimand] = []
    for i, node in enumerate(order):
        if node in restrict:
            after = frozenset(order[i + 1 :])
            from_node = frozenset(order[i:])
            conditionals.append(_Quotient(_marginal(after, p), _marginal(from_node, p)))
    return conditionals


# --------------------------------------------------------------------------------------------------
# The ID algorithm (Shpitser & Pearl 2006).
# --------------------------------------------------------------------------------------------------
def _id(y: frozenset[str], x: frozenset[str], graph: CausalGraph, p: Estimand) -> Estimand:
    v = frozenset(graph.nodes)

    # 1. No remaining intervention: just marginalize.
    if not x:
        return _marginal(v - y, p)

    # 2. Restrict to the ancestors of Y.
    an_y = frozenset(graph.ancestors(y))
    if v != an_y:
        return _id(y, x & an_y, graph.induced_subgraph(an_y), _marginal(v - an_y, p))

    # 3. Force unhelpful non-ancestors (in G with edges into X cut) into the intervention set.
    g_cut = graph
    for node in x:
        g_cut = g_cut.remove_incoming_edges(node)
    w = (v - x) - frozenset(g_cut.ancestors(y))
    if w:
        return _id(y, x | w, graph, p)

    # 4. Decompose G[V \ X] into C-components.
    components = [frozenset(c) for c in graph.induced_subgraph(v - x).c_components()]
    if len(components) > 1:
        terms = [_id(s, v - s, graph, p) for s in components]
        return _marginal(v - (y | x), _product(terms))

    # 5. A single C-component S = V \ X.
    s = components[0]
    g_components = [frozenset(c) for c in graph.c_components()]

    # 6. The whole graph is one C-component: a hedge, so the effect is not identifiable.
    if len(g_components) == 1:
        raise NotIdentifiableError(
            f"P({sorted(y)} | do({sorted(x)})) is not identifiable: "
            f"hedge formed by C-component {sorted(s)} within {sorted(v)}",
            witness=(sorted(s), sorted(v)),
        )

    order = graph.topological_order()

    # 7. S is itself a C-component of G: factorize directly.
    if s in g_components:
        return _marginal(s - y, _product(_topo_conditionals(p, order, s)))

    # 8. S is strictly inside a C-component S': recurse into G[S'] with its factorization.
    s_prime = next(c for c in g_components if s < c)
    p_prime = _product(_topo_conditionals(p, order, s_prime))
    return _id(y, x & s_prime, graph.induced_subgraph(s_prime), p_prime)


def identify_effect(
    graph: CausalGraph, treatment: Iterable[str], outcome: Iterable[str]
) -> Estimand:
    """Return a do-free :class:`Estimand` for ``P(outcome | do(treatment))``, or raise.

    Runs the ID algorithm. Raises :class:`NotIdentifiableError` (with the witnessing hedge attached
    as ``.witness``) when the effect is not non-parametrically identifiable, and
    :class:`CausalGraphError` for malformed inputs (unknown nodes, empty outcome, or a treatment and
    outcome that overlap).
    """
    x, y = frozenset(treatment), frozenset(outcome)
    unknown = (x | y) - set(graph.nodes)
    if unknown:
        raise CausalGraphError(f"unknown nodes: {sorted(unknown)}")
    if not y:
        raise CausalGraphError("outcome must be non-empty")
    if x & y:
        raise CausalGraphError(f"treatment and outcome overlap: {sorted(x & y)}")
    return _id(y, x, graph, _Joint(frozenset(graph.nodes)))


def is_identifiable_effect(
    graph: CausalGraph, treatment: Iterable[str], outcome: Iterable[str]
) -> bool:
    """Whether ``P(outcome | do(treatment))`` is identifiable from observational data."""
    try:
        identify_effect(graph, treatment, outcome)
    except NotIdentifiableError:
        return False
    return True


def _empirical_joint(data: Mapping[str, np.ndarray], variables: Iterable[str]) -> _Factor:
    names = tuple(sorted(variables))
    n = len(data[names[0]])
    counts: dict[tuple[int, ...], int] = defaultdict(int)
    for row in range(n):
        counts[tuple(int(data[name][row]) for name in names)] += 1
    return _Factor(names, {key: count / n for key, count in counts.items()})


def estimate_effect(
    graph: CausalGraph,
    treatment: Iterable[str],
    outcome: Iterable[str],
    data: Mapping[str, np.ndarray],
    *,
    do: Mapping[str, int],
) -> dict[tuple[int, ...], float]:
    """Estimate ``P(outcome | do(treatment = do))`` from observational ``data`` via the ID estimand.

    Identifies the effect (raising if it cannot), then evaluates the resulting estimand on the
    empirical joint of ``data`` (discrete integer columns over ``graph.nodes``) at the intervention
    ``do``. Returns the outcome distribution as ``{assignment: probability}`` with assignments keyed
    in ``sorted(outcome)`` order.
    """
    estimand = identify_effect(graph, treatment, outcome)
    factor = _evaluate(estimand, _empirical_joint(data, graph.nodes))
    factor = factor.condition(do)
    targets = sorted(outcome)
    factor = factor.marginalize(set(factor.variables) - set(targets)).reorder(targets)
    return dict(factor.table)
