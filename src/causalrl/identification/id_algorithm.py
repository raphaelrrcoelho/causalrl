"""General causal-effect identification: the ID algorithm (taxonomy Task 4).

Decide whether an interventional distribution ``P(y | do(x))`` is identifiable from observational
data in an ADMG, and if so return a do-free *estimand* — a symbolic expression over the
observational distribution that can be rendered as a formula and evaluated numerically on data.
When the effect is not identifiable the algorithm raises :class:`NotIdentifiableError`, attaching
the hedge that witnesses it.

This is the sound and complete non-parametric identification algorithm, superseding the conservative
adjustment-only slice in :mod:`causalrl.identification.criteria` for the observational case. When
surrogate experiments are available, `identify_effect_with_experiments` runs general identification
(gID): the same recursion, but a c-factor that observation cannot identify (a hedge) may instead be
obtained from an available experiment.

Faithful to:

- I. Shpitser, J. Pearl, *Identification of Joint Interventional Distributions in Recursive
  Semi-Markovian Causal Models*, AAAI 2006 (the ID algorithm and the hedge criterion).
- J. Tian, J. Pearl, *A General Identification Condition for Causal Effects*, AAAI 2002 (the
  C-component / Q-decomposition the recursion rests on, and the ``Identify`` subroutine).
- S. Lee, J. Correa, E. Bareinboim, *General Identifiability with Arbitrary Surrogate Experiments*,
  UAI 2019 (gID — obtaining hedged c-factors from available experiments).

No external code is ported; the recursion runs on our own :class:`~causalrl.scm.graph.CausalGraph`.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, overload

import numpy as np

from causalrl.exceptions import CausalGraphError, NotIdentifiableError
from causalrl.scm.graph import CausalGraph

if TYPE_CHECKING:
    from causalrl.certify.certificate import Certificate

__all__ = [
    "Domain",
    "Estimand",
    "estimate_effect",
    "estimate_effect_with_experiments",
    "estimate_transport_general",
    "estimate_transported_effect",
    "identify_effect",
    "identify_effect_with_experiments",
    "identify_transport",
    "identify_transport_general",
    "is_gid_identifiable",
    "is_identifiable_effect",
    "is_transportable_effect",
    "is_transportable_general",
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


@dataclass(frozen=True)
class _Experiment(Estimand):
    """An available experimental c-factor ``P(variables | do(intervened))`` (a surrogate)."""

    intervened: frozenset[str]
    variables: frozenset[str]

    def render(self) -> str:
        do = ",".join(sorted(self.intervened)) or "."
        return f"P({','.join(sorted(self.variables))} | do({do}))"


@dataclass(frozen=True)
class _DomainJoint(Estimand):
    """The observational marginal ``P_domain(variables)`` in a named domain (source or target)."""

    domain: str
    variables: frozenset[str]

    def render(self) -> str:
        return f"P_{self.domain}({','.join(sorted(self.variables))})"


@dataclass(frozen=True)
class _DomainExperiment(Estimand):
    """An experimental c-factor ``P_domain(variables | do(intervened))`` from a named domain."""

    domain: str
    intervened: frozenset[str]
    variables: frozenset[str]

    def render(self) -> str:
        do = ",".join(sorted(self.intervened)) or "."
        return f"P_{self.domain}({','.join(sorted(self.variables))} | do({do}))"


@dataclass(frozen=True)
class _EvalContext:
    """The data an estimand evaluates against: observational, experimental, and per-domain joints
    and experiments."""

    joint: _Factor
    experiments: Mapping[frozenset[str], _Factor]
    domains: Mapping[str, _Factor]
    domain_experiments: Mapping[tuple[str, frozenset[str]], _Factor]


def _evaluate(estimand: Estimand, ctx: _EvalContext) -> _Factor:
    """Evaluate an estimand against the observational, experimental, and domain data in ``ctx``."""
    if isinstance(estimand, _Joint):
        return ctx.joint.marginalize(set(ctx.joint.variables) - estimand.variables)
    if isinstance(estimand, _DomainJoint):
        if estimand.domain not in ctx.domains:
            raise CausalGraphError(f"missing {estimand.domain!r} domain data")
        domain = ctx.domains[estimand.domain]
        return domain.marginalize(set(domain.variables) - estimand.variables)
    if isinstance(estimand, _Experiment):
        if estimand.intervened not in ctx.experiments:
            raise CausalGraphError(f"missing experiment do({sorted(estimand.intervened)})")
        full = ctx.experiments[estimand.intervened]  # the joint over V with `intervened` randomized
        intervened = set(estimand.intervened)
        # Q[V\z] = P(V\z | do(z)) = P(V\z | z) under randomization: condition on the intervened set.
        conditional = full.divide(full.marginalize(set(full.variables) - intervened))
        return conditional.marginalize(set(full.variables) - (set(estimand.variables) | intervened))
    if isinstance(estimand, _DomainExperiment):
        key = (estimand.domain, estimand.intervened)
        if key not in ctx.domain_experiments:
            raise CausalGraphError(
                f"missing {estimand.domain!r} experiment do({sorted(estimand.intervened)})"
            )
        full = ctx.domain_experiments[key]
        intervened = set(estimand.intervened)
        conditional = full.divide(full.marginalize(set(full.variables) - intervened))
        return conditional.marginalize(set(full.variables) - (set(estimand.variables) | intervened))
    if isinstance(estimand, _Marginal):
        return _evaluate(estimand.child, ctx).marginalize(estimand.over)
    if isinstance(estimand, _Product):
        result = _evaluate(estimand.terms[0], ctx)
        for term in estimand.terms[1:]:
            result = result.product(_evaluate(term, ctx))
        return result
    if isinstance(estimand, _Quotient):
        return _evaluate(estimand.numerator, ctx).divide(_evaluate(estimand.denominator, ctx))
    raise TypeError(f"unknown estimand node: {type(estimand).__name__}")


def _eval_observational(estimand: Estimand, joint: _Factor) -> _Factor:
    return _evaluate(estimand, _EvalContext(joint, {}, {}, {}))


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


def _identify_c_factor(
    c: frozenset[str], t: frozenset[str], graph: CausalGraph, q: Estimand
) -> Estimand:
    """Tian's ``Identify``: compute the c-factor ``Q[c]`` from a known c-factor ``q = Q[t]`` (``c``
    a subset of ``t``), within ``graph`` induced on ``t``. Raises if ``Q[c]`` is not obtainable from
    ``Q[t]`` (the caller then tries another available experiment).
    """
    g_t = graph.induced_subgraph(t)
    a = frozenset(g_t.ancestors(c))
    if a == c:
        return _marginal(t - c, q)
    if a == t:
        raise NotIdentifiableError(f"Q[{sorted(c)}] not obtainable from Q[{sorted(t)}]")
    g_a = graph.induced_subgraph(a)
    q_a = _marginal(t - a, q)
    t_c = next(frozenset(comp) for comp in g_a.c_components() if c <= frozenset(comp))
    q_tc = _product(_topo_conditionals(q_a, g_a.topological_order(), t_c))
    return _identify_c_factor(c, t_c, graph, q_tc)


def _c_factor_from(
    c: frozenset[str], t: frozenset[str], graph: CausalGraph, q: Estimand
) -> Estimand:
    """Compute ``Q[c]`` from ``q = Q[t]`` for any subset ``c`` of ``t``.

    ``Identify`` assumes its domain is a single c-component, so first decompose ``G[t]`` into
    c-components, take ``Q[D]`` for the component ``D`` that contains ``c`` (Tian's c-factorization
    of ``q``), and only then run :func:`_identify_c_factor` within ``D``.
    """
    g_t = graph.induced_subgraph(t)
    d = next(frozenset(comp) for comp in g_t.c_components() if c <= frozenset(comp))
    if d == t:
        return _identify_c_factor(c, t, graph, q)
    q_d = _product(_topo_conditionals(q, g_t.topological_order(), d))
    return _identify_c_factor(c, d, graph, q_d)


# --------------------------------------------------------------------------------------------------
# The ID algorithm (Shpitser & Pearl 2006).
# --------------------------------------------------------------------------------------------------
def _id(
    y: frozenset[str],
    x: frozenset[str],
    graph: CausalGraph,
    p: Estimand,
    *,
    experiments: list[frozenset[str]] | None = None,
    original: CausalGraph | None = None,
) -> Estimand:
    """The ID recursion. With ``experiments`` (a list of available intervention targets) and
    ``original`` (the top-level graph), it becomes gID: at the hedge step (6) it tries to obtain the
    needed c-factor from a surrogate experiment instead of failing. With ``experiments is None`` it
    is exactly the Shpitser-Pearl ID algorithm.
    """
    v = frozenset(graph.nodes)

    # 1. No remaining intervention: just marginalize.
    if not x:
        return _marginal(v - y, p)

    # 2. Restrict to the ancestors of Y.
    an_y = frozenset(graph.ancestors(y))
    if v != an_y:
        return _id(
            y,
            x & an_y,
            graph.induced_subgraph(an_y),
            _marginal(v - an_y, p),
            experiments=experiments,
            original=original,
        )

    # 3. Force unhelpful non-ancestors (in G with edges into X cut) into the intervention set.
    g_cut = graph
    for node in x:
        g_cut = g_cut.remove_incoming_edges(node)
    w = (v - x) - frozenset(g_cut.ancestors(y))
    if w:
        return _id(y, x | w, graph, p, experiments=experiments, original=original)

    # 4. Decompose G[V \ X] into C-components.
    components = [frozenset(c) for c in graph.induced_subgraph(v - x).c_components()]
    if len(components) > 1:
        terms = [
            _id(s, v - s, graph, p, experiments=experiments, original=original) for s in components
        ]
        return _marginal(v - (y | x), _product(terms))

    # 5. A single C-component S = V \ X.
    s = components[0]
    g_components = [frozenset(c) for c in graph.c_components()]

    # 6. The whole graph is one C-component: a hedge. ID fails here; gID tries to obtain the needed
    #    c-factor Q[S] from a surrogate experiment do(z) (Q[S] is the same original-graph quantity).
    if len(g_components) == 1:
        if experiments is not None and original is not None:
            v_original = frozenset(original.nodes)
            for z in sorted(experiments, key=len):
                t = v_original - z
                if s <= t:
                    try:
                        q_s = _c_factor_from(s, t, original, _Experiment(z, t))
                        return _marginal(s - y, q_s)
                    except NotIdentifiableError:
                        continue
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
    return _id(
        y,
        x & s_prime,
        graph.induced_subgraph(s_prime),
        p_prime,
        experiments=experiments,
        original=original,
    )


@overload
def identify_effect(
    graph: CausalGraph,
    treatment: Iterable[str],
    outcome: Iterable[str],
    *,
    return_certificate: Literal[True] | None = ...,
) -> Certificate: ...
@overload
def identify_effect(
    graph: CausalGraph,
    treatment: Iterable[str],
    outcome: Iterable[str],
    *,
    return_certificate: Literal[False],
) -> Estimand: ...
def identify_effect(
    graph: CausalGraph,
    treatment: Iterable[str],
    outcome: Iterable[str],
    *,
    return_certificate: bool | None = None,
) -> Estimand | Certificate:
    """Return a do-free :class:`Estimand` for ``P(outcome | do(treatment))``, or raise.

    Runs the ID algorithm. Raises :class:`NotIdentifiableError` (with the witnessing hedge attached
    as ``.witness``) when the effect is not non-parametrically identifiable, and
    :class:`CausalGraphError` for malformed inputs (unknown nodes, empty outcome, or a treatment and
    outcome that overlap).

    ``return_certificate`` (2.0 default): returns a :class:`Certificate`; pass ``False`` for the
    legacy :class:`Estimand`, or ``True`` (equivalently, ``identify_effect_certified``) to be
    explicit.
    """
    if return_certificate is None:
        return_certificate = True  # causalrl 2.0: return a Certificate by default
    if return_certificate:
        from causalrl.certify.routines import identify_effect_certified

        return identify_effect_certified(graph, treatment, outcome)
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
        identify_effect(graph, treatment, outcome, return_certificate=False)
    except NotIdentifiableError:
        return False
    return True


def identify_effect_with_experiments(
    graph: CausalGraph,
    treatment: Iterable[str],
    outcome: Iterable[str],
    experiments: Iterable[Iterable[str]],
) -> Estimand:
    """Return an :class:`Estimand` for ``P(outcome | do(treatment))`` using surrogate experiments.

    This is general identification (gID): it runs the ID recursion but, where observational data
    hits a hedge, it tries to obtain the needed c-factor from one of the available ``experiments``
    (each a set of variables you can intervene on; observational data is always available too).
    Raises :class:`NotIdentifiableError` if no combination of observational data and experiments
    identifies the effect.

    Faithful to S. Lee, J. Correa, E. Bareinboim, *General Identifiability with Arbitrary Surrogate
    Experiments*, UAI 2019, building on Tian & Pearl's c-factor identification. No code is ported.
    """
    x, y = frozenset(treatment), frozenset(outcome)
    nodes = set(graph.nodes)
    exps = [frozenset(z) for z in experiments]
    referenced = set(x) | set(y)
    for z in exps:
        referenced |= z
    unknown = referenced - nodes
    if unknown:
        raise CausalGraphError(f"unknown nodes: {sorted(unknown)}")
    if not y:
        raise CausalGraphError("outcome must be non-empty")
    if x & y:
        raise CausalGraphError(f"treatment and outcome overlap: {sorted(x & y)}")
    return _id(y, x, graph, _Joint(frozenset(graph.nodes)), experiments=exps, original=graph)


def is_gid_identifiable(
    graph: CausalGraph,
    treatment: Iterable[str],
    outcome: Iterable[str],
    experiments: Iterable[Iterable[str]],
) -> bool:
    """Whether ``P(outcome | do(treatment))`` is identifiable from data plus those experiments."""
    try:
        identify_effect_with_experiments(graph, treatment, outcome, experiments)
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
    estimand = identify_effect(graph, treatment, outcome, return_certificate=False)
    factor = _eval_observational(estimand, _empirical_joint(data, graph.nodes))
    factor = factor.condition(do)
    targets = sorted(outcome)
    factor = factor.marginalize(set(factor.variables) - set(targets)).reorder(targets)
    return dict(factor.table)


def estimate_effect_with_experiments(
    graph: CausalGraph,
    treatment: Iterable[str],
    outcome: Iterable[str],
    data: Mapping[str, np.ndarray],
    experiments_data: Mapping[frozenset[str], Mapping[str, np.ndarray]],
    *,
    do: Mapping[str, int],
) -> dict[tuple[int, ...], float]:
    """Estimate ``P(outcome | do(treatment = do))`` from observational ``data`` plus experiments.

    ``experiments_data`` maps each available intervention target (a ``frozenset`` of variables) to a
    dataset drawn from ``do(target)`` over ``graph.nodes``. Identifies the effect via gID
    (:func:`identify_effect_with_experiments`), then evaluates the estimand against the empirical
    observational joint and each experiment's empirical c-factor. Returns the outcome distribution
    keyed in ``sorted(outcome)`` order.
    """
    estimand = identify_effect_with_experiments(graph, treatment, outcome, experiments_data.keys())
    obs = _empirical_joint(data, graph.nodes)
    experiments = {
        target: _empirical_joint(rows, graph.nodes) for target, rows in experiments_data.items()
    }
    factor = _evaluate(estimand, _EvalContext(obs, experiments, {}, {})).condition(do)
    targets = sorted(outcome)
    factor = factor.marginalize(set(factor.variables) - set(targets)).reorder(targets)
    return dict(factor.table)


# --------------------------------------------------------------------------------------------------
# Transportability (sID / mz / meta): identify a target effect P*(y | do(x)) across one or more
# source domains, each a selection diagram optionally carrying surrogate experiments. Each target
# c-factor is resolved by searching the domains that can supply it; if none can, the c-factor forms
# a transport-hedge and the effect is not transportable.
# --------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Domain:
    """A source domain relative to the target: which mechanisms differ, and what data it offers.

    ``selection`` lists the variables whose mechanism differs from the target (each carries an
    implicit selection node ``S -> v``); a c-factor transfers from this domain only if it touches no
    selection-marked variable. ``experiments`` lists intervention targets available here (each a set
    of variables that can be randomized). The target domain is always implicitly available
    observationally as a fallback, so it need not be listed.
    """

    name: str
    selection: frozenset[str] = frozenset()
    experiments: frozenset[frozenset[str]] = frozenset()


def _transfers(c: frozenset[str], selection: frozenset[str]) -> bool:
    """Whether the c-factor ``Q[c]`` is invariant from a domain with the given ``selection``.

    ``Q[c]`` is a function of the structural mechanisms of ``c``'s members only, so it equals the
    target's iff none of those members is selection-marked — exactly ``c.isdisjoint(selection)``
    (the empty selection of the implicit target always transfers). At c-factor granularity this set
    test coincides with the graphical S-admissibility criterion.
    """
    return c.isdisjoint(selection)


def _resolve_c_factor(c: frozenset[str], graph: CausalGraph, domains: Sequence[Domain]) -> Estimand:
    """Obtain ``Q[c]`` from the first domain/experiment that can supply it.

    Searches the source ``domains`` (fewest shifts first) then the implicit target as a fallback. In
    each domain that transfers ``c``, tries observational data then each surrogate experiment
    (smallest target first), reusing Tian's :func:`_c_factor_from`. Raises
    :class:`NotIdentifiableError` (a transport-hedge) if no domain/experiment supplies ``Q[c]``.
    """
    v = frozenset(graph.nodes)
    ordered = [*sorted(domains, key=lambda d: len(d.selection)), Domain("target")]
    for d in ordered:
        if not _transfers(c, d.selection):
            continue
        for z in [frozenset[str](), *sorted(d.experiments, key=len)]:
            t = v - z
            if not (c <= t):
                continue
            leaf: Estimand = _DomainJoint(d.name, t) if not z else _DomainExperiment(d.name, z, t)
            try:
                return _c_factor_from(c, t, graph, leaf)
            except NotIdentifiableError:
                continue
    raise NotIdentifiableError(
        f"c-factor Q[{sorted(c)}] is not transportable from any domain (transport-hedge)",
        witness=(sorted(c), sorted(v)),
    )


def _general_transport(
    y: frozenset[str], x: frozenset[str], graph: CausalGraph, domains: Sequence[Domain]
) -> Estimand:
    """The transportability recursion: the ID-style reductions, then each target c-factor resolved
    over all domains by :func:`_resolve_c_factor`. With a single observational source it coincides
    with the sound c-factor routing; experiments and extra domains add mz / meta power."""
    v = frozenset(graph.nodes)
    if not x:
        return _marginal(v - y, _DomainJoint("target", v))

    an_y = frozenset(graph.ancestors(y))
    if v != an_y:
        sub = [Domain(d.name, d.selection & an_y, d.experiments) for d in domains]
        return _general_transport(y, x & an_y, graph.induced_subgraph(an_y), sub)

    g_cut = graph
    for node in x:
        g_cut = g_cut.remove_incoming_edges(node)
    w = (v - x) - frozenset(g_cut.ancestors(y))
    if w:
        return _general_transport(y, x | w, graph, domains)

    terms = [
        _resolve_c_factor(frozenset(c), graph, domains)
        for c in graph.induced_subgraph(v - x).c_components()
    ]
    return _marginal(v - (y | x), _product(terms))


def _transport(
    y: frozenset[str], x: frozenset[str], graph: CausalGraph, selection: frozenset[str]
) -> Estimand:
    return _general_transport(y, x, graph, [Domain("source", selection=selection)])


def identify_transport(
    graph: CausalGraph,
    treatment: Iterable[str],
    outcome: Iterable[str],
    selection: Iterable[str],
) -> Estimand:
    """Return an :class:`Estimand` for the *target* effect ``P*(outcome | do(treatment))`` across a
    selection diagram, or raise :class:`NotIdentifiableError`.

    ``selection`` lists the variables whose mechanism differs between source and target (each
    carries an implicit selection node). The target effect is decomposed into c-factors; a c-factor
    with no selection-marked variable is invariant and taken from the **source** distribution, while
    one that contains a selection-marked variable is identified from the **target**'s observational
    distribution. With an empty ``selection`` this reduces to source identification (the ID
    algorithm).

    This is the sound c-factor-routing transportability algorithm; it subsumes direct and
    S-admissible-adjustment transportability. It is not the complete sID (a c-factor identifiable
    only by combining source and target is reported as non-transportable rather than guessed).

    Faithful to J. Pearl, E. Bareinboim, *Transportability of Causal and Statistical Relations*,
    AAAI 2011 / *External Validity*, Statistical Science 2014, via Tian's c-factor decomposition.
    """
    x, y, s = frozenset(treatment), frozenset(outcome), frozenset(selection)
    unknown = (x | y | s) - set(graph.nodes)
    if unknown:
        raise CausalGraphError(f"unknown nodes: {sorted(unknown)}")
    if not y:
        raise CausalGraphError("outcome must be non-empty")
    if x & y:
        raise CausalGraphError(f"treatment and outcome overlap: {sorted(x & y)}")
    return _transport(y, x, graph, s)


def is_transportable_effect(
    graph: CausalGraph,
    treatment: Iterable[str],
    outcome: Iterable[str],
    selection: Iterable[str],
) -> bool:
    """Whether the target effect ``P*(outcome | do(treatment))`` is transportable (see
    :func:`identify_transport`)."""
    try:
        identify_transport(graph, treatment, outcome, selection)
    except NotIdentifiableError:
        return False
    return True


def identify_transport_general(
    target_graph: CausalGraph,
    treatment: Iterable[str],
    outcome: Iterable[str],
    domains: Iterable[Domain],
) -> Estimand:
    """Return an :class:`Estimand` for the target effect ``P*(outcome | do(treatment))`` combining
    one or more source ``domains`` with the target's own data, or raise
    :class:`NotIdentifiableError`.

    Generalizes :func:`identify_transport` to multiple domains (meta-transportability) and to
    surrogate experiments (mz-transportability): each target c-factor is taken from any domain whose
    mechanism for it is invariant and that can identify it from its observational or experimental
    data, with the target as the fallback. With a single observational source it coincides with
    :func:`identify_transport`; with no selection and no experiments it reduces to the ID algorithm.

    Faithful to E. Bareinboim & J. Pearl, *A General Algorithm for Deciding Transportability of
    Experimental Results* (Journal of Causal Inference 2013) and *Meta-Transportability of Causal
    Effects: A Formal Approach* (AISTATS 2013), unified via the surrogate-experiment view of S. Lee,
    J. Correa & E. Bareinboim (UAI 2019). No external code is ported.
    """
    x, y = frozenset(treatment), frozenset(outcome)
    doms = list(domains)
    referenced = set(x) | set(y)
    for d in doms:
        referenced |= d.selection
        for exp in d.experiments:
            referenced |= exp
    unknown = referenced - set(target_graph.nodes)
    if unknown:
        raise CausalGraphError(f"unknown nodes: {sorted(unknown)}")
    if not y:
        raise CausalGraphError("outcome must be non-empty")
    if x & y:
        raise CausalGraphError(f"treatment and outcome overlap: {sorted(x & y)}")
    return _general_transport(y, x, target_graph, doms)


def is_transportable_general(
    target_graph: CausalGraph,
    treatment: Iterable[str],
    outcome: Iterable[str],
    domains: Iterable[Domain],
) -> bool:
    """Whether ``P*(outcome | do(treatment))`` is transportable from ``domains`` (see
    :func:`identify_transport_general`)."""
    try:
        identify_transport_general(target_graph, treatment, outcome, domains)
    except NotIdentifiableError:
        return False
    return True


def estimate_transport_general(
    target_graph: CausalGraph,
    treatment: Iterable[str],
    outcome: Iterable[str],
    domains: Iterable[Domain],
    *,
    domain_data: Mapping[str, Mapping[str, np.ndarray]],
    experiment_data: Mapping[tuple[str, frozenset[str]], Mapping[str, np.ndarray]] | None = None,
    do: Mapping[str, int],
) -> dict[tuple[int, ...], float]:
    """Estimate the target effect ``P*(outcome | do(treatment = do))`` across ``domains``.

    ``domain_data`` maps each domain name (including ``"target"``) to an observational dataset over
    ``target_graph.nodes``; ``experiment_data`` maps ``(domain, frozenset(target))`` to a randomized
    experimental dataset. Identifies the estimand via :func:`identify_transport_general`, evaluates
    it against the supplied data, and returns the outcome distribution keyed in ``sorted(outcome)``
    order.
    """
    doms = list(domains)
    estimand = identify_transport_general(target_graph, treatment, outcome, doms)
    if "target" not in domain_data:
        raise CausalGraphError("domain_data must include 'target'")
    nodes = target_graph.nodes
    domains_f = {name: _empirical_joint(rows, nodes) for name, rows in domain_data.items()}
    exps_f = {key: _empirical_joint(rows, nodes) for key, rows in (experiment_data or {}).items()}
    ctx = _EvalContext(domains_f["target"], {}, domains_f, exps_f)
    factor = _evaluate(estimand, ctx).condition(do)
    targets = sorted(outcome)
    factor = factor.marginalize(set(factor.variables) - set(targets)).reorder(targets)
    return dict(factor.table)


def estimate_transported_effect(
    graph: CausalGraph,
    treatment: Iterable[str],
    outcome: Iterable[str],
    selection: Iterable[str],
    source_data: Mapping[str, np.ndarray],
    target_data: Mapping[str, np.ndarray],
    *,
    do: Mapping[str, int],
) -> dict[tuple[int, ...], float]:
    """Estimate the target effect ``P*(outcome | do(treatment = do))`` from source + target data.

    Identifies the transport estimand (:func:`identify_transport`), then evaluates it with source
    c-factors read from ``source_data`` and target c-factors from ``target_data`` (both discrete
    integer columns over ``graph.nodes``). Returns the outcome distribution keyed in
    ``sorted(outcome)`` order.
    """
    estimand = identify_transport(graph, treatment, outcome, selection)
    domains = {
        "source": _empirical_joint(source_data, graph.nodes),
        "target": _empirical_joint(target_data, graph.nodes),
    }
    context = _EvalContext(domains["target"], {}, domains, {})
    factor = _evaluate(estimand, context).condition(do)
    targets = sorted(outcome)
    factor = factor.marginalize(set(factor.variables) - set(targets)).reorder(targets)
    return dict(factor.table)
