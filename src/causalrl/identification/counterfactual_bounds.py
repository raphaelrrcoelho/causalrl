# src/causalrl/identification/counterfactual_bounds.py
"""Identified counterfactual bounds on a *fitted* SCM.

Kept out of ``counterfactual.py``, which is torch-free at runtime: evaluating a fitted mechanism
needs torch, and that module's import-time weight is part of the torch-optional surface.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import NamedTuple, cast

import numpy as np
import torch

from causalrl.exceptions import NotIdentifiableError, RealizabilityError
from causalrl.identification.bounds import Interval
from causalrl.scm.scm import StructuralCausalModel, Value

Tensor = torch.Tensor
_Residual = Callable[[dict[str, Tensor], Tensor], Tensor]

__all__ = ["CounterfactualBound", "counterfactual_interval"]


class CounterfactualBound(NamedTuple):
    """Bounds on a counterfactual expectation, with whether they are tight.

    A separate type from :class:`~causalrl.identification.bounds.Interval`, whose documented
    tuple-unpacking contract (``lo, hi = interval``) a third field would break. ``tight`` is
    ``False`` only for bounds this library knows it cannot close; phase-1
    ``counterfactual_interval`` refuses such queries rather than returning a loose answer, so it
    always reports ``True``.
    """

    lower: float
    upper: float
    tight: bool

    @property
    def interval(self) -> Interval:
        """The bound as a plain :class:`Interval`, for the partial-identification surface."""
        return Interval(self.lower, self.upper)


def counterfactual_interval(
    scm: StructuralCausalModel,
    *,
    evidence: Mapping[str, float],
    interventions: Mapping[str, Value],
    target: str,
    n: int = 20_000,
    seed: int | None = None,
) -> CounterfactualBound:
    """Bound ``E[target_{do(interventions)} | evidence]`` on a fitted SCM.

    A fitted discrete mechanism pins ``P(V | parents)`` but not the coupling from noise to value,
    and the two are indistinguishable in L1 and L2 data. So the counterfactual is an interval over
    admissible couplings, not a point. Its extremes solve a box-and-sum problem: the factual and
    counterfactual columns of the coupling share known marginals, which bounds each cell, and the
    conditioning event fixes their sum.

    Invertible mechanisms contribute no width — their noise is recoverable — so an all-invertible
    SCM returns a point, matching :meth:`StructuralCausalModel.abduct` exactly.
    """
    if scm.provenance != "fitted":
        raise ValueError(
            "counterfactual_interval is for fitted SCMs, whose noise-to-value coupling is not "
            "identified. A specified SCM's mechanisms are asserted by their author, so its "
            "counterfactual is a point — use counterfactual_expectation instead."
        )
    missing = [node for node in scm.graph.nodes if node not in evidence]
    if missing:
        raise KeyError(
            f"evidence must cover every node for a counterfactual on a fully observed unit; "
            f"missing {sorted(missing)}"
        )

    intervened = set(interventions)
    ambiguous = set(scm.non_invertible_nodes())
    upstream = _ambiguous_upstream(scm, ambiguous, intervened, target)
    if upstream:
        raise NotIdentifiableError(
            f"node(s) {sorted(upstream)} are non-invertible and lie upstream of {target!r} on a "
            "directed path from the intervention; composing per-node bounds would be loose rather "
            "than tight. The tight answer needs the neural-causal-model min/max.",
            witness=sorted(upstream),
        )

    cf_parents = _counterfactual_parents(scm, target, evidence, interventions)
    factual_parents = {p: float(evidence[p]) for p in scm.graph.parents(target)}

    if target not in ambiguous:
        # Exact: invert the target's noise from the factual unit, replay it on the cf parents.
        mechanism = scm.mechanisms[target]
        # residual is attached dynamically (Tasks 4-5), so it is unknown to the Mechanism
        # protocol -- cast rather than annotate, matching evaluate_holdout's log_prob pattern
        # (fitters.py), so the Unknown does not propagate into the mechanism(...) call below.
        residual = cast("_Residual", mechanism.residual)  # type: ignore[attr-defined]
        noise = residual(_unit(factual_parents), _unit({target: float(evidence[target])})[target])
        value = float(mechanism(_unit(cf_parents), noise).reshape(-1)[0])
        return CounterfactualBound(value, value, True)

    factual = _pmf_at(scm, target, factual_parents, n=n, seed=seed)
    counterfactual = _pmf_at(scm, target, cf_parents, n=n, seed=seed)
    observed = float(evidence[target])
    p_factual = factual.get(observed, 0.0)
    if p_factual <= 0.0:
        raise RealizabilityError(
            f"evidence {target}={observed} has zero fitted probability at the factual parent "
            f"configuration {factual_parents}; the counterfactual is not defined"
        )

    values = np.array(sorted(counterfactual), dtype=float)
    marginal = np.array([counterfactual[v] for v in values], dtype=float)
    # Cell pi(v) = P(target_cf = v, target_f = observed). Each cell is capped by its own marginal
    # and by the conditioning mass, floored by what the rest of the table cannot absorb, and the
    # cells sum to p_factual — a box-and-sum problem solved exactly by a fractional fill.
    upper_cell = np.minimum(marginal, p_factual)
    lower_cell = np.maximum(0.0, marginal - (1.0 - p_factual))
    low = _extreme_under_sum(values, lower_cell, upper_cell, p_factual, maximize=False)
    high = _extreme_under_sum(values, lower_cell, upper_cell, p_factual, maximize=True)
    return CounterfactualBound(float(low / p_factual), float(high / p_factual), True)


def _extreme_under_sum(
    values: np.ndarray, lo: np.ndarray, hi: np.ndarray, total: float, *, maximize: bool
) -> float:
    """Extreme of ``sum(values * pi)`` over ``lo <= pi <= hi`` with ``sum(pi) == total``.

    A fractional knapsack: start every cell at its floor, then pour the remaining mass into cells in
    value order. Exact, so the resulting bound is tight.

    Not :func:`causalrl.identification.bounds._fractional_extreme` — that one extremizes a *ratio*
    with a free denominator, whereas the conditioning event fixes this sum. Reusing it would give a
    valid but loose interval, and this function's whole contract is tightness.
    """
    order = np.argsort(-values if maximize else values)
    allocation = lo.astype(float).copy()
    remaining = total - float(allocation.sum())
    for index in order:
        if remaining <= 0.0:
            break
        room = float(hi[index] - lo[index])
        added = min(room, remaining)
        allocation[index] += added
        remaining -= added
    return float(np.sum(values * allocation))


def _unit(assignment: Mapping[str, float]) -> dict[str, Tensor]:
    """One-sample tensors, the granularity of a counterfactual on a single observed unit."""
    return {
        name: torch.tensor([float(value)], dtype=torch.float32)  # type: ignore[reportPrivateImportUsage]
        for name, value in assignment.items()
    }


def _ambiguous_upstream(
    scm: StructuralCausalModel, ambiguous: set[str], intervened: set[str], target: str
) -> set[str]:
    """Non-invertible nodes strictly between an intervened node and ``target``.

    A non-invertible node that the intervention does not reach keeps its factual value, so it costs
    nothing; only one that is both downstream of the intervention and upstream of the target forces
    a composition this phase refuses to approximate.
    """
    downstream = scm.graph.descendants(intervened) - intervened
    ancestors = scm.graph.ancestors(target) - {target}
    return (ambiguous & downstream & ancestors) - intervened


def _counterfactual_parents(
    scm: StructuralCausalModel,
    target: str,
    evidence: Mapping[str, float],
    interventions: Mapping[str, Value],
) -> dict[str, float]:
    """The target's parent values in the counterfactual world, for one observed unit.

    An intervened parent takes its assigned value; a parent the intervention cannot reach keeps its
    factual value; anything in between is invertible (``_ambiguous_upstream`` refused otherwise), so
    its noise is recovered from the factual unit and replayed forward.

    Resolution is scoped to ``target``'s ancestors, matching exactly what ``_ambiguous_upstream``
    checked. A node the intervention reaches but that is not an ancestor of ``target`` (a sibling
    branch) may be non-invertible without blocking this query — it is never on a path to ``target``,
    so its own counterfactual value is irrelevant and must never be touched, let alone inverted.
    """
    intervened = set(interventions)
    affected = scm.graph.descendants(intervened) - intervened
    needed = scm.graph.ancestors(target) - {target}
    resolved: dict[str, float] = {}
    for node in scm.graph.topological_order():
        if node == target or node not in needed:
            continue
        if node in intervened:
            resolved[node] = float(np.asarray(interventions[node]).reshape(-1)[0])
        elif node not in affected:
            resolved[node] = float(evidence[node])
        else:
            mechanism = scm.mechanisms[node]
            parents = scm.graph.parents(node)
            residual = cast("_Residual", mechanism.residual)  # type: ignore[attr-defined]
            noise = residual(
                _unit({p: float(evidence[p]) for p in parents}),
                _unit({node: float(evidence[node])})[node],
            )
            cf_parents = _unit({p: resolved[p] for p in parents})
            resolved[node] = float(mechanism(cf_parents, noise).reshape(-1)[0])
    return {p: resolved[p] for p in scm.graph.parents(target)}


def _pmf_at(
    scm: StructuralCausalModel,
    target: str,
    parents: Mapping[str, float],
    *,
    n: int,
    seed: int | None,
) -> dict[float, float]:
    """The fitted ``P(target | parents = this configuration)``, read off the mechanism directly."""
    generator = torch.Generator()  # type: ignore[reportPrivateImportUsage]
    if seed is not None:
        generator.manual_seed(seed)
    with torch.random.fork_rng():  # type: ignore[reportUnknownMemberType]
        torch.random.set_rng_state(generator.get_state())  # type: ignore[reportUnknownMemberType]
        noise = scm.exogenous[target].sample((n,)).reshape(n).float()
    columns = {
        name: torch.full((n,), float(value))  # type: ignore[reportPrivateImportUsage]
        for name, value in parents.items()
    }
    drawn = scm.mechanisms[target](columns, noise).reshape(-1).numpy()
    values, counts = np.unique(drawn, return_counts=True)
    return {float(v): float(c) / len(drawn) for v, c in zip(values, counts, strict=True)}
