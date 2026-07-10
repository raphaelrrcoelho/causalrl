"""Estimation layer for transportability (plan §7.5).

The shipped :func:`~causalrl.identification.transport.transport_formula` decides *whether* an effect
``P*(outcome | do(treatment))`` transports across a selection diagram (source vs target mechanisms)
and returns the closed form; :func:`~causalrl.identification.transport.transported_effect` estimates
it from two :class:`StructuralCausalModel` objects. This module adds the **data-plane** estimator:
given source and target *observational data* (numpy, discrete adjustment covariates), it estimates
the transported mean and returns a unified :class:`Certificate`:

* transportable ``direct`` -> the source interventional mean (back-door g-computation) transfers.
* transportable ``adjustment`` -> source conditionals reweighted by the target covariate marginal.
* not transportable -> a hedge (I3); ``transport_regret_certificate`` is the shipped floor.

Discrete adjustment covariates are assumed (as in the shipped ``transported_effect``); the estimate
is g-computation over their strata, renormalised over strata present in both samples.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from causalrl.certify.certificate import (
    Assumption,
    Certificate,
    EstimandSpec,
    Hedge,
    Kind,
    Provenance,
    Witness,
)
from causalrl.estimate.sequential import OutcomeFactory, sequential_ice_values
from causalrl.exceptions import NotIdentifiableError
from causalrl.graphs import graph_hash
from causalrl.identification.criteria import backdoor_adjustment_set
from causalrl.identification.transport import SelectionDiagram, transport_formula

__all__ = [
    "certify_sequential_transport",
    "certify_transported_effect",
    "transport_gcomp",
    "transport_sequential_gcomp",
]

FloatArray = NDArray[np.float64]


def _strata_keys(data: Mapping[str, Any], z_vars: list[str]) -> list[tuple[int, ...]]:
    cols = [np.rint(np.asarray(data[z], dtype=np.float64)).astype(int).tolist() for z in z_vars]
    return [tuple(k) for k in zip(*cols, strict=True)]


def transport_gcomp(
    conditional: Mapping[str, Any],
    marginal: Mapping[str, Any],
    *,
    treatment: str,
    outcome: str,
    adjustment: list[str],
    treated_value: float = 1.0,
) -> float:
    """g-computation ``sum_z E_conditional[outcome | treatment=treated_value, z] P_marginal(z)``.

    ``conditional`` supplies the outcome conditionals (the source); ``marginal`` supplies ``P(z)``
    (the source itself for ``direct`` transport, the target for ``adjustment``). Renormalises over
    strata present in both. Raises ``ValueError`` if the treated arm or the overlap is empty.
    """
    xc = np.asarray(conditional[treatment], dtype=np.float64)
    yc = np.asarray(conditional[outcome], dtype=np.float64)
    treated = xc == treated_value
    if not bool(treated.any()):
        raise ValueError(f"no source units at {treatment}={treated_value}")
    if not adjustment:
        return float(yc[treated].mean())

    keys_c = _strata_keys(conditional, adjustment)
    keys_m = _strata_keys(marginal, adjustment)
    sums: dict[tuple[int, ...], float] = {}
    counts: dict[tuple[int, ...], int] = {}
    for k, y, t in zip(keys_c, yc.tolist(), treated.tolist(), strict=True):
        if t:
            sums[k] = sums.get(k, 0.0) + y
            counts[k] = counts.get(k, 0) + 1

    marginal_counts = Counter(keys_m)
    n_marginal = len(keys_m)
    total = 0.0
    weight = 0.0
    for k, cnt in marginal_counts.items():
        if k in counts:
            p_z = cnt / n_marginal
            total += (sums[k] / counts[k]) * p_z
            weight += p_z
    if weight == 0.0:
        raise ValueError("no adjustment strata shared between source and marginal samples")
    return total / weight


def _hedged(
    treatment: str, outcome: str, selection: list[str], diagram: SelectionDiagram, reason: str
) -> Certificate:
    return Certificate(
        claim=f"P*({outcome} | do({treatment})) refused: {reason}",
        estimand=EstimandSpec(query="transport", target="mean", domains=("source", "target")),
        kind=Kind.IDENTIFIED,
        value=None,
        alpha=None,
        assumptions=(),
        method="refused",
        witness=None,
        hedge=Hedge(
            reason=reason,
            detail={"selection": selection, "fallback": "transport_regret_certificate"},
        ),
        provenance=Provenance.create(graph_hash=graph_hash(diagram.graph)),
    )


def certify_transported_effect(
    diagram: SelectionDiagram,
    source_data: Mapping[str, Any],
    target_data: Mapping[str, Any],
    *,
    treatment: str,
    outcome: str,
    treated_value: float = 1.0,
    max_adjustment_size: int = 3,
) -> Certificate:
    """Certify the transported mean ``E*[outcome | do(treatment=treated_value)]`` from data.

    Uses the shipped :func:`~causalrl.identification.transport.transport_formula` to decide
    transportability, then estimates the closed form from ``source_data`` / ``target_data`` (numpy,
    discrete adjustment covariates). Returns a ``kind=IDENTIFIED`` certificate with the point value
    and the transport witness, or a hedged certificate when the effect is not transportable (I3).
    """
    formula = transport_formula(
        diagram, treatment=treatment, outcome=outcome, max_adjustment_size=max_adjustment_size
    )
    selection = sorted(diagram.selection_variables)
    if formula is None:
        return _hedged(treatment, outcome, selection, diagram, "non-transportable")

    if formula.kind == "direct":
        try:
            adjustment = sorted(backdoor_adjustment_set(diagram.graph, treatment, outcome))
        except NotIdentifiableError:
            return _hedged(treatment, outcome, selection, diagram, "source-effect-unsupported")
        marginal: Mapping[str, Any] = source_data  # direct: source interventional law transfers
        witness_kind = "direct-transport"
    else:
        adjustment = sorted(formula.adjustment_set)
        marginal = target_data  # adjustment: reweight by the target covariate marginal
        witness_kind = "s-admissible-adjustment"

    value = transport_gcomp(
        source_data,
        marginal,
        treatment=treatment,
        outcome=outcome,
        adjustment=adjustment,
        treated_value=treated_value,
    )
    return Certificate(
        claim=f"E*[{outcome} | do({treatment}={treated_value:g})] = {value:.4g}",
        estimand=EstimandSpec(query="transport", target="mean", domains=("source", "target")),
        kind=Kind.IDENTIFIED,
        value=value,
        alpha=None,
        assumptions=(
            Assumption(name="selection-diagram", params={"selection": selection}, checkable=True),
        ),
        method=f"transport-{formula.kind}",
        witness=Witness(
            kind=witness_kind,
            detail={
                "adjustment_set": adjustment,
                "selection": selection,
                "formula": formula.expression,
            },
        ),
        hedge=None,
        provenance=Provenance.create(graph_hash=graph_hash(diagram.graph)),
    )


def _stack_named(data: Mapping[str, Any], names: list[str]) -> FloatArray:
    return np.column_stack([np.asarray(data[nm], dtype=np.float64) for nm in names])


def transport_sequential_gcomp(
    source_values: Sequence[float],
    baseline_source: list[tuple[int, ...]],
    baseline_target: list[tuple[int, ...]],
) -> float:
    """Reweight per-unit source ICE values to the target baseline marginal over discrete strata.

    ``source_values[i]`` is a source unit's iterated-conditional-expectation policy value given its
    baseline stratum ``baseline_source[i]``; averaging within stratum and reweighting by the target
    stratum marginal transports the sequential policy value to the target baseline distribution.
    Renormalises over shared strata. An empty baseline (no shift) reduces to the source mean.
    """
    sums: dict[tuple[int, ...], float] = {}
    counts: dict[tuple[int, ...], int] = {}
    for k, q in zip(baseline_source, source_values, strict=True):
        sums[k] = sums.get(k, 0.0) + float(q)
        counts[k] = counts.get(k, 0) + 1
    target_counts = Counter(baseline_target)
    n_target = len(baseline_target)
    total = 0.0
    weight = 0.0
    for k, cnt in target_counts.items():
        if k in counts:
            p_z = cnt / n_target
            total += (sums[k] / counts[k]) * p_z
            weight += p_z
    if weight == 0.0:
        raise ValueError("no baseline strata shared between source and target samples")
    return total / weight


def _hedged_sequential(
    diagram: SelectionDiagram, selection: list[str], horizon: int, *, reason: str
) -> Certificate:
    return Certificate(
        claim=f"V*(pi) over horizon {horizon} refused: {reason}",
        estimand=EstimandSpec(query="policy_value", target="mean", domains=("source", "target")),
        kind=Kind.IDENTIFIED,
        value=None,
        alpha=None,
        assumptions=(),
        method="refused",
        witness=None,
        hedge=Hedge(
            reason=reason,
            detail={
                "selection": selection,
                "fallback": "transport_regret_certificate",
                "note": (
                    "sequential/policy-value transport is identified here only when the selection "
                    "difference is confined to the baseline distribution"
                ),
            },
        ),
        provenance=Provenance.create(graph_hash=graph_hash(diagram.graph)),
    )


def certify_sequential_transport(
    diagram: SelectionDiagram,
    source_data: Mapping[str, Any],
    target_data: Mapping[str, Any],
    *,
    stages: Sequence[Mapping[str, Any]],
    outcome: str,
    target_actions: Sequence[float],
    alpha: float = 0.05,
    outcome_model: OutcomeFactory | None = None,
    policy: str = "pi",
) -> Certificate:
    """Certify a deterministic policy's transported finite-horizon value (hedge-first; §7.5).

    ``stages[t]`` is ``{"history": (names...), "treatment": name}`` for stage ``t``;
    ``target_actions`` gives a constant target action per stage. Sequential/policy-value transport
    is research-grade, so this is **identified only in one subcase** — when the selection difference
    is confined to the baseline distribution ``H_1`` (a population shift, downstream mechanisms
    shared): the source sequential g-computation is reweighted to the target baseline marginal
    (discrete baseline strata, as in :func:`certify_transported_effect`). Any selection node on a
    time-varying covariate, treatment, or the outcome yields a **hedge** (I3), with
    ``transport_regret_certificate`` the shipped floor. Sequential ignorability is recorded as a
    non-checkable assumption.
    """
    if not stages:
        raise ValueError("need at least one stage")
    if len(target_actions) != len(stages):
        raise ValueError("target_actions must have one action per stage")
    baseline_vars = [str(v) for v in stages[0]["history"]]
    selection = frozenset(diagram.selection_variables)
    horizon = len(stages)

    if not selection <= set(baseline_vars):
        return _hedged_sequential(
            diagram, sorted(selection), horizon, reason="non-transportable-sequential"
        )

    y_src = np.asarray(source_data[outcome], dtype=np.float64)
    n_src = len(y_src)
    histories = [_stack_named(source_data, list(s["history"])) for s in stages]
    treatments = [np.asarray(source_data[s["treatment"]], dtype=np.float64) for s in stages]
    tgt_actions = [np.full(n_src, float(a), dtype=np.float64) for a in target_actions]
    q1 = sequential_ice_values(
        histories, treatments, tgt_actions, y_src, outcome_model=outcome_model
    )

    if baseline_vars:
        baseline_src = _strata_keys(source_data, baseline_vars)
        baseline_tgt = _strata_keys(target_data, baseline_vars)
    else:
        n_tgt = len(np.asarray(target_data[outcome]))
        baseline_src = [() for _ in range(n_src)]
        baseline_tgt = [() for _ in range(n_tgt)]
    value = transport_sequential_gcomp(q1.tolist(), baseline_src, baseline_tgt)

    return Certificate(
        claim=f"V*({policy}) over horizon {horizon} = {value:.4g}",
        estimand=EstimandSpec(
            query="policy_value", target="mean", policy=policy, domains=("source", "target")
        ),
        kind=Kind.IDENTIFIED,
        value=value,
        alpha=alpha,
        assumptions=(
            Assumption("sequential-ignorability", {"horizon": horizon}, checkable=False),
            Assumption(
                "selection-diagram",
                {"selection": sorted(selection), "scope": "baseline-only"},
                checkable=True,
            ),
        ),
        method="sequential-transport-gcomp",
        witness=Witness(
            kind="sequential-baseline-transport",
            detail={"baseline": baseline_vars, "selection": sorted(selection), "horizon": horizon},
        ),
        hedge=None,
        provenance=Provenance.create(graph_hash=graph_hash(diagram.graph)),
    )
