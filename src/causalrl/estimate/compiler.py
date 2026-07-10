"""Estimand compiler + ``certify_effect`` front door (§7.2).

Turns a graph query ``P(outcome | do(treatment))`` into an estimator plan by consuming the shipped
identification machinery, then estimates it and returns a unified :class:`Certificate`:

* not identifiable -> ``NotIdentifiableError`` caught, hedged certificate (I3; acceptance d)
* identified but not by parent adjustment (front-door / general ID) -> hedged certificate (honest;
  the DR estimators only implement back-door adjustment)
* identified back-door + adequate overlap -> ``kind=IDENTIFIED`` certificate carrying the DR/DML
  point estimate, its confidence interval, the adjustment-set witness, and provenance
* identified back-door but destroyed positivity -> hedged certificate (I3)

The estimation is never silent: any query we cannot back-door-estimate yields a hedge, never a
point.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from causalrl.certify.certificate import (
    Assumption,
    Certificate,
    EstimandSpec,
    Hedge,
    Kind,
    Provenance,
    Witness,
)
from causalrl.estimate.estimators import (
    EffectEstimate,
    OutcomeFactory,
    PropensityFactory,
    estimate_ate,
)
from causalrl.exceptions import NotIdentifiableError
from causalrl.graphs import graph_hash
from causalrl.identification.criteria import backdoor_adjustment_set
from causalrl.identification.id_algorithm import identify_effect
from causalrl.scm.graph import CausalGraph

__all__ = ["EstimandNotSupportedError", "EstimatorPlan", "certify_effect", "compile_estimand"]


class EstimandNotSupportedError(Exception):
    """The effect is identifiable, but not by the back-door adjustment the estimators implement
    (e.g. it needs front-door or general ID). Raised by :func:`compile_estimand`; certifiers turn
    it into an honest hedge rather than an incorrect estimate."""


@dataclass(frozen=True)
class EstimatorPlan:
    """An identified effect's estimator plan: admissible adjustment set + ID formula string."""

    adjustment_set: tuple[str, ...]
    estimand_render: str


def compile_estimand(graph: CausalGraph, treatment: str, outcome: str) -> EstimatorPlan:
    """Compile ``P(outcome | do(treatment))`` into a back-door :class:`EstimatorPlan`.

    Raises :class:`NotIdentifiableError` if the effect is not identifiable at all, or
    :class:`EstimandNotSupportedError` if it is identifiable only by a method the estimators do not
    implement (front-door / general ID).
    """
    estimand = identify_effect(
        graph, [treatment], [outcome]
    )  # NotIdentifiableError if unidentified
    try:
        adjustment = backdoor_adjustment_set(graph, treatment, outcome)
    except NotIdentifiableError as exc:
        raise EstimandNotSupportedError(
            f"P({outcome} | do({treatment})) is identifiable but needs front-door/general ID, "
            f"not parent/back-door adjustment: {exc}"
        ) from exc
    return EstimatorPlan(
        adjustment_set=tuple(sorted(adjustment)), estimand_render=estimand.render()
    )


def _data_fingerprint(data: Mapping[str, Any]) -> str:
    h = hashlib.sha256()
    for name in sorted(data):
        h.update(name.encode())
        h.update(np.ascontiguousarray(np.asarray(data[name], dtype=float)).tobytes())
    return h.hexdigest()[:16]


def _hedged(
    treatment: str, outcome: str, alpha: float, hedge: Hedge, graph: CausalGraph
) -> Certificate:
    return Certificate(
        claim=f"P({outcome} | do({treatment})) refused: {hedge.reason}",
        estimand=EstimandSpec(query="do", target="mean"),
        kind=Kind.IDENTIFIED,
        value=None,
        alpha=alpha,
        assumptions=(),
        method="refused",
        witness=None,
        hedge=hedge,
        provenance=Provenance.create(graph_hash=graph_hash(graph)),
        ci=None,
    )


def certify_effect(
    graph: CausalGraph,
    treatment: str,
    outcome: str,
    data: Mapping[str, Any],
    *,
    method: str = "dml",
    alpha: float = 0.05,
    n_folds: int = 5,
    seed: int = 0,
    outcome_model: OutcomeFactory | None = None,
    propensity_model: PropensityFactory | None = None,
    overlap_eps: float = 0.01,
    clip: float = 1e-3,
) -> Certificate:
    """Certify the back-door ATE ``E[Y|do(X=1)] - E[Y|do(X=0)]`` of a binary ``treatment``.

    Returns a ``kind=IDENTIFIED`` :class:`Certificate` with the point estimate (``value``), its
    confidence interval (``ci``), the adjustment-set witness, and provenance — or a hedged
    certificate (``value=None``, ``hedge`` set) when the effect is not identifiable, needs an
    unsupported ID method, or when estimated positivity falls below ``overlap_eps`` (I3). ``data``
    maps variable names to arrays (see :func:`~causalrl.estimate.estimators.estimate_ate`).
    """
    try:
        plan = compile_estimand(graph, treatment, outcome)
    except NotIdentifiableError as exc:
        return _hedged(
            treatment,
            outcome,
            alpha,
            Hedge(reason="not-identifiable", detail={"message": str(exc)}),
            graph,
        )
    except EstimandNotSupportedError as exc:
        return _hedged(
            treatment,
            outcome,
            alpha,
            Hedge(reason="estimand-unsupported", detail={"message": str(exc)}),
            graph,
        )

    est: EffectEstimate = estimate_ate(
        data,
        treatment,
        outcome,
        plan.adjustment_set,
        method=method,
        alpha=alpha,
        n_folds=n_folds,
        seed=seed,
        outcome_model=outcome_model,
        propensity_model=propensity_model,
        clip=clip,
    )

    min_e = est.overlap.get("min_propensity", float("nan"))
    max_e = est.overlap.get("max_propensity", float("nan"))
    if np.isfinite(min_e) and (min_e < overlap_eps or max_e > 1.0 - overlap_eps):
        return _hedged(
            treatment,
            outcome,
            alpha,
            Hedge(
                reason="overlap-violation",
                detail={"overlap_eps": overlap_eps, **est.overlap},
            ),
            graph,
        )

    return Certificate(
        claim=f"E[{outcome}|do({treatment})=1] - E[{outcome}|do({treatment})=0] = {est.value:.4g}",
        estimand=EstimandSpec(query="do", target="mean"),
        kind=Kind.IDENTIFIED,
        value=est.value,
        alpha=alpha,
        assumptions=(
            Assumption(
                name="backdoor",
                params={"adjustment_set": list(plan.adjustment_set)},
                checkable=True,
            ),
            Assumption(
                name="overlap",
                params={"eps": overlap_eps},
                checkable=True,
                diagnostic=est.overlap,
            ),
        ),
        method=f"{est.method}" + (f" (cross-fit K={est.n_folds})" if est.n_folds else ""),
        witness=Witness(
            kind="adjustment",
            detail={"set": list(plan.adjustment_set), "estimand": plan.estimand_render},
        ),
        hedge=None,
        provenance=Provenance.create(
            seeds=(seed,),
            data_fingerprint=_data_fingerprint(data),
            graph_hash=graph_hash(graph),
        ),
        ci=est.ci,
    )
