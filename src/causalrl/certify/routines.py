"""Certificate-returning variants of shipped inferential routines (plan §I1, Phase-0 Task 3).

Additive and opt-in: the bare functions are untouched (their internal callers and the
``certify_decision`` byte-pin stay byte-identical). In causalrl 2.0 the bare functions will return
certificates by default (a matured deprecation, tracked in the CHANGELOG); the eager warning is
introduced in a pre-2.0 minor (emitting it here would cascade through the internal bounds call
graph, including the byte-pinned path). ``certify`` -> ``identification`` is one-way, so no cycle.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from causalrl.certify.certificate import (
    Assumption,
    Certificate,
    EstimandSpec,
    Kind,
    Provenance,
    Witness,
)
from causalrl.graphs import graph_hash
from causalrl.identification.bounds import (
    Interval,
    ipw_sensitivity_bounds,
    msm_policy_value_bounds,
)
from causalrl.identification.id_algorithm import identify_effect
from causalrl.scm.graph import CausalGraph


def identify_effect_certified(
    graph: CausalGraph, treatment: Iterable[str], outcome: Iterable[str]
) -> Certificate:
    """`identify_effect` as an ``IDENTIFIED`` certificate; raises ``NotIdentifiableError`` if not.

    The witness carries the do-free identification formula. Identification is symbolic, so ``value``
    is ``None`` (there is no numeric estimate to report — estimation is Phase 1).
    """
    treat = sorted(treatment)
    out = sorted(outcome)
    estimand = identify_effect(graph, treat, out)  # raises NotIdentifiableError (witnessing hedge)
    return Certificate(
        claim=f"P({','.join(out)} | do({','.join(treat)})) is identified",
        estimand=EstimandSpec(query="do", target="mean", domains=tuple(out)),
        kind=Kind.IDENTIFIED,
        value=None,
        alpha=None,
        assumptions=(),
        method="identify_effect",
        witness=Witness(
            "id-formula", {"formula": estimand.render(), "treatment": treat, "outcome": out}
        ),
        hedge=None,
        provenance=Provenance.create(graph_hash=graph_hash(graph)),
    )


def ipw_sensitivity_bounds_certified(
    outcomes: Sequence[float], propensities: Sequence[float], *, gamma: float
) -> Certificate:
    """`ipw_sensitivity_bounds` as a ``BOUNDED`` certificate (Tan MSM on ``E[Y(1)]``)."""
    interval: Interval = ipw_sensitivity_bounds(outcomes, propensities, gamma=gamma)
    return Certificate(
        claim=f"E[Y(1)] bounded under the marginal sensitivity model (gamma={gamma})",
        estimand=EstimandSpec(query="do", target="mean"),
        kind=Kind.BOUNDED,
        value=interval,
        alpha=None,
        assumptions=(Assumption("MSM", {"gamma": gamma}),),
        method="ipw_sensitivity_bounds",
        witness=None,
        hedge=None,
        provenance=Provenance.create(),
    )


def msm_policy_value_bounds_certified(
    outcomes: Sequence[float],
    logging_propensities: Sequence[float],
    target_propensities: Sequence[float],
    *,
    gamma: float,
) -> Certificate:
    """`msm_policy_value_bounds` as a ``BOUNDED`` certificate (off-policy value under Tan MSM)."""
    interval: Interval = msm_policy_value_bounds(
        outcomes, logging_propensities, target_propensities, gamma=gamma
    )
    return Certificate(
        claim=f"V(pi_t) bounded under the marginal sensitivity model (gamma={gamma})",
        estimand=EstimandSpec(query="policy_value", target="mean", policy="pi_t"),
        kind=Kind.BOUNDED,
        value=interval,
        alpha=None,
        assumptions=(Assumption("MSM", {"gamma": gamma}),),
        method="msm_policy_value_bounds",
        witness=None,
        hedge=None,
        provenance=Provenance.create(),
    )
