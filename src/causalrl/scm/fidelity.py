"""Does the fitted model describe the regime you are querying?

A fitted SCM will answer any ``do`` query put to it. Nothing in the type says whether the
mechanism it used was ever any good *in that part of the world* -- and a mechanism fitted mostly on
one regime can be badly wrong in another while its overall holdout score looks respectable. The
concrete failure this exists to catch: a fitted model asked for ``do(stim=on)`` predicting 1.77 Hz
where the data under ``stim=on`` plainly show 5.50. That should have been caught by the library,
not by the person reading the number.

**This is a fidelity gate, not a calibrated counterfactual interval.** Conformal prediction
calibrates coverage for *factual* predictions drawn exchangeably with a calibration set; a
counterfactual query is not exchangeable with that set in general, so a "conformal counterfactual
interval" would carry a coverage claim it has not earned. What conformal-style residual checking
CAN honestly do is test the model where it is testable -- against the factual outcomes in the
regime being queried -- and downgrade the certificate when it fails. That is what happens here.

The check is deliberately narrow, which is what makes it sound. It runs the OUTCOME's own
mechanism forward at the parent values actually observed in the matching rows and compares the
predictions with the outcomes actually recorded. No back-door reasoning is involved, so a large
error means one thing only: the mechanism does not describe this regime. A caller who sees the
hedge should refit, restrict the query, or bound instead of pointing.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from causalrl.certify.certificate import (
    Assumption,
    Certificate,
    EstimandSpec,
    Hedge,
    Kind,
    Provenance,
)
from causalrl.scm.scm import StructuralCausalModel

__all__ = ["FidelityReport", "certify_fitted_query"]

_MAX_PROBE_ROWS = 64
"""Matching rows whose parent configurations are probed through the mechanism.

Each probe runs the model forward, so the cost is linear in this. Sixty-four configurations is
enough to detect a mechanism that is wrong in a regime -- the failure mode is a systematic offset,
not a subtle one -- and keeps the gate cheap enough to run on every query.
"""


@dataclass(frozen=True)
class FidelityReport:
    """How well the outcome's fitted mechanism predicts the regime being queried.

    ``standardised_error`` is the mean absolute prediction error over the probed rows, divided by
    the observed spread of the outcome in that regime. Standardising is what makes one tolerance
    meaningful across outcomes measured in different units; when the observed spread is zero it
    falls back to the absolute error, since "off by 3" against a constant is still off by 3.
    """

    outcome: str
    regime: dict[str, float]
    support: int = 0
    probed: int = 0
    predicted_mean: float = float("nan")
    observed_mean: float = float("nan")
    standardised_error: float = float("inf")
    tolerance: float = 2.0

    @property
    def trustworthy(self) -> bool:
        """Whether the mechanism predicts this regime well enough to point-answer a query in it."""
        return self.support > 0 and self.standardised_error <= self.tolerance

    def summary(self) -> str:
        if self.support == 0:
            return (
                f"{self.outcome}: NO FACTUAL SUPPORT for regime {self.regime} -- "
                "the query extrapolates entirely"
            )
        return (
            f"{self.outcome}: {'ok' if self.trustworthy else 'FAILS'} "
            f"(standardised error {self.standardised_error:.2f} vs tolerance {self.tolerance:.2f}, "
            f"predicted {self.predicted_mean:.4g} vs observed {self.observed_mean:.4g}, "
            f"support {self.support} rows)"
        )


def _matching_rows(
    data: Mapping[str, np.ndarray], intervention: Mapping[str, float], atol: float
) -> np.ndarray:
    """Row indices whose intervened variables already take the queried values."""
    n = len(next(iter(data.values())))
    mask = np.ones(n, dtype=bool)
    for name, value in intervention.items():
        column = np.asarray(data[name], dtype=np.float64).ravel()
        mask &= np.abs(column - float(value)) <= atol
    return np.flatnonzero(mask)


def certify_fitted_query(
    model: StructuralCausalModel,
    data: Mapping[str, np.ndarray],
    *,
    intervention: Mapping[str, float],
    outcome: str,
    tolerance: float = 2.0,
    atol: float = 1e-8,
    n_samples: int = 256,
    seed: int = 0,
) -> Certificate:
    """Answer ``E[outcome | do(intervention)]`` and gate it on the model's fidelity in that regime.

    Runs the query, then tests the outcome's mechanism against the rows of ``data`` where the
    intervened variables already take the queried values. A mechanism that mispredicts those rows
    by more than ``tolerance`` observed standard deviations gets a ``model-fidelity``
    :class:`~causalrl.certify.Hedge` and a ``kind`` downgraded to ``EMPIRICAL``; a regime with no
    matching rows at all gets a ``no-factual-support`` hedge, because a query there is pure
    extrapolation and nothing in the data speaks to it either way.

    The returned certificate always carries the model's answer. Gating changes what the answer is
    *claimed to be*, not whether it is reported -- a caller who wants the number anyway can still
    read it, and one who is deciding on it can read the hedge.
    """
    if not intervention:
        raise ValueError(
            "intervention must name at least one variable: with nothing intervened on there is no "
            "regime to check the model's fidelity in, and the query is an observational one."
        )
    missing = [name for name in (*intervention, outcome) if name not in data]
    if missing:
        raise KeyError(f"data is missing column(s) {sorted(missing)}")

    answered = model.do(dict(intervention)).see(n_samples, seed=seed)
    predicted = float(np.asarray(answered[outcome], dtype=np.float64).mean())

    rows = _matching_rows(data, intervention, atol)
    parents = sorted(model.graph.parents(outcome))
    observed_column = np.asarray(data[outcome], dtype=np.float64).ravel()

    if rows.size == 0:
        report = FidelityReport(
            outcome=outcome, regime=dict(intervention), support=0, tolerance=tolerance
        )
    else:
        rng = np.random.default_rng(seed)
        probe = rows if rows.size <= _MAX_PROBE_ROWS else rng.choice(rows, _MAX_PROBE_ROWS, False)
        errors: list[float] = []
        for row in probe:
            assignment = {p: float(np.asarray(data[p]).ravel()[row]) for p in parents}
            drawn = model.do(assignment).see(n_samples, seed=seed) if assignment else answered
            predicted_row = float(np.asarray(drawn[outcome], dtype=np.float64).mean())
            errors.append(abs(predicted_row - float(observed_column[row])))
        observed = observed_column[rows]
        spread = float(observed.std())
        mean_error = float(np.mean(errors))
        report = FidelityReport(
            outcome=outcome,
            regime=dict(intervention),
            support=int(rows.size),
            probed=len(errors),
            predicted_mean=predicted,
            observed_mean=float(observed.mean()),
            standardised_error=mean_error / spread if spread > 0.0 else mean_error,
            tolerance=tolerance,
        )

    hedge: Hedge | None = None
    if report.support == 0:
        hedge = Hedge(
            reason=(
                "no-factual-support: the logs contain no row in the queried regime, so this answer "
                "is pure extrapolation and the model's fit there is untested"
            ),
            detail={"regime": dict(intervention)},
            downgraded_from="fitted",
        )
    elif not report.trustworthy:
        hedge = Hedge(
            reason=(
                "model-fidelity: the outcome's fitted mechanism mispredicts the factual rows of "
                f"this regime by {report.standardised_error:.2f} observed standard deviations "
                f"(tolerance {tolerance:.2f}), so its interventional answer here is not to be "
                "read as a point estimate"
            ),
            detail={
                "predicted_mean": report.predicted_mean,
                "observed_mean": report.observed_mean,
                "support": report.support,
            },
            downgraded_from="fitted",
        )

    return Certificate(
        claim=(
            f"E[{outcome} | do({', '.join(f'{k}={v}' for k, v in intervention.items())})] "
            f"= {predicted:.6g} from a fitted SCM; {report.summary()}"
        ),
        estimand=EstimandSpec(query="do", target="mean", domains=(outcome,)),
        kind=Kind.IDENTIFIED if hedge is None else Kind.EMPIRICAL,
        value=None,
        alpha=None,
        assumptions=(
            Assumption(
                name="mechanism-fidelity-in-regime",
                params={"tolerance": tolerance},
                checkable=True,
                diagnostic={
                    "standardised_error": report.standardised_error,
                    "support": report.support,
                    "probed": report.probed,
                },
            ),
        ),
        method="fitted-scm-query-with-fidelity-gate",
        witness=None,
        hedge=hedge,
        provenance=Provenance.create(),
    )
