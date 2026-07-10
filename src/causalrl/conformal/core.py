"""Split / weighted conformal prediction intervals (plan §7.4).

Distribution-free finite-sample coverage: given exchangeable calibration nonconformity scores, the
conformal quantile yields an interval whose marginal coverage is at least ``1 - alpha``. Provided:

* ``conformal_quantile`` — the (optionally weighted) conformal quantile of scores, with the standard
  ``+inf`` test-point mass so coverage is finite-sample valid (Vovk et al.; Tibshirani et al. 2019
  for the weighted / covariate-shift case).
* ``split_conformal_interval`` — symmetric residual interval around a point prediction.
* ``cqr_interval`` — conformalized quantile regression (Romano, Patterson & Candes 2019): adaptive
  intervals for heteroscedastic / heavy-tailed targets.
* ``certify_conformal_interval`` — wraps the above in a ``kind=EMPIRICAL`` certificate that records
  the (weighted-)exchangeability assumption; the natural finite-sample layer for a per-decision
  counterfactual or an off-policy (covariate-shifted) prediction.

Formula-level implementations of the cited methods; no third-party code is ported.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from causalrl.certify.certificate import (
    Assumption,
    Certificate,
    EstimandSpec,
    Kind,
    Provenance,
)
from causalrl.identification.bounds import Interval

__all__ = [
    "certify_conformal_interval",
    "conformal_quantile",
    "cqr_interval",
    "split_conformal_interval",
]


def conformal_quantile(
    scores: Sequence[float],
    alpha: float = 0.1,
    *,
    weights: Sequence[float] | None = None,
    test_weight: float | None = None,
) -> float:
    """Conformal quantile of nonconformity ``scores`` at miscoverage ``alpha``.

    Returns the smallest score whose (weight-)normalised rank reaches ``1 - alpha`` once the test
    point's ``+inf`` mass is included, so ``prediction ± result`` (or the CQR adjustment) is
    marginally valid at level ``1 - alpha`` under (weighted) exchangeability. Returns ``inf`` (a
    valid, infinite interval) when the calibration set is too small for the requested level.
    ``weights`` are calibration likelihood ratios ``dP_test/dP_cal``; ``test_weight`` defaults to
    their mean.
    """
    s = np.asarray(scores, dtype=np.float64)
    n = s.size
    if n == 0:
        raise ValueError("need at least one calibration score")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=np.float64)
    if w.shape != s.shape:
        raise ValueError("weights must match scores in length")
    wt = float(w.mean()) if test_weight is None else float(test_weight)
    order = np.argsort(s)
    s_sorted = s[order]
    w_sorted = w[order]
    total = float(w_sorted.sum()) + wt
    cum = np.cumsum(w_sorted) / total
    idx = int(np.searchsorted(cum, 1.0 - alpha, side="left"))
    return float("inf") if idx >= n else float(s_sorted[idx])


def split_conformal_interval(
    prediction: float,
    cal_y: Sequence[float],
    cal_pred: Sequence[float],
    alpha: float = 0.1,
    *,
    weights: Sequence[float] | None = None,
    test_weight: float | None = None,
) -> Interval:
    """Symmetric split-conformal interval ``prediction ± q`` from absolute calibration residuals."""
    resid = np.abs(np.asarray(cal_y, dtype=np.float64) - np.asarray(cal_pred, dtype=np.float64))
    q = conformal_quantile(resid.tolist(), alpha, weights=weights, test_weight=test_weight)
    return Interval(float(prediction) - q, float(prediction) + q)


def cqr_interval(
    pred_lo: float,
    pred_hi: float,
    cal_y: Sequence[float],
    cal_lo: Sequence[float],
    cal_hi: Sequence[float],
    alpha: float = 0.1,
    *,
    weights: Sequence[float] | None = None,
    test_weight: float | None = None,
) -> Interval:
    """Conformalized quantile regression interval (Romano, Patterson & Candes 2019).

    ``cal_lo``/``cal_hi`` are the calibration lower/upper quantile predictions; the conformity score
    is ``max(cal_lo - y, y - cal_hi)`` and the returned interval widens the test quantile band
    ``[pred_lo, pred_hi]`` by the conformal quantile, giving adaptive finite-sample coverage.
    """
    y = np.asarray(cal_y, dtype=np.float64)
    lo = np.asarray(cal_lo, dtype=np.float64)
    hi = np.asarray(cal_hi, dtype=np.float64)
    scores = np.maximum(lo - y, y - hi)
    q = conformal_quantile(scores.tolist(), alpha, weights=weights, test_weight=test_weight)
    return Interval(float(pred_lo) - q, float(pred_hi) + q)


def certify_conformal_interval(
    prediction: float,
    cal_y: Sequence[float],
    cal_pred: Sequence[float],
    *,
    alpha: float = 0.1,
    weights: Sequence[float] | None = None,
    test_weight: float | None = None,
    query: str = "counterfactual",
) -> Certificate:
    """A ``kind=EMPIRICAL`` certificate for a split-conformal prediction interval.

    ``value`` is left ``None`` (the claim is coverage of an individual outcome, not a point
    estimate); the interval is in ``ci`` and ``alpha`` is the miscoverage. The
    (weighted-)exchangeability requirement is recorded as a non-checkable :class:`Assumption`.
    """
    interval = split_conformal_interval(
        prediction, cal_y, cal_pred, alpha, weights=weights, test_weight=test_weight
    )
    shifted = weights is not None
    return Certificate(
        claim=(
            f"P(Y in [{interval.lower:.4g}, {interval.upper:.4g}]) >= {1.0 - alpha:.2f} (conformal)"
        ),
        estimand=EstimandSpec(query=query, target="interval"),
        kind=Kind.EMPIRICAL,
        value=None,
        alpha=alpha,
        assumptions=(
            Assumption(
                name="weighted-exchangeability" if shifted else "exchangeability",
                params={"n_calibration": int(np.size(cal_y))},
                checkable=False,
            ),
        ),
        method="weighted-split-conformal" if shifted else "split-conformal",
        witness=None,
        hedge=None,
        provenance=Provenance.create(),
        ci=interval,
    )
