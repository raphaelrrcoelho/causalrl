"""Adapter: a DoWhy propensity-based CausalEstimate -> PolicyValueContrast.

Duck-typed: the adapter reads the fitted propensity scores off the estimate and never imports
``dowhy`` itself, so it is importable and testable without the optional dependency.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from causalrl.identification.estimate import PolicyValueContrast


def _extract_propensities(estimate: Any) -> list[float]:
    """Read per-unit nominal propensities from a fitted DoWhy propensity-based estimate.

    DoWhy stores the fitted scores on the underlying estimator (``estimate.estimator
    .propensity_scores``); some versions expose them on the estimate itself. Raises a clear error if
    the estimate is not propensity-based. This is the one version-coupled point of the adapter.
    """
    for owner in (getattr(estimate, "estimator", None), estimate):
        scores = getattr(owner, "propensity_scores", None)
        if scores is not None:
            return [float(v) for v in np.asarray(scores, dtype=float).ravel()]
    raise TypeError(
        "from_dowhy_estimate needs a propensity-based DoWhy estimate "
        "(e.g. PropensityScoreWeightingEstimator); no propensity_scores found"
    )


def from_dowhy_estimate(
    estimate: Any,
    *,
    outcomes: Sequence[float],
    treated: Sequence[int],
    confounder_bins: Sequence[int] | None = None,
    mi_cap: float | None = None,
) -> PolicyValueContrast:
    """Build a :class:`~causalrl.PolicyValueContrast` from a fitted DoWhy propensity-based estimate.

    ``outcomes`` / ``treated`` are the logged outcome and binary treatment columns the DoWhy model
    was fit on; the fitted propensity scores become the logging propensities, so both the MSM and
    (when ``confounder_bins`` / ``mi_cap`` given) pivotality layers apply under
    :func:`causalrl.certify_estimate`. Honest scope: the certified sensitivity is on the propensity
    weights, not on DoWhy's point estimator.
    """
    e0 = _extract_propensities(estimate)
    return PolicyValueContrast.from_binary(
        outcomes,
        treated,
        propensities=e0,
        confounder_bins=confounder_bins,
        mi_cap=mi_cap,
    )
