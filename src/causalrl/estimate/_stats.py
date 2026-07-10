"""Pure-numpy statistics helpers for the estimation core (§7.2), so the core needs no scipy.

Only what the doubly-robust estimators require: the inverse standard-normal CDF for
confidence-interval critical values.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = ["norm_ppf"]

# Acklam's rational approximation to the inverse standard-normal CDF (relative error < 1.15e-9
# over the whole open interval). Peter J. Acklam, "An algorithm for computing the inverse normal
# cumulative distribution function" (2003). Standard published constants; no code is ported.
_A = (
    -3.969683028665376e01,
    2.209460984245205e02,
    -2.759285104469687e02,
    1.383577518672690e02,
    -3.066479806614716e01,
    2.506628277459239e00,
)
_B = (
    -5.447609879822406e01,
    1.615858368580409e02,
    -1.556989798598866e02,
    6.680131188771972e01,
    -1.328068155288572e01,
)
_C = (
    -7.784894002430293e-03,
    -3.223964580411365e-01,
    -2.400758277161838e00,
    -2.549732539343734e00,
    4.374664141464968e00,
    2.938163982698783e00,
)
_D = (
    7.784695709041462e-03,
    3.224671290700398e-01,
    2.445134137142996e00,
    3.754408661907416e00,
)


def norm_ppf(p: float | NDArray[np.float64]) -> float | NDArray[np.float64]:
    """Inverse standard-normal CDF ``Phi^{-1}(p)`` for ``p`` in the open interval ``(0, 1)``.

    Accepts a scalar or array; returns the same shape. Used for two-sided confidence-interval
    critical values (e.g. ``norm_ppf(1 - alpha/2)``).
    """
    arr = np.asarray(p, dtype=float)
    scalar = arr.ndim == 0
    x = np.atleast_1d(arr)
    if np.any((x <= 0.0) | (x >= 1.0)):
        raise ValueError("norm_ppf requires 0 < p < 1")
    out = np.empty_like(x)
    plow, phigh = 0.02425, 1.0 - 0.02425

    lo = x < plow
    hi = x > phigh
    mid = ~(lo | hi)

    q = x[mid] - 0.5
    r = q * q
    num = (((((_A[0] * r + _A[1]) * r + _A[2]) * r + _A[3]) * r + _A[4]) * r + _A[5]) * q
    den = ((((_B[0] * r + _B[1]) * r + _B[2]) * r + _B[3]) * r + _B[4]) * r + 1.0
    out[mid] = num / den

    ql = np.sqrt(-2.0 * np.log(x[lo]))
    out[lo] = (((((_C[0] * ql + _C[1]) * ql + _C[2]) * ql + _C[3]) * ql + _C[4]) * ql + _C[5]) / (
        (((_D[0] * ql + _D[1]) * ql + _D[2]) * ql + _D[3]) * ql + 1.0
    )

    qh = np.sqrt(-2.0 * np.log(1.0 - x[hi]))
    out[hi] = -(((((_C[0] * qh + _C[1]) * qh + _C[2]) * qh + _C[3]) * qh + _C[4]) * qh + _C[5]) / (
        (((_D[0] * qh + _D[1]) * qh + _D[2]) * qh + _D[3]) * qh + 1.0
    )

    return float(out[0]) if scalar else out
