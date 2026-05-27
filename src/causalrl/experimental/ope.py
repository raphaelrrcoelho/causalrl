"""Experimental off-policy evaluation helpers.

Functions in this module are exploratory utilities, not validated estimators suitable for
methodological claims.
"""


def confounding_sensitivity_bounds(point: float, gamma: float) -> tuple[float, float]:
    """Return a qualitative sensitivity interval around `point`.

    This monotone interval has the expected widening behavior as ``gamma`` grows, but is not
    the published marginal-sensitivity-model bound (Kallus-Zhou / Tan's Gamma).
    """
    if gamma < 1.0:
        raise ValueError("gamma must be >= 1")
    half_width = (gamma - 1.0) / (gamma + 1.0)
    lo = max(0.0, point - half_width)
    hi = min(1.0, point + half_width)
    return lo, hi
