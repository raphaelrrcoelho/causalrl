"""Adapter: an EconML CATE estimator's induced treatment policy -> PolicyValueContrast (MSM-only).

Duck-typed on ``cate_estimator.effect(X)``; never imports econml, so it is importable and testable
without the optional dependency.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from causalrl.identification.estimate import PolicyValueContrast


def policy_from_econml_cate(
    cate_estimator: Any,
    X: Any,
    *,
    outcomes: Sequence[float],
    treated: Sequence[int],
    logging_propensities: Sequence[float],
) -> PolicyValueContrast:
    """Build an MSM :class:`~causalrl.PolicyValueContrast` for an EconML CATE-induced policy.

    The induced policy treats unit ``i`` iff the fitted CATE ``tau_hat(x_i) > 0``; the contrast
    is its off-policy value against the complement policy, under Tan's marginal sensitivity model
    on the supplied ``logging_propensities`` (the logging/behaviour policy's action probabilities).
    **MSM layer only** — the induced policy varies per unit, so it is not a fixed binary arm, and
    the structural (binary-arm) pivotality layer does not apply. Honest scope: the object certified
    is the induced policy's value, not the CATE point estimate itself.
    """
    tau = np.asarray(cate_estimator.effect(X), dtype=float).ravel()
    pi = (tau > 0.0).astype(float)  # induced action per unit
    f = np.asarray(treated).astype(float)
    target_on = np.where(f == pi, 1.0, 0.0)
    target_off = np.where(f == (1.0 - pi), 1.0, 0.0)
    return PolicyValueContrast(
        outcomes=outcomes,
        logging_propensities=list(logging_propensities),
        target_on=target_on.tolist(),
        target_off=target_off.tolist(),
    )
