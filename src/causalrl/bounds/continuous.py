"""Continuous partial-identification bounds: estimated-propensity MSM (plan §7.3).

An addition over the shipped nominal-propensity MSM kernels (``identification.bounds``):

* **Estimated-propensity MSM** (``msm_sensitivity_bounds``): the Tan marginal-sensitivity-model
  bound on ``E[Y(1)]`` when logging propensities are *estimated* from covariates rather than known.
  It reduces exactly to :func:`~causalrl.identification.bounds.ipw_sensitivity_bounds` when
  propensities are supplied directly (the ``propensities=`` path), and otherwise fits a propensity
  model and feeds the estimates to the same sharp box kernel.

Reference: Z. Tan (2006). Formula-level implementation; no third-party code is ported.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from causalrl.certify.certificate import (
    Assumption,
    Certificate,
    EstimandSpec,
    Kind,
    Provenance,
)
from causalrl.estimate.nuisance import Classifier, LogisticRegressor
from causalrl.identification.bounds import Interval, ipw_sensitivity_bounds

__all__ = [
    "certify_sensitivity_bounds",
    "msm_sensitivity_bounds",
]


# --------------------------------------------------------------- estimated-propensity MSM


def msm_sensitivity_bounds(
    outcomes: Sequence[float],
    *,
    gamma: float,
    propensities: Sequence[float] | None = None,
    treatment: Sequence[float] | None = None,
    covariates: Any | None = None,
    propensity_model: Classifier | None = None,
) -> Interval:
    """MSM bounds on ``E[Y(1)]`` with *estimated* logging propensities (Dorn-Guo/Kallus line).

    Two modes (note ``outcomes`` differs between them):

    * ``propensities=`` given -> treated units' outcomes + nominal propensities; delegates to
      ``ipw_sensitivity_bounds`` unchanged (the *known*-propensity case it reduces to exactly).
    * ``treatment=`` + ``covariates=`` given -> ``outcomes``/``treatment``/``covariates`` are the
      *full per-logged-unit* arrays; fits ``e_hat = P(T=1 | covariates)`` with ``propensity_model``
      (default logistic) and applies the same sharp box kernel to the treated subset.
    """
    if propensities is not None:
        return ipw_sensitivity_bounds(outcomes, propensities, gamma=gamma, return_certificate=False)
    if treatment is None or covariates is None:
        raise ValueError("provide either propensities=, or both treatment= and covariates=")
    y = np.asarray(outcomes, dtype=np.float64)
    t = np.asarray(treatment, dtype=np.float64)
    z = np.asarray(covariates, dtype=np.float64)
    model: Classifier = propensity_model if propensity_model is not None else LogisticRegressor()
    proba = np.asarray(model.fit(z, t).predict_proba(z), dtype=np.float64)
    e_hat = proba[:, 1] if proba.ndim == 2 else proba
    treated = t == 1
    return ipw_sensitivity_bounds(
        y[treated].tolist(), e_hat[treated].tolist(), gamma=gamma, return_certificate=False
    )


# --------------------------------------------------------------------------- certificates


def certify_sensitivity_bounds(
    outcomes: Sequence[float],
    *,
    gamma: float,
    propensities: Sequence[float] | None = None,
    treatment: Sequence[float] | None = None,
    covariates: Any | None = None,
    propensity_model: Classifier | None = None,
) -> Certificate:
    """A ``kind=BOUNDED`` certificate wrapping :func:`msm_sensitivity_bounds`."""
    interval = msm_sensitivity_bounds(
        outcomes,
        gamma=gamma,
        propensities=propensities,
        treatment=treatment,
        covariates=covariates,
        propensity_model=propensity_model,
    )
    known = propensities is not None
    return Certificate(
        claim=f"E[Y(1)] in [{interval.lower:.4g}, {interval.upper:.4g}] under MSM(Gamma={gamma})",
        estimand=EstimandSpec(query="do", target="mean"),
        kind=Kind.BOUNDED,
        value=interval,
        alpha=None,
        assumptions=(
            Assumption(name="MSM", params={"gamma": gamma}, checkable=False),
            Assumption(
                name="known-propensity" if known else "estimated-propensity",
                params={},
                checkable=True,
            ),
        ),
        method="msm-sensitivity" if known else "msm-sensitivity-estimated",
        witness=None,
        hedge=None,
        provenance=Provenance.create(),
    )
