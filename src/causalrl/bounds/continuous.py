"""Continuous partial-identification bounds and heavy-tail / quantile targets (plan §7.3).

Two additions over the shipped nominal-propensity MSM kernels (``identification.bounds``):

* **Estimated-propensity MSM** (``msm_sensitivity_bounds``): the Tan marginal-sensitivity-model
  bound on ``E[Y(1)]`` when logging propensities are *estimated* from covariates rather than known.
  It reduces exactly to :func:`~causalrl.identification.bounds.ipw_sensitivity_bounds` when
  propensities are supplied directly (the ``propensities=`` path), and otherwise fits a propensity
  model and feeds the estimates to the same sharp box kernel.
* **Heavy-tail / quantile targets**: a Hill tail-index diagnostic and finite-variance check
  (``moment_diagnostic``); weighted quantiles with percentile-bootstrap intervals
  (``certify_quantile``); and a mean front door (``certify_mean``) that, on an infinite-variance
  sample, *downgrades* the mean request to a median certificate (I3) instead of reporting an
  unreliable mean.

References: Z. Tan (2006); B. Hill, *A Simple General Approach to Inference About the Tail of a
Distribution* (Ann. Statist. 1975); Dorn & Guo and Dorn, Guo & Kallus for the sharp-MSM / quantile
line. Formula-level implementations; no third-party code is ported.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from causalrl.certify.certificate import (
    Assumption,
    Certificate,
    EstimandSpec,
    Hedge,
    Kind,
    Provenance,
)
from causalrl.estimate._stats import norm_ppf
from causalrl.estimate.nuisance import Classifier, LogisticRegressor
from causalrl.identification.bounds import Interval, ipw_sensitivity_bounds

__all__ = [
    "MomentDiagnostic",
    "certify_mean",
    "certify_quantile",
    "certify_sensitivity_bounds",
    "moment_diagnostic",
    "msm_sensitivity_bounds",
    "tail_index_hill",
    "weighted_quantile",
]


# --------------------------------------------------------------------------- moment diagnostics


@dataclass(frozen=True)
class MomentDiagnostic:
    """Heavy-tail diagnostic: the Hill tail-index estimate and whether the variance is finite."""

    tail_index: float  # Hill estimate of the (right-)tail index alpha; larger = lighter tail
    finite_variance: bool  # alpha > variance_threshold (2 by default)
    tail_fraction: float
    n: int


def tail_index_hill(samples: Any, *, tail_fraction: float = 0.1) -> float:
    """Hill estimator of the tail index ``alpha`` from the largest ``tail_fraction`` magnitudes.

    A Pareto/regularly-varying tail with index ``alpha`` has finite variance iff ``alpha > 2``.
    Returns ``inf`` for a degenerate (all-equal or non-positive) tail.
    """
    x = np.abs(np.asarray(samples, dtype=np.float64))
    x = np.sort(x[x > 0])
    n = x.size
    if n < 3:
        return float("inf")
    k = int(np.floor(tail_fraction * n))
    k = max(2, min(k, n - 1))
    top = np.asarray(x[-(k + 1) :], dtype=np.float64)  # k+1 largest; top[0] is the threshold
    threshold = float(top[0])
    if threshold <= 0.0:
        return float("inf")
    log_ratios = np.asarray(np.log(top[1:]) - np.log(threshold), dtype=np.float64)
    hill = float(log_ratios.mean())
    if hill <= 0.0:
        return float("inf")
    return 1.0 / hill


def moment_diagnostic(
    samples: Any, *, tail_fraction: float = 0.1, variance_threshold: float = 2.0
) -> MomentDiagnostic:
    """Estimate the tail index and decide whether a mean/variance target is trustworthy."""
    x = np.asarray(samples, dtype=np.float64)
    alpha = tail_index_hill(x, tail_fraction=tail_fraction)
    return MomentDiagnostic(
        tail_index=alpha,
        finite_variance=alpha > variance_threshold,
        tail_fraction=tail_fraction,
        n=int(x.size),
    )


# --------------------------------------------------------------------------- quantiles


def weighted_quantile(values: Any, tau: float, weights: Any | None = None) -> float:
    """The ``tau``-quantile of ``values`` (optionally IPW-weighted); unweighted matches numpy."""
    v = np.asarray(values, dtype=np.float64)
    if weights is None:
        return float(np.quantile(v, tau))
    w = np.asarray(weights, dtype=np.float64)
    order = np.argsort(v)
    v, w = v[order], w[order]
    cum = np.cumsum(w) - 0.5 * w
    total = w.sum()
    if total <= 0.0:
        raise ValueError("weights must sum to a positive value")
    return float(np.interp(tau, cum / total, v))


def bootstrap_quantile_ci(
    values: Any,
    tau: float,
    *,
    alpha: float = 0.05,
    n_boot: int = 1000,
    seed: int = 0,
    weights: Any | None = None,
) -> Interval:
    """Percentile-bootstrap CI for a (weighted) quantile; heavy-tail safe (no density est.)."""
    v = np.asarray(values, dtype=np.float64)
    n = v.size
    rng = np.random.default_rng(seed)
    if weights is None:
        idx = rng.integers(0, n, size=(n_boot, n))
        stats = np.quantile(v[idx], tau, axis=1)
    else:
        w = np.asarray(weights, dtype=np.float64)
        stats = np.empty(n_boot)
        for b in range(n_boot):
            j = rng.integers(0, n, size=n)
            stats[b] = weighted_quantile(v[j], tau, w[j])
    return Interval(
        float(np.quantile(stats, alpha / 2)), float(np.quantile(stats, 1.0 - alpha / 2))
    )


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
        return ipw_sensitivity_bounds(
            outcomes, propensities, gamma=gamma, return_certificate=False
        )
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


def certify_quantile(
    outcomes: Sequence[float],
    tau: float = 0.5,
    *,
    alpha: float = 0.05,
    weights: Any | None = None,
    n_boot: int = 1000,
    seed: int = 0,
    kind: Kind = Kind.IDENTIFIED,
) -> Certificate:
    """Certify the ``tau``-quantile of ``outcomes`` with a percentile-bootstrap CI."""
    q = weighted_quantile(outcomes, tau, weights)
    ci = bootstrap_quantile_ci(
        outcomes, tau, alpha=alpha, n_boot=n_boot, seed=seed, weights=weights
    )
    return Certificate(
        claim=f"{tau:g}-quantile = {q:.4g}",
        estimand=EstimandSpec(query="do", target="quantile"),
        kind=kind,
        value=q,
        alpha=alpha,
        assumptions=(),
        method="weighted-quantile-bootstrap",
        witness=None,
        hedge=None,
        provenance=Provenance.create(seeds=(seed,)),
        ci=ci,
    )


def certify_mean(
    outcomes: Sequence[float],
    *,
    alpha: float = 0.05,
    seed: int = 0,
    tail_fraction: float = 0.1,
    n_boot: int = 1000,
    downgrade: bool = True,
) -> Certificate:
    """Certify the mean of ``outcomes``; on an infinite-variance sample, downgrade to the median.

    Runs :func:`moment_diagnostic`; on infinite variance (and ``downgrade`` set) returns a median
    certificate with a :class:`Hedge` (``downgraded_from="mean"``, I3), never a fragile mean point.
    Otherwise a ``kind=IDENTIFIED`` mean certificate with a normal CI and the moment diagnostic.
    """
    diag = moment_diagnostic(outcomes, tail_fraction=tail_fraction)
    moment_assumption = Assumption(
        name="moment-condition",
        params={"target": "finite-variance"},
        checkable=True,
        diagnostic={"tail_index": diag.tail_index, "finite_variance": diag.finite_variance},
    )
    if downgrade and not diag.finite_variance:
        median = certify_quantile(outcomes, 0.5, alpha=alpha, n_boot=n_boot, seed=seed)
        return replace(
            median,
            claim=(
                f"mean ill-defined (tail_index={diag.tail_index:.3g} <= 2); "
                f"reporting median = {median.value:.4g}"
            ),
            assumptions=(moment_assumption,),
            hedge=Hedge(
                reason="heavy-tailed-mean",
                detail={"tail_index": diag.tail_index},
                downgraded_from="mean",
            ),
        )
    y = np.asarray(outcomes, dtype=np.float64)
    m = float(y.mean())
    se = float(y.std(ddof=1) / np.sqrt(y.size))
    z = float(norm_ppf(1.0 - alpha / 2.0))
    return Certificate(
        claim=f"mean = {m:.4g}",
        estimand=EstimandSpec(query="do", target="mean"),
        kind=Kind.IDENTIFIED,
        value=m,
        alpha=alpha,
        assumptions=(moment_assumption,),
        method="sample-mean",
        witness=None,
        hedge=None,
        provenance=Provenance.create(seeds=(seed,)),
        ci=Interval(m - z * se, m + z * se),
    )
