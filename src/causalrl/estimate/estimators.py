"""Estimators for an identified back-door ATE ``E[Y | do(X=1)] - E[Y | do(X=0)]`` (§7.2).

Four estimators over a binary treatment ``X`` adjusting for an admissible set ``Z``:

* ``plugin``  — g-computation / outcome regression (T-learner mean contrast).
* ``ipw``     — self-normalised (Hajek) inverse-propensity weighting.
* ``aipw``    — one-step doubly-robust / augmented IPW (efficient-influence-function estimate).
* ``dml``     — cross-fitted AIPW (Chernozhukov et al., DML) removing nuisance overfit bias.

``aipw`` and ``dml`` report influence-function standard errors and asymptotically-normal confidence
intervals. Nuisances are pluggable sklearn-style learners (factories); the pure-numpy defaults in
:mod:`causalrl.estimate.nuisance` keep the core dependency-free. Estimators are formula-level
implementations of the cited estimators; no third-party code is ported.

References: J. Robins, A. Rotnitzky & L. Zhao (1994); H. Bang & J. Robins (2005) for AIPW;
V. Chernozhukov et al., *Double/Debiased Machine Learning* (Econometrics Journal, 2018) for the
cross-fitting.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from causalrl.estimate._stats import norm_ppf
from causalrl.estimate.nuisance import (
    Classifier,
    LogisticRegressor,
    Regressor,
    RidgeRegressor,
)
from causalrl.identification.bounds import Interval

__all__ = ["EffectEstimate", "estimate_ate"]

FloatArray = NDArray[np.float64]
OutcomeFactory = Callable[[], Regressor]
PropensityFactory = Callable[[], Classifier]
_METHODS = ("plugin", "ipw", "aipw", "dml")


@dataclass(frozen=True)
class EffectEstimate:
    """A point estimate of an ATE with an (asymptotic) confidence interval and overlap diagnostics.

    ``overlap`` records ``min_propensity`` / ``max_propensity`` of the estimated propensity score
    (``nan`` for the propensity-free ``plugin`` estimator); a certifier downgrades to a hedge when
    positivity is destroyed (I3). ``std_error``/``ci`` are influence-function based for ``aipw`` and
    ``dml`` (the estimators with a valid asymptotic distribution); for ``plugin`` they are a coarse
    outcome-spread approximation and for ``ipw`` a self-normalised-weight approximation.
    """

    value: float
    std_error: float
    ci: Interval
    alpha: float
    n: int
    method: str
    overlap: dict[str, float]
    n_folds: int | None = None


def _default_outcome() -> Regressor:
    return RidgeRegressor()


def _default_propensity() -> Classifier:
    return LogisticRegressor()


def _prob1(model: Classifier, z: FloatArray) -> FloatArray:
    """Positive-class probability as a 1-D array (accepts sklearn's ``(n, 2)`` or a 1-D vector)."""
    p = np.asarray(model.predict_proba(z), dtype=np.float64)
    return p[:, 1] if p.ndim == 2 else p


def _predict(model: Regressor, z: FloatArray) -> FloatArray:
    return np.asarray(model.predict(z), dtype=np.float64)


def _ci(value: float, se: float, alpha: float) -> Interval:
    z = float(norm_ppf(1.0 - alpha / 2.0))
    return Interval(value - z * se, value + z * se)


def _overlap(e_raw: FloatArray) -> dict[str, float]:
    return {"min_propensity": float(e_raw.min()), "max_propensity": float(e_raw.max())}


def _fit_arms(
    outcome: OutcomeFactory,
    z_tr: FloatArray,
    y_tr: FloatArray,
    x_tr: FloatArray,
    z_eval: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """Fit the treated/control outcome models on a training split, predict on ``z_eval``.

    Falls back to the training arm mean when a split is missing a treatment arm (rare; guards
    small folds / near-deterministic treatment)."""
    treated = x_tr == 1
    control = x_tr == 0
    if bool(treated.any()):
        mu1 = _predict(outcome().fit(z_tr[treated], y_tr[treated]), z_eval)
    else:
        mu1 = np.full(z_eval.shape[0], float(y_tr.mean()))
    if bool(control.any()):
        mu0 = _predict(outcome().fit(z_tr[control], y_tr[control]), z_eval)
    else:
        mu0 = np.full(z_eval.shape[0], float(y_tr.mean()))
    return mu1, mu0


def _plugin(
    x: FloatArray, z: FloatArray, y: FloatArray, *, outcome: OutcomeFactory
) -> tuple[float, float, dict[str, float]]:
    mu1, mu0 = _fit_arms(outcome, z, y, x, z)
    d = mu1 - mu0
    value = float(d.mean())
    se = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("nan")
    return value, se, {"min_propensity": float("nan"), "max_propensity": float("nan")}


def _ipw(
    x: FloatArray,
    z: FloatArray,
    y: FloatArray,
    *,
    propensity: PropensityFactory,
    clip: float,
) -> tuple[float, float, dict[str, float]]:
    e_raw = _prob1(propensity().fit(z, x), z)
    e = np.clip(e_raw, clip, 1.0 - clip)
    w1 = x / e
    w0 = (1.0 - x) / (1.0 - e)
    v1 = float((w1 * y).sum() / w1.sum())
    v0 = float((w0 * y).sum() / w0.sum())
    value = v1 - v0
    # Self-normalised (Hajek) influence function for the difference of the two weighted means.
    psi = w1 * (y - v1) / w1.mean() - w0 * (y - v0) / w0.mean()
    se = float(psi.std(ddof=1) / np.sqrt(len(y)))
    return value, se, _overlap(e_raw)


def _aipw_from_nuisances(
    x: FloatArray, y: FloatArray, mu1: FloatArray, mu0: FloatArray, e: FloatArray
) -> tuple[float, float, FloatArray]:
    psi = mu1 - mu0 + x * (y - mu1) / e - (1.0 - x) * (y - mu0) / (1.0 - e)
    value = float(psi.mean())
    se = float(psi.std(ddof=1) / np.sqrt(len(y)))
    return value, se, psi


def _aipw(
    x: FloatArray,
    z: FloatArray,
    y: FloatArray,
    *,
    outcome: OutcomeFactory,
    propensity: PropensityFactory,
    clip: float,
) -> tuple[float, float, dict[str, float]]:
    e_raw = _prob1(propensity().fit(z, x), z)
    e = np.clip(e_raw, clip, 1.0 - clip)
    mu1, mu0 = _fit_arms(outcome, z, y, x, z)
    value, se, _ = _aipw_from_nuisances(x, y, mu1, mu0, e)
    return value, se, _overlap(e_raw)


def _dml(
    x: FloatArray,
    z: FloatArray,
    y: FloatArray,
    *,
    outcome: OutcomeFactory,
    propensity: PropensityFactory,
    clip: float,
    n_folds: int,
    seed: int,
) -> tuple[float, float, dict[str, float]]:
    n = len(y)
    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    folds = np.array_split(idx, n_folds)
    mu1 = np.empty(n)
    mu0 = np.empty(n)
    e_raw = np.empty(n)
    for te in folds:
        tr = np.setdiff1d(idx, te, assume_unique=True)
        m1, m0 = _fit_arms(outcome, z[tr], y[tr], x[tr], z[te])
        mu1[te] = m1
        mu0[te] = m0
        e_raw[te] = _prob1(propensity().fit(z[tr], x[tr]), z[te])
    e = np.clip(e_raw, clip, 1.0 - clip)
    value, se, _ = _aipw_from_nuisances(x, y, mu1, mu0, e)
    return value, se, _overlap(e_raw)


def _assemble_z(data: Mapping[str, Any], adjustment: tuple[str, ...], n: int) -> FloatArray:
    if not adjustment:
        return np.empty((n, 0))
    cols: list[FloatArray] = []
    for name in adjustment:
        if name not in data:
            raise ValueError(f"adjustment variable {name!r} not found in data")
        cols.append(np.asarray(data[name], dtype=np.float64).reshape(n, -1))
    return np.concatenate(cols, axis=1)


def estimate_ate(
    data: Mapping[str, Any],
    treatment: str,
    outcome: str,
    adjustment: tuple[str, ...],
    *,
    method: str = "dml",
    alpha: float = 0.05,
    n_folds: int = 5,
    seed: int = 0,
    outcome_model: OutcomeFactory | None = None,
    propensity_model: PropensityFactory | None = None,
    clip: float = 1e-3,
) -> EffectEstimate:
    """Estimate the back-door ATE of binary ``treatment`` on ``outcome`` given ``adjustment``.

    ``data`` maps variable name -> 1-D array (covariates may be name -> ``(n,)`` or ``(n, k)``).
    ``method`` selects the estimator (``plugin``/``ipw``/``aipw``/``dml``). ``outcome_model`` and
    ``propensity_model`` are zero-argument factories returning fresh sklearn-style learners
    (defaults: ridge outcome, logistic propensity). Raises ``ValueError`` for a non-binary treatment
    or a missing adjustment variable.
    """
    if method not in _METHODS:
        raise ValueError(f"method must be one of {_METHODS}, got {method!r}")
    x = np.asarray(data[treatment], dtype=np.float64)
    y = np.asarray(data[outcome], dtype=np.float64)
    uniq = set(np.unique(x).tolist())
    if not uniq <= {0.0, 1.0}:
        raise ValueError(f"treatment {treatment!r} must be binary 0/1; got values {sorted(uniq)}")
    n = len(x)
    z = _assemble_z(data, adjustment, n)
    of = outcome_model or _default_outcome
    pf = propensity_model or _default_propensity

    n_folds_out: int | None = None
    if method == "plugin":
        value, se, ov = _plugin(x, z, y, outcome=of)
    elif method == "ipw":
        value, se, ov = _ipw(x, z, y, propensity=pf, clip=clip)
    elif method == "aipw":
        value, se, ov = _aipw(x, z, y, outcome=of, propensity=pf, clip=clip)
    else:  # dml
        value, se, ov = _dml(
            x, z, y, outcome=of, propensity=pf, clip=clip, n_folds=n_folds, seed=seed
        )
        n_folds_out = n_folds

    return EffectEstimate(
        value=value,
        std_error=se,
        ci=_ci(value, se, alpha),
        alpha=alpha,
        n=n,
        method=method,
        overlap=ov,
        n_folds=n_folds_out,
    )
