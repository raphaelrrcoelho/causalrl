"""Default nuisance learners for the estimation core (§7.2): pure-numpy so ``estimate/`` has no
hard scipy/sklearn dependency, while remaining sklearn-compatible.

``Regressor`` / ``Classifier`` are minimal duck-typed protocols matching the sklearn estimator API
(``fit`` + ``predict`` / ``predict_proba``). Any sklearn regressor/classifier satisfies them, so a
caller can pass ``outcome_model=lambda: sklearn.linear_model.LinearRegression()`` and the DR/DML
estimators use it unchanged. The defaults below are deliberately simple, well-conditioned learners
(near-OLS ridge; L2-penalised logistic via IRLS) that are correctly specified on linear-Gaussian
mechanisms and keep the core dependency-free.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

__all__ = ["Classifier", "LogisticRegressor", "Regressor", "RidgeRegressor"]

FloatArray = NDArray[np.float64]


@runtime_checkable
class Regressor(Protocol):
    """A fitted-then-predict real-valued learner (the sklearn regressor surface)."""

    def fit(self, x: Any, y: Any) -> Any: ...
    def predict(self, x: Any) -> Any: ...


@runtime_checkable
class Classifier(Protocol):
    """A binary probabilistic classifier (the sklearn ``predict_proba`` surface)."""

    def fit(self, x: Any, y: Any) -> Any: ...
    def predict_proba(self, x: Any) -> Any: ...


def _design(x: Any) -> FloatArray:
    """Return a 2-D design matrix with a leading intercept column (handles 0 covariates)."""
    a = np.asarray(x, dtype=np.float64)
    if a.ndim == 1:
        a = a[:, None]
    n = a.shape[0]
    return np.concatenate([np.ones((n, 1)), a], axis=1)


class RidgeRegressor:
    """Closed-form ridge regression. With the default tiny penalty it is effectively OLS but never
    singular; increase ``alpha`` for ill-conditioned or wide covariate matrices."""

    def __init__(self, alpha: float = 1e-6) -> None:
        self.alpha = float(alpha)
        self.beta: FloatArray = np.empty(0, dtype=np.float64)

    def fit(self, x: Any, y: Any) -> RidgeRegressor:
        xd = _design(x)
        yv = np.asarray(y, dtype=np.float64)
        d = xd.shape[1]
        gram = xd.T @ xd + self.alpha * np.eye(d)
        self.beta = np.asarray(np.linalg.solve(gram, xd.T @ yv), dtype=np.float64)
        return self

    def predict(self, x: Any) -> FloatArray:
        return _design(x) @ self.beta


def _sigmoid(z: FloatArray) -> FloatArray:
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


class LogisticRegressor:
    """L2-penalised logistic regression fitted by IRLS/Newton. ``predict_proba`` returns the
    probability of the positive class as a 1-D array."""

    def __init__(self, l2: float = 1e-6, max_iter: int = 100, tol: float = 1e-8) -> None:
        self.l2 = float(l2)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.beta: FloatArray = np.empty(0, dtype=np.float64)

    def fit(self, x: Any, y: Any) -> LogisticRegressor:
        xd = _design(x)
        yv = np.asarray(y, dtype=np.float64)
        d = xd.shape[1]
        beta = np.zeros(d)
        for _ in range(self.max_iter):
            p = _sigmoid(xd @ beta)
            w = np.clip(p * (1.0 - p), 1e-9, None)
            grad = xd.T @ (yv - p) - self.l2 * beta
            hess = (xd.T * w) @ xd + self.l2 * np.eye(d)
            step = np.linalg.solve(hess, grad)
            beta = beta + step
            if float(np.max(np.abs(step))) < self.tol:
                break
        self.beta = np.asarray(beta, dtype=np.float64)
        return self

    def predict_proba(self, x: Any) -> FloatArray:
        return _sigmoid(_design(x) @ self.beta)
