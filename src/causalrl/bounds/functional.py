"""Manski bounds as *functions of state features*, rather than per discrete cell.

:func:`causalrl.causal_q_bounds` bounds ``E[R | do(a), s]`` for a discrete ``s`` from the empirical
propensity and mean in that cell. In feature space there are no cells, so the same two quantities
become fitted functions::

    lower(x, a) = mu(x, a) * e(x, a) + r_min * (1 - e(x, a))
    upper(x, a) = mu(x, a) * e(x, a) + r_max * (1 - e(x, a))

with ``mu(x, a) = E[R | X = x, A = a]`` an outcome regression and ``e(x, a) = P(A = a | X = x)`` a
propensity model. The logic is identical to the tabular bound — the logged fraction contributes its
observed mean, the unlogged fraction is bounded only by the reward range — and with indicator
features and saturated models the two coincide, which is the property :mod:`causalrl.state` set up
and the tests here check.

**What the generalisation costs.** The tabular bound needs no model: the cell mean and the cell
propensity are just counts, so the only error is sampling error. Here both are *estimated
functions*, and the bound inherits their misspecification. A ``mu`` that is too smooth or an ``e``
that is over-confident produces an interval that is too narrow — the anti-conservative direction,
which is the dangerous one for a bound. Two things follow, both of which this module does rather
than merely warns about:

* Nuisances are **cross-fitted** (as the DML estimators here already are), so a training row's
  bound never uses a model that saw that row. Plug-in bounds from in-sample fits are optimistically
  tight, and the effect grows with model flexibility.
* An **overlap diagnostic** is reported. Where ``e(x, a)`` is near zero the bound is near-vacuous
  and honest; where a flexible propensity model has driven it near one, the interval collapses and
  is trustworthy only if that model is right.

Correct specification of ``mu`` and ``e`` is therefore an *assumption*, and one the tabular path
does not need. Consumers should record it — :class:`causalrl.agents.bounded_fitted
.BoundedFittedQIteration` attaches it to every certificate it issues.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from causalrl.estimate.nuisance import (
    Classifier,
    FloatArray,
    LogisticRegressor,
    Regressor,
    RidgeRegressor,
)

__all__ = ["FunctionalManskiBounds", "OverlapDiagnostic"]

_VACUOUS_PROPENSITY = 0.05
"""Estimated propensity below which a cell's bound is reported as effectively vacuous.

Not a threshold the bound itself uses -- the formula is continuous in ``e`` and needs no cutoff.
It exists so :class:`OverlapDiagnostic` can report *how much* of the data sits where the interval
is nearly the full reward range, which is the regime where a bound is honest but uninformative.
"""


@dataclass(frozen=True)
class OverlapDiagnostic:
    """How much support the fitted propensity actually found, per action.

    ``min_propensity`` and ``mean_propensity`` are taken over the estimated ``e(x, a)`` at the
    training rows for the action actually taken. ``vacuous_fraction`` is the share of ``(row,
    action)`` pairs whose estimated propensity falls below :data:`_VACUOUS_PROPENSITY`, i.e. where
    the interval is close to the whole reward range.

    A high ``vacuous_fraction`` is not a failure — it is the bound correctly reporting that the logs
    do not pin the effect down. The case to be suspicious of is the opposite: a very high
    ``mean_propensity`` from a flexible model, which yields tight intervals that are only as
    trustworthy as the propensity fit.
    """

    min_propensity: float
    mean_propensity: float
    vacuous_fraction: float

    def summary(self) -> str:
        return (
            f"overlap: min={self.min_propensity:.3f} mean={self.mean_propensity:.3f} "
            f"vacuous={self.vacuous_fraction:.1%}"
        )


class FunctionalManskiBounds:
    """Cross-fitted pointwise Manski bounds on ``E[R | do(a), X = x]``.

    ``outcome_model`` and ``propensity_model`` are factories returning a fresh
    :class:`~causalrl.estimate.nuisance.Regressor` / :class:`~causalrl.estimate.nuisance.Classifier`
    — the same duck-typed protocols the DML estimators use, so any scikit-learn estimator works and
    the defaults keep the core dependency-free. The propensity is fitted one-vs-rest and normalised
    across actions.

    ``reward_range`` must genuinely contain the reward; it is what the unlogged fraction is bounded
    by, and the bound is only valid if it holds. It defaults to ``(0, 1)``, matching
    :func:`causalrl.causal_q_bounds`.

    Cross-fitting produces two different predictors, deliberately kept apart:
    :attr:`in_sample` are out-of-fold values for the training rows (each from a model that never saw
    that row), and :meth:`bounds` evaluates new points by averaging the fold models. Use the former
    to summarise the data you fitted on and the latter to bound anywhere else.
    """

    def __init__(
        self,
        n_actions: int,
        *,
        outcome_model: Callable[[], Regressor] | None = None,
        propensity_model: Callable[[], Classifier] | None = None,
        reward_range: tuple[float, float] = (0.0, 1.0),
        n_folds: int = 5,
        seed: int = 0,
    ) -> None:
        if n_actions < 1:
            raise ValueError(f"n_actions={n_actions} must be at least 1")
        if n_folds < 2:
            raise ValueError(
                f"n_folds={n_folds} must be at least 2: cross-fitting needs a held-out part, and "
                "a single fold would fit and predict on the same rows -- the in-sample plug-in "
                "this class exists to avoid, since it makes the interval optimistically tight."
            )
        low, high = float(reward_range[0]), float(reward_range[1])
        if not low < high:
            raise ValueError(f"reward_range={reward_range} must satisfy low < high")
        self.n_actions = n_actions
        self.reward_range = (low, high)
        self.n_folds = n_folds
        self.seed = seed
        self._outcome = outcome_model if outcome_model is not None else RidgeRegressor
        self._propensity = propensity_model if propensity_model is not None else LogisticRegressor
        self._fold_models: list[tuple[dict[int, Regressor], dict[int, Classifier]]] = []
        self._in_sample: tuple[FloatArray, FloatArray] | None = None
        self._diagnostic: OverlapDiagnostic | None = None

    def fit(
        self, features: FloatArray, actions: NDArray[np.int_], rewards: FloatArray
    ) -> FunctionalManskiBounds:
        """Cross-fit the outcome and propensity models and record the out-of-fold bounds."""
        x = np.asarray(features, dtype=np.float64)
        a = np.asarray(actions, dtype=int).reshape(-1)
        r = np.asarray(rewards, dtype=np.float64).reshape(-1)
        if x.ndim != 2:
            raise ValueError(f"features must be a 2-D (n, dim) design matrix; got shape {x.shape}")
        if not (len(x) == len(a) == len(r)):
            raise ValueError(
                f"features, actions and rewards must agree in length; got {len(x)}, {len(a)}, "
                f"{len(r)}"
            )
        if len(x) < self.n_folds:
            raise ValueError(
                f"{len(x)} row(s) cannot be split into n_folds={self.n_folds}: every fold must "
                "hold at least one row for the out-of-fold bound to be defined."
            )
        if a.min() < 0 or a.max() >= self.n_actions:
            raise ValueError(
                f"actions must lie in [0, {self.n_actions}); observed [{a.min()}, {a.max()}]"
            )

        n = len(x)
        index = np.arange(n)
        np.random.default_rng(self.seed).shuffle(index)
        folds = np.array_split(index, self.n_folds)

        self._fold_models = []
        mu_oof = np.zeros((n, self.n_actions))
        e_oof = np.zeros((n, self.n_actions))
        for test in folds:
            train = np.setdiff1d(index, test, assume_unique=True)
            outcome, propensity = self._fit_fold(x[train], a[train], r[train])
            self._fold_models.append((outcome, propensity))
            mu_oof[test], e_oof[test] = self._predict(x[test], [(outcome, propensity)])

        low, high = self.reward_range
        logged = mu_oof * e_oof
        unlogged = 1.0 - e_oof
        self._in_sample = (logged + low * unlogged, logged + high * unlogged)
        taken = e_oof[np.arange(n), a]
        self._diagnostic = OverlapDiagnostic(
            min_propensity=float(taken.min()),
            mean_propensity=float(taken.mean()),
            vacuous_fraction=float(np.mean(e_oof < _VACUOUS_PROPENSITY)),
        )
        return self

    def _fit_fold(
        self, x: FloatArray, a: FloatArray, r: FloatArray
    ) -> tuple[dict[int, Regressor], dict[int, Classifier]]:
        outcome: dict[int, Regressor] = {}
        propensity: dict[int, Classifier] = {}
        for action in range(self.n_actions):
            rows = a == action
            if not rows.any():
                continue  # never logged in this fold: e stays 0, so the bound stays vacuous
            outcome[action] = self._outcome().fit(x[rows], r[rows])
            if rows.all():
                continue  # only action present: normalisation below gives it probability 1
            propensity[action] = self._propensity().fit(x, rows.astype(np.float64))
        return outcome, propensity

    def _predict(
        self,
        x: FloatArray,
        models: list[tuple[dict[int, Regressor], dict[int, Classifier]]],
    ) -> tuple[FloatArray, FloatArray]:
        """Average ``mu`` and (one-vs-rest, normalised) ``e`` over the supplied fold models."""
        mu = np.zeros((len(x), self.n_actions))
        e = np.zeros((len(x), self.n_actions))
        for outcome, propensity in models:
            raw = np.zeros((len(x), self.n_actions))
            for action in range(self.n_actions):
                if action in outcome:
                    mu[:, action] += np.asarray(outcome[action].predict(x)).reshape(-1)
                if action in propensity:
                    raw[:, action] = np.clip(
                        np.asarray(propensity[action].predict_proba(x)).reshape(-1), 0.0, 1.0
                    )
                elif action in outcome:
                    raw[:, action] = 1.0  # sole logged action in this fold
            total = raw.sum(axis=1, keepdims=True)
            e += np.divide(raw, total, out=np.zeros_like(raw), where=total > 0.0)
        return mu / len(models), e / len(models)

    @property
    def in_sample(self) -> tuple[FloatArray, FloatArray]:
        """Out-of-fold ``(lower, upper)`` for the training rows, each ``(n, n_actions)``."""
        if self._in_sample is None:
            raise RuntimeError("FunctionalManskiBounds has not been fitted yet; call fit() first")
        return self._in_sample

    def bounds(self, features: FloatArray) -> tuple[FloatArray, FloatArray]:
        """``(lower, upper)`` at arbitrary feature rows, from the averaged fold models.

        Each is ``(n_rows, n_actions)``. Evaluating the training rows through this path is
        *not* the same as :attr:`in_sample`: the average includes the fold model that saw them, so
        the interval is optimistically tight there.
        """
        if not self._fold_models:
            raise RuntimeError("FunctionalManskiBounds has not been fitted yet; call fit() first")
        x = np.asarray(features, dtype=np.float64)
        if x.ndim != 2:
            raise ValueError(f"features must be a 2-D (n, dim) design matrix; got shape {x.shape}")
        mu, e = self._predict(x, self._fold_models)
        low, high = self.reward_range
        return mu * e + low * (1.0 - e), mu * e + high * (1.0 - e)

    def diagnostic(self) -> OverlapDiagnostic:
        """Overlap summary from the out-of-fold propensities — see :class:`OverlapDiagnostic`."""
        if self._diagnostic is None:
            raise RuntimeError("FunctionalManskiBounds has not been fitted yet; call fit() first")
        return self._diagnostic
