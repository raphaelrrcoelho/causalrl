"""Mechanism fitters: (parent columns, child column) -> a fitted structural equation.

Each fitter returns the mechanism, the exogenous distribution its noise is drawn from, whether
that noise is recoverable from (parents, value) — which decides whether counterfactuals at this
node are identified — and a held-out-comparable fit score.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple, Protocol, cast

import numpy as np
import torch
from torch.distributions import Distribution, Normal, Uniform

from causalrl.scm.mechanisms import (
    FunctionalMechanism,
    LinearGaussianMechanism,
    Mechanism,
    NeuralMechanism,
)

Tensor = torch.Tensor
_EMPTY_SIZE = torch.Size()  # type: ignore[reportPrivateImportUsage]  # avoids a B008 call-default

_MAX_CPT_ROWS = 100_000
"""Largest conditional probability table :class:`TabularCPT` will build, in rows.

One row per parent configuration, so the count is the PRODUCT of the parents' cardinalities and
grows explosively with a continuous parent (which contributes one level per distinct observed
value). 100k rows is already far more than any sample can populate: fitting one means most rows
are pure Laplace prior, and the table has stopped being a conditional distribution and become a
nearest-neighbour memoriser of the training rows.
"""


class FittedMechanism(NamedTuple):
    """A fitted structural equation plus what the fit does and does not license."""

    mechanism: Mechanism
    noise: Distribution
    invertible: bool
    score: float


class MechanismFitter(Protocol):
    """Fits ``V = f(parents, noise)`` for one node.

    A fitter reporting ``invertible=True`` should attach a ``residual(parent_values, value) ->
    Tensor`` closure to the returned mechanism (see :func:`_attach_residual`), the way
    :class:`LinearGaussianFit`, :class:`ANMFit`, and :class:`NeuralFit` all do. A fitter reporting
    ``invertible=False`` must instead attach a ``log_prob(parent_values, value) -> Tensor``
    closure, the way :class:`TabularCPT` does: :func:`evaluate_holdout` requires one of the two to
    score a fitted mechanism out-of-sample, and raises a clear error naming which is missing
    rather than letting an opaque ``AttributeError`` escape from inside a user-supplied fitter.
    """

    def fit(self, parents: dict[str, np.ndarray], child: np.ndarray) -> FittedMechanism: ...


class TabularCPT:
    """Discrete node: a Laplace-smoothed conditional probability table, sampled by inverse CDF.

    The mechanism is ``V = F^-1(U | parents)`` with ``U ~ Uniform(0, 1)``. That construction is
    one of many couplings reproducing the same ``P(V | parents)``, and the data cannot
    distinguish them — hence ``invertible=False``, which makes counterfactuals at this node an
    interval rather than a point.

    **Every parent is discretised by its distinct observed values**, and the table holds one row
    per parent configuration — the product of those cardinalities. So this family fits a discrete
    node with *discrete* parents. A continuous parent contributes one level per observed value and
    the product explodes: the commonest real-data shape, a binary treatment with two continuous
    confounders at n=800, already asks for 409,600 rows. ``fit`` therefore raises ``ValueError``
    above :data:`_MAX_CPT_ROWS` rather than exhausting memory, or (where it fits) silently
    becoming a nearest-neighbour memoriser of the training rows. Pass a continuous family for that
    node instead — ``fit_scm(..., families={"A": ANMFit()})``.

    ``alpha`` is the Laplace pseudo-count and must be positive: at ``alpha=0`` an unobserved parent
    configuration would give an all-zero count row and every draw there would silently be the
    smallest level.
    """

    def __init__(self, alpha: float = 1.0) -> None:
        if alpha <= 0.0:
            raise ValueError(
                f"TabularCPT(alpha={alpha}) must be positive: a non-positive pseudo-count leaves "
                "an unobserved parent configuration with an all-zero count row, whose 0/0 "
                "normalisation makes every draw there the smallest level rather than a draw from "
                "any distribution. Use a small alpha (e.g. 1e-6) for near-maximum-likelihood."
            )
        self.alpha = alpha

    def fit(self, parents: dict[str, np.ndarray], child: np.ndarray) -> FittedMechanism:
        values = np.unique(child)
        parent_names = sorted(parents)
        levels = {name: np.unique(parents[name]) for name in parent_names}
        strides: list[int] = []
        size = 1
        for name in parent_names:
            strides.append(size)
            size *= len(levels[name])
        if size > _MAX_CPT_ROWS:
            cardinalities = ", ".join(f"{name}={len(levels[name])}" for name in parent_names)
            raise ValueError(
                f"TabularCPT would need {size} rows -- one per parent configuration, the product "
                f"of the parents' distinct observed values ({cardinalities}) -- above the "
                f"_MAX_CPT_ROWS={_MAX_CPT_ROWS} limit. A parent with that many levels is "
                f"continuous, not tabular, so the table would be a nearest-neighbour memoriser of "
                f"the {len(child)} training rows rather than a conditional distribution. Fit this "
                f"node with a continuous family instead: fit_scm(..., families={{...: ANMFit()}})."
            )

        rows = (
            np.zeros(len(child), dtype=int)
            if not parent_names
            else self._config_index(parents, parent_names, levels, strides, size)
        )
        counts = np.full((size, len(values)), self.alpha, dtype=float)
        col_of = {v: j for j, v in enumerate(values)}
        for row, value in zip(rows, child, strict=True):
            counts[row, col_of[value]] += 1.0
        table = counts / counts.sum(axis=1, keepdims=True)

        # Mean conditional log-likelihood of the training child values under the fitted table.
        columns: list[int] = [col_of[v] for v in child]
        score = float(np.log(table[rows, columns]).mean())

        cum = torch.tensor(np.cumsum(table, axis=1), dtype=torch.float32)  # type: ignore[reportPrivateImportUsage]
        value_tensor = torch.tensor(values, dtype=torch.float32)  # type: ignore[reportPrivateImportUsage]
        table_tensor = torch.tensor(table, dtype=torch.float32)  # type: ignore[reportPrivateImportUsage]
        level_tensors = {
            name: torch.tensor(levels[name], dtype=torch.float32)  # type: ignore[reportPrivateImportUsage]
            for name in parent_names
        }

        def row_index(parent_values: dict[str, Tensor], n: int) -> Tensor:
            row = torch.zeros(n, dtype=torch.long)  # type: ignore[reportPrivateImportUsage]
            for name, stride in zip(parent_names, strides, strict=True):
                column = parent_values[name].reshape(-1).float()
                # Nearest level, so an unseen/off-grid parent value maps to its closest bucket.
                distance = (column.unsqueeze(1) - level_tensors[name].unsqueeze(0)).abs()
                row = row + distance.argmin(dim=1) * stride
            return row

        def mechanism(parent_values: dict[str, Tensor], noise: Tensor) -> Tensor:
            n = noise.reshape(-1).shape[0]
            row = row_index(parent_values, n)
            picked = (noise.reshape(-1).unsqueeze(1) > cum[row]).sum(dim=1)
            return value_tensor[picked.clamp(max=len(values) - 1)]

        def log_prob(parent_values: dict[str, Tensor], value: Tensor) -> Tensor:
            """Per-sample log P(value | parents) under the fitted table -- ``evaluate_holdout``'s
            score for this non-invertible family, since there is no residual to exploit here."""
            flat = value.reshape(-1).float()
            row = row_index(parent_values, flat.shape[0])
            # Nearest fitted value, so an unseen/off-grid child value scores against its closest
            # bucket instead of a lookup error -- degrades gracefully, matching how mechanism()
            # already handles an off-grid parent value.
            distance = (flat.unsqueeze(1) - value_tensor.unsqueeze(0)).abs()
            col = distance.argmin(dim=1)
            return table_tensor[row, col].log()

        fitted_mechanism = FunctionalMechanism(parent_names, mechanism)
        fitted_mechanism.log_prob = log_prob  # type: ignore[attr-defined]
        return FittedMechanism(
            mechanism=fitted_mechanism,
            noise=Uniform(0.0, 1.0),
            invertible=False,
            score=score,
        )

    @staticmethod
    def _config_index(
        parents: dict[str, np.ndarray],
        parent_names: list[str],
        levels: dict[str, np.ndarray],
        strides: list[int],
        size: int,
    ) -> np.ndarray:
        n = len(next(iter(parents.values()))) if parents else 0
        if not parent_names:
            return np.zeros(max(n, 1) if parents else 0, dtype=int)
        rows = np.zeros(n, dtype=int)
        for name, stride in zip(parent_names, strides, strict=True):
            rows += np.searchsorted(levels[name], parents[name]) * stride
        return rows


def _design(parents: dict[str, np.ndarray], names: list[str], n: int) -> np.ndarray:
    """Column-stacked ``[1, parents...]`` design matrix."""
    columns = [np.ones(n)] + [np.asarray(parents[name], dtype=float) for name in names]
    return np.column_stack(columns)


def _r2(y: np.ndarray, predicted: np.ndarray) -> float:
    total = float(np.sum((y - y.mean()) ** 2))
    if total == 0.0:
        return 1.0
    return float(1.0 - np.sum((y - predicted) ** 2) / total)


def _attach_residual(
    mechanism: Mechanism, names: list[str], mean_fn: Callable[[np.ndarray], np.ndarray]
) -> None:
    """Give an additive-noise mechanism its inverse map ``U = V - g(parents)``.

    This is what ``invertible=True`` buys: exact abduction by solving for the noise, instead of
    rejection sampling — which never matches continuous evidence. ``counterfactual_interval`` uses
    it for every invertible node in a query.
    """

    def residual(parent_values: dict[str, Tensor], value: Tensor) -> Tensor:
        flat = value.reshape(-1)
        columns = (
            np.column_stack(
                [parent_values[name].reshape(-1).numpy().astype(float) for name in names]
            )
            if names
            else np.zeros((flat.shape[0], 0))
        )
        mean = torch.tensor(  # type: ignore[reportPrivateImportUsage]
            np.asarray(mean_fn(columns), dtype=float),
            dtype=torch.float32,  # type: ignore[reportPrivateImportUsage]
        )
        return flat - mean

    mechanism.residual = residual  # type: ignore[attr-defined]


def evaluate_holdout(
    fitted: FittedMechanism, parents: dict[str, np.ndarray], child: np.ndarray
) -> float:
    """Score the DEPLOYED (train-fitted) mechanism against data the fit never saw.

    One shared path for all four families, dispatched on ``invertible`` rather than duplicated
    per family. Invertible (additive-noise) mechanisms recover their mean prediction via
    ``mechanism(parents, zeros)`` -- zero noise contributes nothing to an additive coupling -- and
    are scored by :func:`_r2` against the held-out targets, the same metric as their in-sample
    ``score``. The one non-invertible family (:class:`TabularCPT`) has no residual to exploit, so
    it is scored via mean held-out log-likelihood through the ``log_prob`` closure attached at fit
    time, matching what ``TabularCPT.fit`` already reports in-sample. Either way the result sits
    on the same scale as ``FittedMechanism.score``, so the two numbers are comparable.
    """
    names = sorted(parents)
    child_array = np.asarray(child, dtype=float)
    n = len(child_array)
    parent_tensors = {
        name: torch.tensor(  # type: ignore[reportPrivateImportUsage]
            np.asarray(parents[name], dtype=float),
            dtype=torch.float32,  # type: ignore[reportPrivateImportUsage]
        )
        for name in names
    }
    if fitted.invertible:
        zeros = torch.zeros(n)  # type: ignore[reportPrivateImportUsage]
        with torch.no_grad():
            predicted = fitted.mechanism(parent_tensors, zeros)
        return _r2(child_array, predicted.detach().numpy().astype(float))
    if not hasattr(fitted.mechanism, "log_prob"):
        raise AttributeError(
            f"evaluate_holdout: a non-invertible FittedMechanism (invertible=False) must attach "
            f"a log_prob(parent_values, value) -> Tensor closure to its mechanism, the way "
            f"TabularCPT.fit does (see MechanismFitter's docstring) -- "
            f"{type(fitted.mechanism).__name__} has no such attribute, so there is no way to "
            f"score its holdout fit."
        )
    child_tensor = torch.tensor(  # type: ignore[reportPrivateImportUsage]
        child_array,
        dtype=torch.float32,  # type: ignore[reportPrivateImportUsage]
    )
    # log_prob is attached dynamically (TabularCPT.fit), so it is unknown to the Mechanism
    # protocol -- cast rather than annotate, so the attribute access's inferred Unknown is fully
    # replaced instead of merely blended into "Tensor | Unknown" (which still poisons .mean()).
    log_prob = cast(
        "Callable[[dict[str, Tensor], Tensor], Tensor]",
        fitted.mechanism.log_prob,  # type: ignore[attr-defined]
    )
    with torch.no_grad():
        log_probs = log_prob(parent_tensors, child_tensor)
    return float(log_probs.mean())


class LinearGaussianFit:
    """Continuous node: closed-form OLS mean with Gaussian residual noise.

    The noise enters additively and is therefore recoverable from (parents, value), so
    counterfactuals at this node are identified — ``invertible=True``.
    """

    def fit(self, parents: dict[str, np.ndarray], child: np.ndarray) -> FittedMechanism:
        y = np.asarray(child, dtype=float)
        names = sorted(parents)
        design = _design(parents, names, len(y))
        coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
        predicted = design @ coefficients
        residual = y - predicted
        sigma = float(residual.std())
        weights = {name: float(coefficients[i + 1]) for i, name in enumerate(names)}
        mechanism = LinearGaussianMechanism(names, weights, bias=float(coefficients[0]))
        _attach_residual(
            mechanism,
            names,
            lambda columns: np.column_stack([np.ones(len(columns)), columns]) @ coefficients,
        )
        return FittedMechanism(
            mechanism=mechanism,
            noise=Normal(0.0, max(sigma, 1e-6)),
            invertible=True,
            score=_r2(y, predicted),
        )


class ANMFit:
    """Continuous node: additive noise model ``V = g(parents) + U`` for an arbitrary regressor.

    ``estimator`` is a factory returning a fresh object with ``fit(X, y)`` / ``predict(X)`` — the
    same duck-typed contract as ``GFormulaBackdoorAgent(outcome_model=...)``, so scikit-learn stays
    optional. The default is a dependency-free ridge on RBF features. ``U`` is drawn from the
    empirical residual distribution, and is recoverable from (parents, value) — ``invertible=True``.
    """

    def __init__(
        self, estimator: Callable[[], Any] | None = None, *, n_features: int = 32, seed: int = 0
    ) -> None:
        self._estimator = estimator
        self._n_features = n_features
        self._seed = seed

    def fit(self, parents: dict[str, np.ndarray], child: np.ndarray) -> FittedMechanism:
        y = np.asarray(child, dtype=float)
        names = sorted(parents)
        if not names:
            return LinearGaussianFit().fit({}, y)
        raw = np.column_stack([np.asarray(parents[name], dtype=float) for name in names])
        model = (
            self._estimator()
            if self._estimator is not None
            else _RBFRidge(self._n_features, self._seed)
        )
        model.fit(raw, y)
        predicted = np.asarray(model.predict(raw), dtype=float)
        residual = y - predicted
        residual_tensor = torch.tensor(residual, dtype=torch.float32)  # type: ignore[reportPrivateImportUsage]

        def mechanism(parent_values: dict[str, Tensor], noise: Tensor) -> Tensor:
            columns = np.column_stack(
                [parent_values[name].reshape(-1).numpy().astype(float) for name in names]
            )
            mean = torch.tensor(  # type: ignore[reportPrivateImportUsage]
                np.asarray(model.predict(columns), dtype=float),
                dtype=torch.float32,  # type: ignore[reportPrivateImportUsage]
            )
            return mean + noise.reshape(-1)

        fitted_mechanism = FunctionalMechanism(names, mechanism)
        _attach_residual(fitted_mechanism, names, lambda columns: model.predict(columns))
        return FittedMechanism(
            mechanism=fitted_mechanism,
            noise=_Empirical(residual_tensor),
            invertible=True,
            score=_r2(y, predicted),
        )


class _RBFRidge:
    """Dependency-free ridge regression on random RBF features (the ANMFit default)."""

    def __init__(self, n_features: int, seed: int) -> None:
        self._n_features = n_features
        self._seed = seed

    def fit(self, x: np.ndarray, y: np.ndarray) -> _RBFRidge:
        rng = np.random.default_rng(self._seed)
        self._mean, self._scale = x.mean(axis=0), x.std(axis=0) + 1e-9
        standardized = (x - self._mean) / self._scale
        self._centers = rng.normal(size=(self._n_features, x.shape[1]))
        features = self._features(standardized)
        ridge = features.T @ features + 1e-3 * np.eye(features.shape[1])
        self._coefficients = np.linalg.solve(ridge, features.T @ y)
        return self

    def _features(self, standardized: np.ndarray) -> np.ndarray:
        squared = ((standardized[:, None, :] - self._centers[None, :, :]) ** 2).sum(axis=2)
        return np.column_stack([np.ones(len(standardized)), np.exp(-0.5 * squared)])

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self._features((x - self._mean) / self._scale) @ self._coefficients


class _Empirical(Distribution):
    """Resample a fitted residual vector — the ANM's exogenous distribution."""

    has_rsample = False

    def __init__(self, values: Tensor) -> None:
        super().__init__(validate_args=False)
        self.values = values

    def sample(self, sample_shape: torch.Size = _EMPTY_SIZE) -> Tensor:  # type: ignore[override]
        size = torch.Size(sample_shape)  # type: ignore[reportPrivateImportUsage]
        n = int(size.numel())
        index = torch.randint(0, self.values.shape[0], (n,))  # type: ignore[reportPrivateImportUsage]
        return self.values[index]

    @property
    def stddev(self) -> Tensor:
        return self.values.std()


class NeuralFit:
    """Continuous node: ``V = net(parents) + U``, an MLP mean with Gaussian residual noise.

    Fires :class:`causalrl.scm.mechanisms.NeuralMechanism` — the neural causal model primitive.
    The noise head is additive, so the mechanism stays invertible and counterfactuals at this node
    remain identified; a general (non-additive) net would not.
    """

    def __init__(
        self, *, hidden: int = 32, epochs: int = 200, lr: float = 0.01, seed: int = 0
    ) -> None:
        self.hidden = hidden
        self.epochs = epochs
        self.lr = lr
        self.seed = seed

    def fit(self, parents: dict[str, np.ndarray], child: np.ndarray) -> FittedMechanism:
        y = np.asarray(child, dtype=float)
        names = sorted(parents)
        if not names:
            return LinearGaussianFit().fit({}, y)
        x = torch.tensor(  # type: ignore[reportPrivateImportUsage]
            np.column_stack([np.asarray(parents[n], dtype=float) for n in names]),
            dtype=torch.float32,  # type: ignore[reportPrivateImportUsage]
        )
        target = torch.tensor(y, dtype=torch.float32)  # type: ignore[reportPrivateImportUsage]
        # Seed INSIDE a forked RNG: the fit stays deterministic for a given self.seed without
        # reseeding the caller's process-global stream, which would silently make every
        # subsequent torch draw in their program a function of when they happened to fit a
        # mechanism. Same pattern, and same reason, as StructuralCausalModel._sample_exogenous.
        with torch.random.fork_rng():  # type: ignore[reportUnknownMemberType]
            torch.manual_seed(self.seed)  # type: ignore[reportUnknownMemberType]
            net = torch.nn.Sequential(
                torch.nn.Linear(len(names), self.hidden),
                torch.nn.Tanh(),
                torch.nn.Linear(self.hidden, 1),
            )
            optimizer = torch.optim.Adam(net.parameters(), lr=self.lr)
            for _ in range(self.epochs):
                optimizer.zero_grad()
                loss = torch.nn.functional.mse_loss(net(x).squeeze(-1), target)
                loss.backward()  # type: ignore[reportUnknownMemberType]
                optimizer.step()  # type: ignore[reportUnknownMemberType]
        # Fitting is done: freeze the net so every later call (mechanism(...), residual(...),
        # forward sampling through StructuralCausalModel._evaluate) returns a plain tensor
        # instead of silently re-building a live autograd graph through the trained weights.
        for parameter in net.parameters():
            parameter.requires_grad_(False)
        with torch.no_grad():
            predicted = net(x).squeeze(-1)
        residual = target - predicted
        sigma = float(residual.std())

        # NeuralMechanism concatenates [parents..., noise]; wrap so the noise stays additive.
        mechanism = NeuralMechanism(names, _AdditiveHead(net))

        def mean_fn(columns: np.ndarray) -> np.ndarray:
            with torch.no_grad():
                batch = torch.tensor(columns, dtype=torch.float32)  # type: ignore[reportPrivateImportUsage]
                return net(batch).squeeze(-1).numpy()

        _attach_residual(mechanism, names, mean_fn)
        return FittedMechanism(
            mechanism=mechanism,
            noise=Normal(0.0, max(sigma, 1e-6)),
            invertible=True,
            score=_r2(y, predicted.detach().numpy()),
        )


class _AdditiveHead(torch.nn.Module):
    """Split ``NeuralMechanism``'s ``[parents, noise]`` input into ``net(parents) + noise``."""

    def __init__(self, net: torch.nn.Module) -> None:
        super().__init__()
        self.net = net

    def forward(self, columns: Tensor) -> Tensor:
        parents, noise = columns[:, :-1], columns[:, -1:]
        return self.net(parents) + noise
