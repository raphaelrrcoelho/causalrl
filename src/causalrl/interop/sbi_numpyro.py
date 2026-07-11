"""Posterior-over-parameters → :class:`~causalrl.regime.Regime` samplers (plan §10).

Simulation-based inference (SBI) and NumPyro produce a *posterior over environment parameters* — a
cloud of calibrated configurations consistent with observed data. This bridge turns that cloud into
a set of :class:`Regime`s (one per posterior sample), so the transport and partial-identification
layers can quantify a claim **across the calibrated configurations** rather than at a single point
estimate: e.g. the ``[min, max]`` of a transported effect over the posterior (see below).

The core (:func:`regimes_from_posterior`) is pure NumPy and takes plain sample arrays. The
:func:`regimes_from_numpyro` / :func:`regimes_from_sbi_posterior` adapters are fully duck-typed —
neither NumPyro nor ``sbi`` is ever imported — so a real MCMC/posterior object or a lightweight
stand-in drives them unchanged (install them via ``causalrl[numpyro]`` / your own SBI stack).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from causalrl.identification.bounds import Interval
from causalrl.regime import Regime

__all__ = [
    "PosteriorRegimeSampler",
    "across_regimes",
    "regimes_from_numpyro",
    "regimes_from_posterior",
    "regimes_from_sbi_posterior",
]

FloatArray = NDArray[np.float64]
PosteriorSamples = Mapping[str, Any] | tuple[Any, Sequence[str]]


def _as_named_matrix(samples: PosteriorSamples) -> tuple[list[str], FloatArray]:
    """Normalise posterior samples to ``(sorted param names, (n_samples, n_params) matrix)``."""
    if isinstance(samples, Mapping):
        names = sorted(samples.keys())
        if not names:
            return [], np.empty((0, 0), dtype=np.float64)
        cols = [np.asarray(samples[k], dtype=np.float64).reshape(-1) for k in names]
        n = cols[0].shape[0]
        for name, col in zip(names, cols, strict=True):
            if col.shape[0] != n:
                raise ValueError(
                    f"posterior column {name!r} has {col.shape[0]} samples, expected {n}"
                )
        return names, np.column_stack(cols)
    matrix_raw, names_raw = samples
    matrix = np.asarray(matrix_raw, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("posterior sample matrix must be 2-D (n_samples, n_params)")
    names = [str(x) for x in names_raw]
    if matrix.shape[1] != len(names):
        raise ValueError(f"matrix has {matrix.shape[1]} columns but {len(names)} names given")
    return names, matrix


def regimes_from_posterior(
    samples: PosteriorSamples,
    *,
    name_prefix: str = "posterior",
    selection: Iterable[str] = (),
    max_regimes: int | None = None,
    seed: int | None = None,
) -> list[Regime]:
    """Turn posterior samples over environment parameters into one :class:`Regime` per sample.

    ``samples`` is either a mapping ``param_name -> samples`` or a ``(matrix, names)`` pair with
    ``matrix`` of shape ``(n_samples, n_params)``. Each regime carries that sample's values
    and marks ``selection`` as the mechanism variables that differ across the population/config.
    ``max_regimes`` subsamples without replacement (deterministic given ``seed``) to keep a large
    posterior tractable.
    """
    names, matrix = _as_named_matrix(samples)
    n = int(matrix.shape[0])
    idx = np.arange(n)
    if max_regimes is not None and max_regimes < n:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(n, size=max_regimes, replace=False))
    sel = frozenset(selection)
    regimes: list[Regime] = []
    for j in idx.tolist():
        params = {names[k]: float(matrix[j, k]) for k in range(len(names))}
        regimes.append(Regime.create(f"{name_prefix}[{j}]", selection=sel, parameters=params))
    return regimes


class PosteriorRegimeSampler:
    """A reusable sampler over posterior regimes (a calibrated-configuration ensemble).

    Holds the posterior matrix once; :meth:`regimes` materialises all regimes, :meth:`sample` draws
    a bootstrap subset, and :meth:`mean_regime` returns the single posterior-mean configuration.
    """

    def __init__(
        self,
        samples: PosteriorSamples,
        *,
        selection: Iterable[str] = (),
        name_prefix: str = "posterior",
    ) -> None:
        self._names, self._matrix = _as_named_matrix(samples)
        self._selection = frozenset(selection)
        self._name_prefix = name_prefix

    def __len__(self) -> int:
        return int(self._matrix.shape[0])

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._names)

    def regimes(self) -> list[Regime]:
        """All posterior regimes."""
        return regimes_from_posterior(
            (self._matrix, self._names), selection=self._selection, name_prefix=self._name_prefix
        )

    def sample(self, n: int, *, seed: int | None = None) -> list[Regime]:
        """Draw ``n`` regimes uniformly with replacement (a bootstrap over the posterior)."""
        total = len(self)
        if total == 0:
            raise ValueError("empty posterior")
        rng = np.random.default_rng(seed)
        picks = rng.integers(0, total, size=n)
        out: list[Regime] = []
        for j in picks.tolist():
            params = {self._names[k]: float(self._matrix[j, k]) for k in range(len(self._names))}
            out.append(
                Regime.create(
                    f"{self._name_prefix}[{j}]", selection=self._selection, parameters=params
                )
            )
        return out

    def mean_regime(self, *, name: str = "posterior-mean") -> Regime:
        """The single regime at the posterior mean of every parameter."""
        if len(self) == 0:
            raise ValueError("empty posterior")
        means = self._matrix.mean(axis=0)
        params = {self._names[k]: float(means[k]) for k in range(len(self._names))}
        return Regime.create(name, selection=self._selection, parameters=params)


def regimes_from_numpyro(mcmc: Any, **kwargs: Any) -> list[Regime]:
    """Regimes from a NumPyro ``MCMC`` run — reads ``mcmc.get_samples()`` (NumPyro never imported).

    Accepts any object exposing ``get_samples() -> {param_name: array}`` (a NumPyro ``MCMC``, or a
    stand-in). Extra keyword arguments pass through to :func:`regimes_from_posterior`.
    """
    raw: Mapping[str, Any] = mcmc.get_samples()
    posterior = {str(k): np.asarray(v, dtype=np.float64).reshape(-1) for k, v in raw.items()}
    return regimes_from_posterior(posterior, **kwargs)


def regimes_from_sbi_posterior(
    posterior: Any,
    observation: Any,
    *,
    param_names: Sequence[str],
    n: int = 100,
    seed: int | None = None,
    **kwargs: Any,
) -> list[Regime]:
    """Regimes from an SBI posterior via ``posterior.sample((n,), x=observation)`` (``sbi`` unused).

    Accepts any object exposing ``sample((n,), x=...)`` returning an ``(n, n_params)`` array-like
    (an ``sbi`` ``DirectPosterior``, or a stand-in); the draw is coerced with ``np.asarray`` so a
    torch tensor works via its array protocol. ``param_names`` labels the columns.
    """
    draws = np.asarray(posterior.sample((n,), x=np.asarray(observation)), dtype=np.float64)
    if draws.ndim == 1:
        draws = draws.reshape(-1, 1)
    return regimes_from_posterior((draws, list(param_names)), seed=seed, **kwargs)


def across_regimes(regimes: Iterable[Regime], fn: Callable[[Regime], float]) -> Interval:
    """Quantify a functional across calibrated regimes as its ``[min, max]`` :class:`Interval`.

    ``fn`` maps each regime (its parameter configuration) to a scalar — e.g. a transported effect or
    a partial-identification bound evaluated under that configuration. The returned interval is the
    range over the posterior ensemble: a worst-case envelope across calibrated configurations.
    """
    values = [float(fn(r)) for r in regimes]
    if not values:
        raise ValueError("no regimes to evaluate")
    return Interval(min(values), max(values))
