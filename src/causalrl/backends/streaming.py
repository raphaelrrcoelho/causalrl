"""Streaming sufficient statistics for scale-out certificate kernels (plan §9; invariant I4).

Single-pass, mergeable accumulators that consume :meth:`TrajectoryLog.scan` batches without
materialising the whole log. Pure NumPy — the always-on core of the Phase-3 data plane; the optional
JAX backend (:mod:`causalrl.backends.jax`) mirrors these numerics at scale and must agree with them
within a documented tolerance (the determinism acceptance).

* :class:`StreamingMoments` — running count / mean / variance via Chan et al.'s numerically stable
  parallel merge; exact agreement with a one-shot ``mean`` / ``var(ddof=1)``.
* :class:`WeightedStreamingRatio` — self-normalised (Hájek) weighted mean ``Σ w y / Σ w`` with a
  one-pass influence-function standard error and an effective-sample-size overlap diagnostic: the
  streaming backbone of importance-sampling off-policy evaluation.

References: T. F. Chan, G. H. Golub & R. J. LeVeque, *Algorithms for Computing the Sample Variance*
(The American Statistician, 1983). Formula-level implementation; no third-party code is ported.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from causalrl.identification.bounds import Interval

__all__ = ["StreamingMoments", "WeightedStreamingRatio"]

FloatArray = NDArray[np.float64]


class StreamingMoments:
    """Running count / mean / (sample) variance over streamed batches.

    Combines batches with Chan et al.'s parallel merge — carrying ``count``, ``mean`` and ``M2``
    (the running sum of squared deviations) — which is numerically stable and order-independent, so
    ``update``-ing batches or ``merge``-ing partial accumulators yields the same moments (up to
    floating point) as a single pass over the concatenated data.
    """

    def __init__(self) -> None:
        self._n: int = 0
        self._mean: float = 0.0
        self._m2: float = 0.0  # running sum of squared deviations from the mean

    @property
    def count(self) -> int:
        return self._n

    @property
    def mean(self) -> float:
        return self._mean if self._n > 0 else float("nan")

    @property
    def variance(self) -> float:
        """Sample variance (ddof=1); ``nan`` for fewer than two observations."""
        return self._m2 / (self._n - 1) if self._n > 1 else float("nan")

    @property
    def std(self) -> float:
        v = self.variance
        return float("nan") if math.isnan(v) else math.sqrt(v)

    @property
    def se(self) -> float:
        """Standard error of the mean; ``nan`` for fewer than two observations."""
        return self.std / math.sqrt(self._n) if self._n > 1 else float("nan")

    def update(self, values: FloatArray | Sequence[float]) -> StreamingMoments:
        """Absorb a batch of observations; returns ``self`` for chaining."""
        x = np.asarray(values, dtype=np.float64).reshape(-1)
        n_b = int(x.shape[0])
        if n_b == 0:
            return self
        mean_b = float(x.mean())
        m2_b = float(((x - mean_b) ** 2).sum())
        self._absorb(n_b, mean_b, m2_b)
        return self

    def merge(self, other: StreamingMoments) -> StreamingMoments:
        """Fold another accumulator into this one (associative); returns ``self``."""
        if other._n > 0:
            self._absorb(other._n, other._mean, other._m2)
        return self

    def _absorb(self, n_b: int, mean_b: float, m2_b: float) -> None:
        n_a, mean_a, m2_a = self._n, self._mean, self._m2
        n = n_a + n_b
        delta = mean_b - mean_a
        self._n = n
        self._mean = mean_a + delta * n_b / n
        self._m2 = m2_a + m2_b + delta * delta * n_a * n_b / n

    def ci(self, z: float) -> Interval:
        """Wald confidence interval ``mean ± z·se`` (the caller supplies the ``z`` multiplier)."""
        h = z * self.se
        return Interval(self._mean - h, self._mean + h)


class WeightedStreamingRatio:
    """Streaming self-normalised (Hájek) weighted mean ``Σ w y / Σ w`` with a one-pass SE.

    Accumulates the six sufficient statistics ``n, Σw, Σwy, Σw², Σw²y, Σw²y²``, from which the
    self-normalised estimate ``V = Σwy/Σw`` and its standard error follow in a single pass: the
    Hájek influence contribution is ``ψ_i = w_i (y_i - V) / w̄`` (mean zero by construction), so
    ``SE = √(Σ w_i² (y_i - V)²) / Σ w`` — expandable from the stored moments without a second pass.

    With importance weights ``w_i = π_target(a_i|s_i)/π_behavior(a_i|s_i)`` and ``y_i`` the reward,
    ``V`` is the self-normalised IS off-policy value; :meth:`effective_sample_size` is the standard
    ``(Σw)²/Σw²`` overlap diagnostic used to hedge when positivity is too weak (invariant I3).
    """

    def __init__(self) -> None:
        self._n: int = 0
        self._sw: float = 0.0
        self._swy: float = 0.0
        self._sw2: float = 0.0
        self._sw2y: float = 0.0
        self._sw2y2: float = 0.0

    @property
    def count(self) -> int:
        return self._n

    @property
    def sum_weights(self) -> float:
        return self._sw

    @property
    def value(self) -> float:
        return self._swy / self._sw if self._sw != 0.0 else float("nan")

    @property
    def effective_sample_size(self) -> float:
        """Kish effective sample size ``(Σw)² / Σw²`` (0 when no weight has accumulated)."""
        return self._sw * self._sw / self._sw2 if self._sw2 > 0.0 else 0.0

    @property
    def se(self) -> float:
        """Influence-function standard error of :attr:`value`; ``nan`` below two observations."""
        if self._sw == 0.0 or self._n < 2:
            return float("nan")
        v = self.value
        s = self._sw2y2 - 2.0 * v * self._sw2y + v * v * self._sw2
        return math.sqrt(max(s, 0.0)) / self._sw

    def update(
        self, weights: FloatArray | Sequence[float], values: FloatArray | Sequence[float]
    ) -> WeightedStreamingRatio:
        """Absorb a batch of ``(weight, value)`` pairs; returns ``self`` for chaining."""
        w = np.asarray(weights, dtype=np.float64).reshape(-1)
        y = np.asarray(values, dtype=np.float64).reshape(-1)
        if w.shape != y.shape:
            raise ValueError(f"weights {w.shape} and values {y.shape} must have equal length")
        if w.shape[0] == 0:
            return self
        w2 = w * w
        self._n += int(w.shape[0])
        self._sw += float(w.sum())
        self._swy += float((w * y).sum())
        self._sw2 += float(w2.sum())
        self._sw2y += float((w2 * y).sum())
        self._sw2y2 += float((w2 * y * y).sum())
        return self

    def merge(self, other: WeightedStreamingRatio) -> WeightedStreamingRatio:
        """Fold another accumulator into this one (associative); returns ``self``."""
        self._n += other._n
        self._sw += other._sw
        self._swy += other._swy
        self._sw2 += other._sw2
        self._sw2y += other._sw2y
        self._sw2y2 += other._sw2y2
        return self

    def ci(self, z: float) -> Interval:
        """Wald confidence interval ``value ± z·se`` (the caller supplies the ``z`` multiplier)."""
        h = z * self.se
        v = self.value
        return Interval(v - h, v + h)
