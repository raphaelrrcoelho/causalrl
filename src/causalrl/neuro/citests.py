"""Conditional-independence tests for continuous and point-process (spike-train) data.

The shipped :mod:`causalrl.discovery` engine tests independence with thresholded discrete
conditional mutual information. Electrophysiology needs two more families, and this module supplies
them behind the :class:`~causalrl.discovery.CITest` protocol so PC and FCI run unchanged:

* :class:`PartialCorrelationTest` — Fisher-z on the partial correlation. Linear, exact under
  joint Gaussianity, and the right default for mesoscopic signals (LFP, population rates).
* :class:`KnnCMITest` - the Frenzel-Pompe/Kraskov k-nearest-neighbour conditional-mutual-
  information estimator. Nonparametric, catches nonlinear dependence, no binning.
* :class:`PoissonGLMTest` — a likelihood-ratio test between nested point-process GLMs. This is the
  spike-train instrument: it respects the point-process likelihood instead of pretending binned
  counts are Gaussian, and it is the conditional-independence form of the GLM connectivity analysis
  standard in the field (Truccolo et al. 2005; Kim, Putrino, Ghosh & Brown, *PLoS Comput. Biol.*
  2011; Pillow et al. 2008).

All three return a :class:`CITestResult` carrying the statistic, a p-value where one is defined,
and the independence verdict at level ``alpha``. Every test is pure NumPy — the project has no
SciPy dependency, so the required special functions (digamma, regularised incomplete gamma, the
normal tail) are implemented here to the accuracy the tests need.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from causalrl.exceptions import CausalRLError

__all__ = [
    "CITestResult",
    "KnnCMITest",
    "PartialCorrelationTest",
    "PoissonGLMTest",
    "chi2_sf",
    "digamma",
    "normal_sf",
]

FloatArray = NDArray[np.float64]


class CITestError(CausalRLError):
    """A conditional-independence test was given unusable data (degenerate or mismatched)."""


@dataclass(frozen=True)
class CITestResult:
    """Outcome of one conditional-independence test.

    ``p_value`` is ``None`` for tests without a calibrated null (a bare estimator compared against
    a threshold); ``independent`` is then the threshold verdict.
    """

    statistic: float
    p_value: float | None
    independent: bool
    method: str
    n_effective: int

    def __bool__(self) -> bool:
        return self.independent


# --------------------------------------------------------------------------------------------
# Special functions (no SciPy dependency).
# --------------------------------------------------------------------------------------------


def digamma(x: FloatArray | float) -> FloatArray:
    """Digamma ``psi(x)`` for ``x > 0``, vectorised.

    Recurrence up to ``x >= 6`` followed by the standard asymptotic expansion; accurate to ~1e-10
    over the range the kNN estimators use (positive integers and small reals).
    """
    v = np.asarray(x, dtype=np.float64).copy()
    if np.any(v <= 0.0):
        raise CITestError("digamma requires strictly positive arguments")
    out = np.zeros_like(v)
    small = v < 6.0
    while np.any(small):
        out[small] -= 1.0 / v[small]
        v[small] += 1.0
        small = v < 6.0
    inv = 1.0 / v
    inv2 = inv * inv
    out += (
        np.log(v)
        - 0.5 * inv
        - inv2 * (1.0 / 12.0 - inv2 * (1.0 / 120.0 - inv2 / 252.0))
    )
    return out


def normal_sf(z: float) -> float:
    """Upper tail ``P(Z > z)`` of the standard normal."""
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def _gamma_p_series(a: float, x: float) -> float:
    """Regularised lower incomplete gamma ``P(a, x)`` by series (converges fast for x < a + 1)."""
    term = 1.0 / a
    total = term
    n = a
    for _ in range(1000):
        n += 1.0
        term *= x / n
        total += term
        if abs(term) < abs(total) * 1e-15:
            break
    return total * math.exp(-x + a * math.log(x) - math.lgamma(a))


def _gamma_q_cf(a: float, x: float) -> float:
    """Regularised upper incomplete gamma ``Q(a, x)`` by continued fraction (for x >= a + 1)."""
    tiny = 1e-300
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, 1000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return h * math.exp(-x + a * math.log(x) - math.lgamma(a))


def chi2_sf(x: float, df: int) -> float:
    """Upper tail ``P(chi2_df > x)``."""
    if df <= 0:
        raise CITestError("chi-square degrees of freedom must be positive")
    if x <= 0.0:
        return 1.0
    a, y = 0.5 * df, 0.5 * x
    if y < a + 1.0:
        return max(0.0, min(1.0, 1.0 - _gamma_p_series(a, y)))
    return max(0.0, min(1.0, _gamma_q_cf(a, y)))


# --------------------------------------------------------------------------------------------
# Helpers.
# --------------------------------------------------------------------------------------------


def _stack(data: Mapping[str, FloatArray], names: Sequence[str]) -> FloatArray:
    if not names:
        return np.zeros((_n_rows(data), 0), dtype=np.float64)
    cols: list[FloatArray] = []
    for name in names:
        if name not in data:
            raise CITestError(f"variable not in data: {name!r}")
        cols.append(np.asarray(data[name], dtype=np.float64).reshape(-1))
    n = cols[0].shape[0]
    if any(c.shape[0] != n for c in cols):
        raise CITestError("all columns must have the same length")
    return np.column_stack(cols).astype(np.float64)


def _n_rows(data: Mapping[str, FloatArray]) -> int:
    for v in data.values():
        return int(np.asarray(v).reshape(-1).shape[0])
    raise CITestError("empty data mapping")


def _residualise(target: FloatArray, covariates: FloatArray) -> FloatArray:
    """Residuals of ``target`` after least-squares regression on ``[1, covariates]``."""
    n = target.shape[0]
    design = np.column_stack([np.ones(n), covariates]) if covariates.size else np.ones((n, 1))
    coef, *_ = np.linalg.lstsq(design, target, rcond=None)
    return target - design @ coef


# --------------------------------------------------------------------------------------------
# Tests.
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class PartialCorrelationTest:
    """Fisher-z test on the partial correlation ``rho(X, Y | Z)``.

    Exact under joint Gaussianity and linear conditional means; the default choice for mesoscopic
    signals. The statistic is ``z = sqrt(n - |Z| - 3) * atanh(rho)``, standard normal under the
    null of conditional independence.
    """

    alpha: float = 0.01

    def partial_correlation(
        self, data: Mapping[str, FloatArray], x: str, y: str, z: Sequence[str]
    ) -> float:
        xs = _stack(data, [x])[:, 0]
        ys = _stack(data, [y])[:, 0]
        zs = _stack(data, list(z))
        rx, ry = _residualise(xs, zs), _residualise(ys, zs)
        denom = float(np.linalg.norm(rx) * np.linalg.norm(ry))
        if denom < 1e-12:
            return 0.0
        return float(np.clip(np.dot(rx, ry) / denom, -0.999999, 0.999999))

    def __call__(
        self, data: Mapping[str, FloatArray], x: str, y: str, z: Sequence[str]
    ) -> CITestResult:
        n = _n_rows(data)
        rho = self.partial_correlation(data, x, y, z)
        dof = n - len(list(z)) - 3
        if dof <= 0:
            return CITestResult(abs(rho), None, True, "partial-correlation", n)
        stat = math.sqrt(dof) * math.atanh(rho)
        p = 2.0 * normal_sf(abs(stat))
        return CITestResult(float(stat), float(p), p > self.alpha, "partial-correlation", n)


@dataclass(frozen=True)
class KnnCMITest:
    """Frenzel-Pompe k-NN estimator of ``I(X; Y | Z)`` with a permutation-calibrated p-value.

    Nonparametric and binning-free. The conditional estimator (Frenzel & Pompe, *Phys. Rev. Lett.*
    2007) extends Kraskov-Stoegbauer-Grassberger (2004) to a conditioning set:

        I(X;Y|Z) = psi(k) - < psi(n_xz + 1) + psi(n_yz + 1) - psi(n_z + 1) >

    with counts taken inside the max-norm ball of the k-th joint neighbour. Cost is quadratic in
    the sample, so ``max_samples`` subsamples first; ``permutations`` shuffles X within nearest-
    neighbour blocks of Z to calibrate a p-value, and is skipped (threshold verdict only) when set
    to zero.
    """

    k: int = 5
    alpha: float = 0.01
    threshold: float = 0.0
    max_samples: int = 2000
    permutations: int = 0
    seed: int = 0

    def estimate(
        self, data: Mapping[str, FloatArray], x: str, y: str, z: Sequence[str]
    ) -> tuple[float, int]:
        rng = np.random.default_rng(self.seed)
        xs, ys, zs = self._prepare(data, x, y, z, rng)
        return self._cmi(xs, ys, zs), xs.shape[0]

    def _prepare(
        self,
        data: Mapping[str, FloatArray],
        x: str,
        y: str,
        z: Sequence[str],
        rng: np.random.Generator,
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        xs, ys, zs = _stack(data, [x]), _stack(data, [y]), _stack(data, list(z))
        n = xs.shape[0]
        if n > self.max_samples:
            idx = rng.choice(n, size=self.max_samples, replace=False)
            xs, ys, zs = xs[idx], ys[idx], zs[idx]
        # Standardise, then dither: binned spike counts are heavily tied, and the k-NN estimator
        # needs distinct distances. Noise at 1e-6 of the spread cannot create dependence.
        out: list[FloatArray] = []
        for arr in (xs, ys, zs):
            if arr.size == 0:
                out.append(arr)
                continue
            sd = arr.std(axis=0, keepdims=True)
            sd = np.where(sd < 1e-12, 1.0, sd)
            scaled = (arr - arr.mean(axis=0, keepdims=True)) / sd
            dithered: FloatArray = np.asarray(
                scaled + 1e-6 * rng.standard_normal(scaled.shape), dtype=np.float64
            )
            out.append(dithered)
        return out[0], out[1], out[2]

    @staticmethod
    def _maxnorm(a: FloatArray) -> FloatArray:
        if a.shape[1] == 0:
            return np.zeros((a.shape[0], a.shape[0]), dtype=np.float64)
        diff = np.abs(a[:, np.newaxis, :] - a[np.newaxis, :, :])
        return diff.max(axis=2)

    def _cmi(self, xs: FloatArray, ys: FloatArray, zs: FloatArray) -> float:
        return self._cmi_from_distances(
            self._maxnorm(xs), self._maxnorm(ys), self._maxnorm(zs), n_cond=zs.shape[1]
        )

    def _cmi_from_distances(
        self, dx: FloatArray, dy: FloatArray, dz: FloatArray, *, n_cond: int
    ) -> float:
        """The estimator proper, taking precomputed max-norm distance matrices.

        Split out so a permutation loop recomputes only the shuffled variable's distances; the
        conditioning set's are the expensive part and never change.
        """
        k = min(self.k, dx.shape[0] - 1)
        if k < 1:
            return 0.0
        joint = np.maximum(np.maximum(dx, dy), dz)
        np.fill_diagonal(joint, np.inf)
        eps = np.sort(joint, axis=1)[:, k - 1]
        strict = eps[:, np.newaxis]
        d_xz, d_yz = np.maximum(dx, dz), np.maximum(dy, dz)
        np.fill_diagonal(d_xz, np.inf)
        np.fill_diagonal(d_yz, np.inf)
        dz_off = dz.copy()
        np.fill_diagonal(dz_off, np.inf)
        n_xz = (d_xz < strict).sum(axis=1)
        n_yz = (d_yz < strict).sum(axis=1)
        n_z = (dz_off < strict).sum(axis=1)
        if n_cond == 0:
            # No conditioning set: the KSG mutual-information estimator.
            value = (
                digamma(float(k))
                + digamma(float(dx.shape[0]))
                - float(np.mean(digamma((n_xz + 1).astype(np.float64))
                                + digamma((n_yz + 1).astype(np.float64))))
            )
        else:
            value = digamma(float(k)) - float(
                np.mean(
                    digamma((n_xz + 1).astype(np.float64))
                    + digamma((n_yz + 1).astype(np.float64))
                    - digamma((n_z + 1).astype(np.float64))
                )
            )
        return max(float(value), 0.0)

    def __call__(
        self, data: Mapping[str, FloatArray], x: str, y: str, z: Sequence[str]
    ) -> CITestResult:
        rng = np.random.default_rng(self.seed)
        xs, ys, zs = self._prepare(data, x, y, z, rng)
        dx, dy, dz = self._maxnorm(xs), self._maxnorm(ys), self._maxnorm(zs)
        n_cond = zs.shape[1]
        stat = self._cmi_from_distances(dx, dy, dz, n_cond=n_cond)
        n = xs.shape[0]
        if self.permutations > 0 and self.alpha < 1.0 / (self.permutations + 1.0):
            raise CITestError(
                f"alpha={self.alpha} is unreachable with {self.permutations} permutations "
                f"(the smallest attainable p-value is {1.0 / (self.permutations + 1.0):.4g}); "
                f"raise permutations to at least {math.ceil(1.0 / self.alpha) - 1}"
            )
        if self.permutations <= 0:
            return CITestResult(stat, None, stat < self.threshold, "knn-cmi", n)
        null = np.empty(self.permutations, dtype=np.float64)
        for b in range(self.permutations):
            perm = rng.permutation(n)
            null[b] = self._cmi_from_distances(
                dx[np.ix_(perm, perm)], dy, dz, n_cond=n_cond
            )
        p = float((1.0 + np.sum(null >= stat)) / (self.permutations + 1.0))
        return CITestResult(stat, p, p > self.alpha, "knn-cmi-permutation", n)


@dataclass(frozen=True)
class PoissonGLMTest:
    """Likelihood-ratio conditional-independence test between nested point-process GLMs.

    Fits ``counts_Y ~ Poisson(exp(b0 + B'Z))`` against ``~ Poisson(exp(b0 + B'Z + a'X))`` by IRLS
    and refers ``2 * (loglik_full - loglik_reduced)`` to a chi-square with ``dim(X)`` degrees of
    freedom. This is the conditional-independence form of point-process GLM connectivity analysis:
    it uses the actual spike-count likelihood, so it stays calibrated at the low counts per bin
    that binned spike trains produce, where a Gaussian partial correlation is badly misspecified.

    ``ridge`` adds a small quadratic penalty for numerical stability under collinear regressors
    (routine at short bin widths); it is reported in the result's method string when non-zero.
    """

    alpha: float = 0.01
    ridge: float = 1e-6
    max_iter: int = 50
    tol: float = 1e-9

    def _fit(self, y: FloatArray, design: FloatArray) -> float:
        """IRLS for a Poisson log-link GLM; returns the maximised log-likelihood."""
        p = design.shape[1]
        beta = np.zeros(p, dtype=np.float64)
        beta[0] = math.log(max(float(y.mean()), 1e-6))
        penalty = self.ridge * np.eye(p)
        for _ in range(self.max_iter):
            eta = np.clip(design @ beta, -30.0, 30.0)
            mu = np.exp(eta)
            w = np.maximum(mu, 1e-10)
            zed = eta + (y - mu) / w
            wd = design * w[:, np.newaxis]
            hessian = design.T @ wd + penalty
            grad = wd.T @ zed
            try:
                nxt = np.linalg.solve(hessian, grad)
            except np.linalg.LinAlgError:
                nxt = np.linalg.lstsq(hessian, grad, rcond=None)[0]
            if float(np.max(np.abs(nxt - beta))) < self.tol:
                beta = nxt
                break
            beta = nxt
        eta = np.clip(design @ beta, -30.0, 30.0)
        mu = np.exp(eta)
        # Poisson log-likelihood up to the y! term, which cancels in the ratio.
        return float(np.sum(y * eta - mu))

    def __call__(
        self, data: Mapping[str, FloatArray], x: str, y: str, z: Sequence[str]
    ) -> CITestResult:
        ys = _stack(data, [y])[:, 0]
        if np.any(ys < 0):
            raise CITestError(f"PoissonGLMTest requires non-negative counts for {y!r}")
        xs = _stack(data, [x])
        zs = _stack(data, list(z))
        n = ys.shape[0]
        ones = np.ones((n, 1), dtype=np.float64)
        reduced = np.column_stack([ones, zs]) if zs.size else ones
        full = np.column_stack([reduced, xs])
        if float(ys.sum()) == 0.0:
            return CITestResult(0.0, 1.0, True, "poisson-glm-lrt", n)
        ll_reduced = self._fit(ys, reduced)
        ll_full = self._fit(ys, full)
        stat = max(2.0 * (ll_full - ll_reduced), 0.0)
        df = xs.shape[1]
        p = chi2_sf(stat, df)
        method = "poisson-glm-lrt" if self.ridge == 0.0 else f"poisson-glm-lrt(ridge={self.ridge})"
        return CITestResult(float(stat), float(p), p > self.alpha, method, n)
