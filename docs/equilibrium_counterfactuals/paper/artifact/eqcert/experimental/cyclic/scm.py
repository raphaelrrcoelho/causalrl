"""Linear cyclic structural causal models (experimental; plan §11).

A linear cyclic SCM is the fixed-point system ``x = B x + u`` over variables ``x`` with an
exogenous ``u ~ N(mean, cov)``; ``B[i, j]`` is the structural coefficient of variable ``j`` in the
equation for variable ``i`` (a directed edge ``j -> i``, cycles and self-loops allowed). Its
*equilibrium* (reduced form) is ``x = (I - B)^{-1} u`` -- **uniquely** defined iff ``I - B`` is
invertible. Outside the uniquely-solvable class :meth:`LinearCyclicSCM.solve` returns a typed hedge;
it never fabricates an arbitrary solution (plan §11, invariant I3).

The *contractive* subclass (spectral radius ``rho(B) < 1``) additionally guarantees that the naive
unrolling ``x_{k+1} = B x_k + u`` converges to that equilibrium -- the property the
equilibrium-vs-unrolling comparator (:mod:`eqcert.experimental.cyclic.comparator`) checks.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from eqcert.certify.certificate import Hedge
from eqcert.exceptions import EqcertError
from eqcert.experimental.cyclic.graph import CyclicCausalGraph

FloatArray = NDArray[np.float64]


class CyclicSolveError(EqcertError):
    """Raised when a numeric solution is requested from a non-uniquely-solvable cyclic SCM."""


@dataclass(frozen=True)
class CyclicSolution:
    """The equilibrium distribution of a linear cyclic SCM, or a typed hedge if none is unique.

    When :attr:`hedge` is ``None`` the equilibrium is the Gaussian ``N(mean, cov)`` over the
    (unique) solution ``x = (I - B)^{-1} u``. When :attr:`hedge` is populated the system is not
    uniquely solvable and :attr:`mean` / :attr:`cov` are ``None`` -- no solution is invented.
    """

    variables: tuple[str, ...]
    mean: FloatArray | None
    cov: FloatArray | None
    unique: bool  # I - B invertible (a unique equilibrium exists)
    contractive: bool  # spectral radius < 1 (unrolling converges to the equilibrium)
    hedge: Hedge | None

    @property
    def solved(self) -> bool:
        """Whether a unique equilibrium was found (no hedge)."""
        return self.hedge is None

    def mean_dict(self) -> dict[str, float]:
        """Equilibrium mean keyed by variable name (raises if hedged)."""
        if self.mean is None:
            reason = self.hedge.reason if self.hedge else ""
            raise CyclicSolveError(f"no unique solution: {reason}")
        return {name: float(value) for name, value in zip(self.variables, self.mean, strict=True)}


class LinearCyclicSCM:
    """A linear cyclic SCM ``x = B x + u`` with Gaussian exogenous noise ``u ~ N(mean, cov)``."""

    def __init__(
        self,
        coefficients: FloatArray | Sequence[Sequence[float]],
        variables: Sequence[str],
        *,
        noise_mean: FloatArray | Sequence[float] | None = None,
        noise_cov: FloatArray | Sequence[Sequence[float]] | None = None,
    ) -> None:
        self.coefficients: FloatArray = np.asarray(coefficients, dtype=np.float64)
        if self.coefficients.ndim != 2 or self.coefficients.shape[0] != self.coefficients.shape[1]:
            raise ValueError("coefficients must be a square matrix")
        self.variables: tuple[str, ...] = tuple(variables)
        dim = int(self.coefficients.shape[0])
        if len(self.variables) != dim:
            raise ValueError(f"variables ({len(self.variables)}) must match dim ({dim})")
        self.noise_mean: FloatArray = (
            np.zeros(dim) if noise_mean is None else np.asarray(noise_mean, dtype=np.float64)
        )
        self.noise_cov: FloatArray = (
            np.eye(dim) if noise_cov is None else np.asarray(noise_cov, dtype=np.float64)
        )
        if self.noise_mean.shape != (dim,):
            raise ValueError(f"noise_mean must have shape ({dim},)")
        if self.noise_cov.shape != (dim, dim):
            raise ValueError(f"noise_cov must have shape ({dim}, {dim})")

    @property
    def dim(self) -> int:
        return len(self.variables)

    def _index(self, name: str) -> int:
        try:
            return self.variables.index(name)
        except ValueError:
            raise KeyError(f"unknown variable: {name!r}") from None

    def spectral_radius(self) -> float:
        """``rho(B)`` -- the largest eigenvalue modulus of the coefficient matrix."""
        if self.dim == 0:
            return 0.0
        return float(np.max(np.abs(np.linalg.eigvals(self.coefficients))))

    def is_contractive(self, tol: float = 1e-9) -> bool:
        """Whether ``rho(B) < 1`` -- the unrolling ``x_{k+1} = B x_k + u`` converges."""
        return self.spectral_radius() < 1.0 - tol

    def is_uniquely_solvable(self, tol: float = 1e-9) -> bool:
        """Whether ``I - B`` is invertible -- a unique equilibrium exists."""
        return not self._is_singular(np.eye(self.dim) - self.coefficients, tol)

    def spectral_abscissa(self) -> float:
        """``max Re(lambda(B - I))`` -- the abscissa of the mean-dynamics Jacobian.

        The associated mean (adaptive-learning) dynamics of ``x = B x + u`` are the ODE
        ``x' = (B - I) x + u``; they are locally stable at the equilibrium iff this is negative.
        """
        if self.dim == 0:
            return -1.0
        return float(np.max(np.linalg.eigvals(self.coefficients).real)) - 1.0

    def stability_margin(self) -> float:
        """``-spectral_abscissa()`` -- positive iff the mean dynamics equilibrate.

        This is strictly weaker than contractivity: ``rho(B) < 1`` implies a positive margin, but a
        positive margin allows ``rho(B) >= 1`` (e.g. ``B = [[-2]]``), where the naive unrolling
        diverges yet damped adaptive dynamics still converge to the equilibrium ``do()``.
        """
        return -self.spectral_abscissa()

    def max_stable_learning_rate(self) -> float:
        """Largest ``gamma`` below which ``x + gamma (B x + u - x)`` converges (0.0 if none).

        Each eigenvalue ``nu`` of ``B - I`` maps to ``1 + gamma nu``; that stays inside the unit
        circle iff ``gamma < -2 Re(nu) / |nu|^2``, so the binding constraint is their minimum. A
        positive :meth:`stability_margin` guarantees this is positive (small-gain convergence).
        """
        if self.dim == 0:
            return 2.0
        if self.stability_margin() <= 0.0:
            return 0.0
        shifted = np.linalg.eigvals(self.coefficients) - 1.0
        return float(np.min(-2.0 * shifted.real / np.abs(shifted) ** 2))

    @staticmethod
    def _is_singular(matrix: FloatArray, tol: float) -> bool:
        if matrix.size == 0:
            return False
        singular_values = np.linalg.svd(matrix, compute_uv=False)
        return bool(singular_values[-1] <= tol * max(1.0, float(singular_values[0])))

    def graph(self) -> CyclicCausalGraph:
        """Induced :class:`CyclicCausalGraph`: edge ``j -> i`` for each nonzero ``B[i, j]``."""
        edges = [
            (self.variables[j], self.variables[i])
            for i in range(self.dim)
            for j in range(self.dim)
            if self.coefficients[i, j] != 0.0
        ]
        return CyclicCausalGraph(edges, nodes=list(self.variables))

    def intervene(self, do: Mapping[str, float]) -> LinearCyclicSCM:
        """Mutilated SCM under ``do``: each intervened variable is pinned, incoming edges cut."""
        coefficients = self.coefficients.copy()
        noise_mean = self.noise_mean.copy()
        noise_cov = self.noise_cov.copy()
        for name, value in do.items():
            i = self._index(name)
            coefficients[i, :] = 0.0  # remove incoming edges: x_i no longer depends on its parents
            noise_mean[i] = float(value)  # x_i = value deterministically
            noise_cov[i, :] = 0.0
            noise_cov[:, i] = 0.0
        return LinearCyclicSCM(
            coefficients, self.variables, noise_mean=noise_mean, noise_cov=noise_cov
        )

    def _effective_noise(
        self, context: Mapping[str, float] | None
    ) -> tuple[FloatArray, FloatArray]:
        mean = self.noise_mean.copy()
        cov = self.noise_cov.copy()
        for name, value in (context or {}).items():
            i = self._index(name)
            mean[i] = float(value)
            cov[i, :] = 0.0
            cov[:, i] = 0.0
        return mean, cov

    def solve(
        self,
        *,
        context: Mapping[str, float] | None = None,
        do: Mapping[str, float] | None = None,
        tol: float = 1e-9,
    ) -> CyclicSolution:
        """Equilibrium distribution under optional exogenous ``context`` and intervention ``do``.

        Returns a :class:`CyclicSolution`; if ``I - B`` is singular the solution is a typed hedge
        (no unique equilibrium) rather than an arbitrary point.
        """
        scm = self.intervene(do) if do else self
        identity_minus_b = np.eye(scm.dim) - scm.coefficients
        if self._is_singular(identity_minus_b, tol):
            hedge = Hedge(
                reason=(
                    "linear cyclic system is not uniquely solvable (I - B is singular): no unique "
                    "equilibrium -- refusing to return an arbitrary solution"
                ),
                detail={
                    "det_I_minus_B": float(np.linalg.det(identity_minus_b)),
                    "spectral_radius": scm.spectral_radius(),
                },
            )
            return CyclicSolution(scm.variables, None, None, False, False, hedge)
        inverse = np.linalg.inv(identity_minus_b)
        mean_u, cov_u = scm._effective_noise(context)
        mean = inverse @ mean_u
        cov = inverse @ cov_u @ inverse.T
        return CyclicSolution(scm.variables, mean, cov, True, scm.is_contractive(tol), None)

    def sample(
        self,
        n: int,
        *,
        context: Mapping[str, float] | None = None,
        do: Mapping[str, float] | None = None,
        seed: int | None = None,
    ) -> FloatArray:
        """Draw ``n`` equilibrium samples ``(n, dim)``; raises if not uniquely solvable."""
        scm = self.intervene(do) if do else self
        identity_minus_b = np.eye(scm.dim) - scm.coefficients
        if self._is_singular(identity_minus_b, 1e-9):
            raise CyclicSolveError(
                "cannot sample a non-uniquely-solvable cyclic SCM (I - B is singular)"
            )
        inverse = np.linalg.inv(identity_minus_b)
        mean_u, cov_u = scm._effective_noise(context)
        rng = np.random.default_rng(seed)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_u)
        root = eigenvectors @ np.diag(np.sqrt(np.clip(eigenvalues, 0.0, None)))
        standard = rng.standard_normal((n, scm.dim))
        noise = mean_u + standard @ root.T
        return noise @ inverse.T
