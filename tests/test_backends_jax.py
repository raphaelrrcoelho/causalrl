"""Phase 3 §9: JAX scale backend — parity with the numpy core + PRNG determinism.

Runs only where JAX is installed (the dedicated py3.11 ``jax`` CI lane); JAX/jaxlib have no py3.14
wheels, so the module is skipped on the default matrix and coverage-omitted there.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")

from causalrl.backends import get_namespace
from causalrl.backends.jax import (
    available,
    batched_do_linear_gaussian,
    ipw_value_jax,
    vmap_sample_linear_gaussian,
)
from causalrl.backends.streaming import WeightedStreamingRatio

# Chain X(0) -> Y(1): Y = 2X + eps_y, X = eps_x. adjacency[1, 0] = 1 (0 is a parent of 1).
_ADJ = np.array([[0.0, 0.0], [1.0, 0.0]])
_COEFF = np.array([[0.0, 0.0], [2.0, 0.0]])
_BIAS = np.zeros(2)
_SCALE = np.ones(2)


def test_available_true_in_lane() -> None:
    assert available() is True


def test_vmap_sample_recovers_linear_gaussian_moments() -> None:
    s = vmap_sample_linear_gaussian(_ADJ, _COEFF, _BIAS, _SCALE, n=200_000, seed=0)
    assert s.shape == (200_000, 2)
    x, y = s[:, 0], s[:, 1]
    assert abs(float(x.mean())) < 0.02
    assert abs(float(y.var()) - 5.0) < 0.1  # Var(2X + eps) = 4 + 1
    assert abs(float(np.cov(x, y)[0, 1]) - 2.0) < 0.05  # Cov(X, 2X + eps) = 2


def test_vmap_sample_is_deterministic_in_seed() -> None:
    a = vmap_sample_linear_gaussian(_ADJ, _COEFF, _BIAS, _SCALE, n=5_000, seed=7)
    b = vmap_sample_linear_gaussian(_ADJ, _COEFF, _BIAS, _SCALE, n=5_000, seed=7)
    assert np.array_equal(a, b)  # identical seed -> identical draws (PRNG-key discipline)


def test_batched_do_matches_analytic_effect() -> None:
    grid = [-2.0, -1.0, 0.0, 1.0, 2.0]
    vals = batched_do_linear_gaussian(
        _ADJ, _COEFF, _BIAS, _SCALE, target=0, grid=grid, outcome=1, n=100_000, seed=0
    )
    # do(X = g) => E[Y | do(X=g)] = 2g exactly (co-parents unaffected).
    assert np.allclose(vals, 2.0 * np.asarray(grid), atol=0.05)


def test_ipw_value_jax_parity_with_numpy() -> None:
    rng = np.random.default_rng(0)
    w = rng.uniform(0.2, 3.0, size=10_000)
    r = rng.standard_normal(10_000)
    jax_value = ipw_value_jax(w.tolist(), r.tolist())
    numpy_value = WeightedStreamingRatio().update(w, r).value
    assert abs(jax_value - numpy_value) < 1e-4  # backend parity (float32 JAX vs float64 numpy)


def test_get_namespace_dispatches_to_jax() -> None:
    import jax.numpy as jnp

    assert get_namespace(jnp.asarray([1.0, 2.0])) is jnp
    assert get_namespace(np.asarray([1.0, 2.0])) is np  # numpy still dispatches to numpy
