# pyright: reportMissingImports=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownParameterType=false
"""Vmapped linear-Gaussian SCM sampling and batched interventions on JAX (plan §9).

A linear-Gaussian SCM is the canonical vectorisable structural model: each node is
``V_i = bias_i + Σ_j coeff_ij · V_j + noise_scale_i · ε_i`` evaluated in topological order.
:func:`vmap_sample_linear_gaussian` draws the exogenous noise from a PRNG key and maps the
per-sample structural pass with ``jax.vmap``; :func:`batched_do_linear_gaussian` maps an atomic
intervention over a grid of values (``do``-effects at simulator scale). Both are deterministic in
``seed``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from causalrl.backends.jax import require_jax

FloatArray = NDArray[np.float64]


def _toposort(adjacency: FloatArray) -> list[int]:
    """Topological order of a DAG (``adjacency[i, j] != 0`` => ``j`` is a parent of ``i``)."""
    d = adjacency.shape[0]
    parents = [set(np.nonzero(adjacency[i])[0].tolist()) for i in range(d)]
    order: list[int] = []
    placed: set[int] = set()
    while len(order) < d:
        progressed = False
        for i in range(d):
            if i not in placed and parents[i] <= placed:
                order.append(i)
                placed.add(i)
                progressed = True
        if not progressed:
            raise ValueError("adjacency is cyclic; a linear-Gaussian SCM must be a DAG")
    return order


def _prepare(
    adjacency: FloatArray, coeff: FloatArray, bias: FloatArray, noise_scale: FloatArray
) -> tuple[Any, Any, Any, Any, list[int]]:
    _, jnp = require_jax()
    a = np.asarray(adjacency, dtype=np.float64)
    order = _toposort(a)
    # Zero any coefficient off the declared parent set so the masked matrix is authoritative.
    weight = jnp.asarray(np.asarray(coeff, dtype=np.float64) * (a != 0.0))
    return (
        weight,
        jnp.asarray(np.asarray(bias, dtype=np.float64)),
        jnp.asarray(np.asarray(noise_scale, dtype=np.float64)),
        jnp,
        order,
    )


def vmap_sample_linear_gaussian(
    adjacency: FloatArray,
    coeff: FloatArray,
    bias: FloatArray,
    noise_scale: FloatArray,
    *,
    n: int,
    seed: int,
) -> FloatArray:
    """Draw ``n`` samples from a linear-Gaussian SCM, ``jax.vmap``-ed over the exogenous noise.

    ``adjacency`` is the ``(d, d)`` parent mask (``adjacency[i, j] != 0`` ⇒ ``j`` is a parent of
    ``i``), ``coeff`` the structural coefficients, ``bias`` / ``noise_scale`` the ``(d,)`` per-node
    intercepts and Gaussian noise scales. Returns an ``(n, d)`` NumPy array of samples in node-index
    order; deterministic in ``seed``.
    """
    jax, _ = require_jax()
    weight, b, s, jnp, order = _prepare(adjacency, coeff, bias, noise_scale)
    d = int(b.shape[0])
    eps = jax.random.normal(jax.random.PRNGKey(seed), (n, d))

    def one(eps_row: Any) -> Any:
        v = jnp.zeros(d)
        for i in order:
            v = v.at[i].set(b[i] + jnp.dot(weight[i], v) + s[i] * eps_row[i])
        return v

    return np.asarray(jax.vmap(one)(eps))


def batched_do_linear_gaussian(
    adjacency: FloatArray,
    coeff: FloatArray,
    bias: FloatArray,
    noise_scale: FloatArray,
    *,
    target: int,
    grid: Sequence[float],
    outcome: int,
    n: int,
    seed: int,
) -> FloatArray:
    """Mean of ``outcome`` under ``do(target = g)`` for every ``g`` in ``grid``, ``vmap``-ed twice.

    Clamps node ``target`` to each grid value (an atomic intervention), samples ``n`` draws sharing
    one noise tensor, and returns the ``(len(grid),)`` array of outcome means — the batched-``do``
    over an intervention grid from plan §9. Deterministic in ``seed``.
    """
    jax, _ = require_jax()
    weight, b, s, jnp, order = _prepare(adjacency, coeff, bias, noise_scale)
    d = int(b.shape[0])
    eps = jax.random.normal(jax.random.PRNGKey(seed), (n, d))

    def value_under_do(do_value: Any) -> Any:
        def one(eps_row: Any) -> Any:
            v = jnp.zeros(d)
            for i in order:
                if i == target:
                    v = v.at[i].set(do_value)
                else:
                    v = v.at[i].set(b[i] + jnp.dot(weight[i], v) + s[i] * eps_row[i])
            return v[outcome]

        return jnp.mean(jax.vmap(one)(eps))

    return np.asarray(jax.vmap(value_under_do)(jnp.asarray(list(grid), dtype=jnp.float32)))
