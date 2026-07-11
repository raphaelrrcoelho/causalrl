# pyright: reportMissingImports=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownParameterType=false
"""Vectorised certificate kernels on JAX mirroring the NumPy core (plan §9; parity acceptance).

These kernels compute the same statistics as :mod:`causalrl.backends.streaming`, so on identical
inputs the JAX and NumPy paths agree within a documented floating-point tolerance — the backend
parity that lets a certificate be reproduced on either backend.
"""

from __future__ import annotations

from collections.abc import Sequence

from causalrl.backends.jax import require_jax


def ipw_value_jax(weights: Sequence[float], rewards: Sequence[float]) -> float:
    """Self-normalised (Hájek) importance-sampling value ``Σ w r / Σ w`` on JAX.

    Mirrors :attr:`causalrl.backends.streaming.WeightedStreamingRatio.value`; agrees with it within
    tolerance on the same inputs.
    """
    _, jnp = require_jax()
    w = jnp.asarray(weights)
    r = jnp.asarray(rewards)
    return float(jnp.sum(w * r) / jnp.sum(w))
