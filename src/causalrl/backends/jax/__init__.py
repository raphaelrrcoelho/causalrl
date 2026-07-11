# pyright: reportMissingImports=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownParameterType=false
"""JAX scale backend (plan §9; invariant I4). Optional ``[jax]`` extra; py3.11 CI lane only.

Vectorised (``vmap``) SCM sampling, batched interventions over a value grid, and vectorised
certificate kernels that mirror the NumPy core (:mod:`causalrl.backends.streaming`) within a
documented tolerance — the backend-parity / determinism acceptance. JAX is imported lazily so
nothing here can destabilise the NumPy-only default matrix; JAX/jaxlib ship no py3.14 wheels, hence
the ``python_version < '3.14'`` marker on the extra and the dedicated CI lane.

Determinism rests on JAX's explicit PRNG-key discipline: a call with a given integer ``seed`` builds
``jax.random.PRNGKey(seed)`` and is bit-for-bit reproducible.
"""

from __future__ import annotations

from typing import Any

_MISSING = "the JAX scale backend requires jax; install the 'causalrl[jax]' extra (Python < 3.14)"


def require_jax() -> tuple[Any, Any]:
    """Lazily import and return ``(jax, jax.numpy)``; raise a helpful error if absent."""
    try:
        import jax
        import jax.numpy as jnp
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(_MISSING) from exc
    return jax, jnp


def available() -> bool:
    """``True`` iff JAX is importable (the ``[jax]`` extra is installed)."""
    import importlib.util

    return importlib.util.find_spec("jax") is not None


# Re-exports come after the helpers above so the submodules can import ``require_jax`` from this
# partially-initialised package without a circular-import failure.
from causalrl.backends.jax.kernels import ipw_value_jax  # noqa: E402
from causalrl.backends.jax.sample import (  # noqa: E402
    batched_do_linear_gaussian,
    vmap_sample_linear_gaussian,
)

__all__ = [
    "available",
    "batched_do_linear_gaussian",
    "ipw_value_jax",
    "vmap_sample_linear_gaussian",
]
