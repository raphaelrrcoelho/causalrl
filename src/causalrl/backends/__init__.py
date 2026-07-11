"""Array-API dispatch skeleton (plan §4 ``backends/``; invariant I4).

Phase 0 ships a NumPy-only seam: a namespace getter and thin helpers that later phases extend with
JAX (``backends/jax``) and torch (``backends/torch``) dispatch. The existing private torch numerics
seam (:mod:`causalrl._backend`) is left untouched and will be folded in later. Nothing here may
hard-depend on JAX or torch.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["asarray", "get_namespace"]


def get_namespace(*arrays: Any) -> Any:
    """Return the array-API namespace backing ``arrays`` (NumPy by default; ``jax.numpy`` for JAX).

    Dispatch is duck-typed on each array's defining module, so importing :mod:`causalrl.backends`
    never imports JAX — only an actual JAX array triggers the lazy ``import jax.numpy`` (plan §9,
    invariant I4). Callers should treat the result as an Array-API namespace.
    """
    for array in arrays:
        if type(array).__module__.startswith("jax"):  # pragma: no cover - jax lane only
            import jax.numpy as jnp  # pyright: ignore[reportMissingImports]

            return jnp
    return np


def asarray(obj: Any, /, **kwargs: Any) -> np.ndarray[Any, Any]:
    """Convert ``obj`` to the default-backend (NumPy) array type."""
    return np.asarray(obj, **kwargs)
