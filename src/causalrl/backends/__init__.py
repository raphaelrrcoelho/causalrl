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
    """Return the array-API namespace backing ``arrays``.

    NumPy is the only backend in Phase 0; later phases inspect ``arrays`` to dispatch to
    ``jax.numpy`` or a torch shim. Callers should treat the result as an Array-API namespace.
    """
    return np


def asarray(obj: Any, /, **kwargs: Any) -> np.ndarray[Any, Any]:
    """Convert ``obj`` to the default-backend (NumPy) array type."""
    return np.asarray(obj, **kwargs)
