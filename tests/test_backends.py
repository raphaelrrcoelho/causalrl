"""Phase 0: array-API dispatch skeleton (I4). NumPy-only in this phase."""

import numpy as np

from causalrl.backends import asarray, get_namespace


def test_get_namespace_returns_numpy_for_numpy_arrays() -> None:
    assert get_namespace(np.array([1.0, 2.0])) is np


def test_get_namespace_default_is_numpy() -> None:
    assert get_namespace() is np


def test_asarray_returns_ndarray() -> None:
    out = asarray([1, 2, 3])
    assert isinstance(out, np.ndarray)
    assert out.tolist() == [1, 2, 3]
