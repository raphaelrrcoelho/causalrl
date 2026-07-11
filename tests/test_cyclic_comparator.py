"""Equilibrium-vs-unrolling comparator (plan §11: contraction agreement, empirical/hedge otherwise).

The certificate logic is pure NumPy and tested directly. The end-to-end path reuses the shipped
torch-backed ``build_unrolled_scm``; those tests are skipped where the optional backend is absent
(they run on the main CI matrix, which installs torch) and assert it matches the NumPy unrolling.
"""

from __future__ import annotations

import numpy as np
import pytest

from causalrl.certify.certificate import Certificate, Kind
from causalrl.experimental.cyclic import LinearCyclicSCM, compare_equilibrium_unrolling
from causalrl.experimental.cyclic.comparator import _convergence_certificate


def _torch_backend_available() -> bool:
    try:
        import torch.nn  # noqa: F401
    except Exception:
        return False
    return True


_TORCH_OK = _torch_backend_available()


def _feedback_scm() -> LinearCyclicSCM:
    return LinearCyclicSCM([[0.0, 0.5], [0.5, 0.0]], ["x0", "x1"], noise_mean=[1.0, 2.0])


def _numpy_unrolled_mean(
    scm: LinearCyclicSCM, do: dict[str, float] | None, horizon: int
) -> np.ndarray:
    intervened = scm.intervene(do) if do else scm
    x = np.zeros(intervened.dim)
    for _ in range(horizon):
        x = intervened.coefficients @ x + intervened.noise_mean
    return x


def test_contractive_and_converged_is_identified() -> None:
    scm = _feedback_scm()
    eq = scm.solve().mean
    unrolled = _numpy_unrolled_mean(scm, None, 400)
    cert = _convergence_certificate(scm, None, eq, unrolled, 400, 1e-6, 0)
    assert cert.kind is Kind.IDENTIFIED
    assert cert.hedge is None
    assert cert.value is not None and cert.value < 1e-6


def test_contractive_but_not_converged_hedges_empirical() -> None:
    scm = _feedback_scm()
    eq = scm.solve().mean
    assert eq is not None
    cert = _convergence_certificate(scm, None, eq, eq + 1.0, horizon=3, tol=1e-6, seed=0)
    assert cert.kind is Kind.EMPIRICAL
    assert cert.hedge is not None and "not converged" in cert.hedge.reason


def test_non_contractive_is_empirical_even_when_the_gap_is_zero() -> None:
    # x0 = 1.5 x0 + u0 : a unique equilibrium exists (I - B invertible) but unrolling diverges.
    scm = LinearCyclicSCM([[1.5, 0.0], [0.0, 0.0]], ["x0", "x1"], noise_mean=[1.0, 0.0])
    assert scm.is_uniquely_solvable()
    assert not scm.is_contractive()
    eq = scm.solve().mean
    cert = _convergence_certificate(scm, None, eq, eq, horizon=100, tol=1e-6, seed=0)
    assert cert.kind is Kind.EMPIRICAL
    assert cert.hedge is not None and "non-contractive" in cert.hedge.reason


def test_no_unique_equilibrium_hedges_without_touching_torch() -> None:
    # solve() hedges before the unrolled side is built, so this runs without the torch backend.
    scm = LinearCyclicSCM([[0.0, 1.0], [1.0, 0.0]], ["x0", "x1"])
    cert = compare_equilibrium_unrolling(scm)
    assert cert.kind is Kind.EMPIRICAL
    assert cert.value is None
    assert cert.hedge is not None and "not uniquely solvable" in cert.hedge.reason


def test_certificate_serializes() -> None:
    scm = _feedback_scm()
    cert = _convergence_certificate(
        scm, None, scm.solve().mean, _numpy_unrolled_mean(scm, None, 400), 400, 1e-6, 0
    )
    restored = Certificate.from_json(cert.to_json())
    assert restored.kind is cert.kind
    assert restored.value == pytest.approx(cert.value)


@pytest.mark.skipif(not _TORCH_OK, reason="requires the torch backend (build_unrolled_scm)")
def test_end_to_end_shipped_unroller_reaches_the_equilibrium() -> None:
    scm = _feedback_scm()
    cert = compare_equilibrium_unrolling(scm, horizon=400, tol=1e-3, seed=0)
    assert cert.kind is Kind.IDENTIFIED
    assert cert.value is not None and cert.value < 1e-3
    from causalrl.experimental.cyclic._unroll import unrolled_state_mean

    shipped = unrolled_state_mean(scm, None, 400, 0)
    np.testing.assert_allclose(shipped, _numpy_unrolled_mean(scm, None, 400), atol=1e-5)


@pytest.mark.skipif(not _TORCH_OK, reason="requires the torch backend (build_unrolled_scm)")
def test_end_to_end_under_intervention() -> None:
    scm = _feedback_scm()
    cert = compare_equilibrium_unrolling(scm, do={"x0": 5.0}, horizon=400, tol=1e-3, seed=1)
    assert cert.kind is Kind.IDENTIFIED
    assert cert.witness is not None
    np.testing.assert_allclose(cert.witness.detail["equilibrium_mean"], [5.0, 4.5], atol=1e-6)
