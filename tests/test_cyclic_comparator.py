"""Equilibrium-vs-unrolling comparator (plan §11: contraction agreement, empirical/hedge otherwise).

The unrolled side is the exact linear mean dynamics (pure NumPy), so every test runs everywhere.
"""

from __future__ import annotations

import numpy as np
import pytest

from causalrl.certify.certificate import Certificate, Kind
from causalrl.experimental.cyclic import LinearCyclicSCM, compare_equilibrium_unrolling
from causalrl.experimental.cyclic.comparator import _convergence_certificate


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


def test_end_to_end_reaches_the_equilibrium() -> None:
    scm = _feedback_scm()
    cert = compare_equilibrium_unrolling(scm, horizon=400, tol=1e-3, seed=0)
    assert cert.kind is Kind.IDENTIFIED
    assert cert.value is not None and cert.value < 1e-3
    # the unrolled mean converges to the closed-form equilibrium [8/3, 10/3]
    assert cert.witness is not None
    got = cert.witness.detail["unrolled_mean"]
    np.testing.assert_allclose(got, [8.0 / 3.0, 10.0 / 3.0], atol=1e-3)


def test_end_to_end_under_intervention() -> None:
    scm = _feedback_scm()
    cert = compare_equilibrium_unrolling(scm, do={"x0": 5.0}, horizon=400, tol=1e-3, seed=1)
    assert cert.kind is Kind.IDENTIFIED
    assert cert.witness is not None
    np.testing.assert_allclose(cert.witness.detail["equilibrium_mean"], [5.0, 4.5], atol=1e-6)


def test_naive_hedge_carries_stability_diagnostics() -> None:
    # rho(B) = 2 but margin = 3 > 0: the hedge must say the mean dynamics are still stable and
    # name the learning rate below which the adaptive unrolling converges.
    scm = LinearCyclicSCM([[-2.0]], ["x0"], noise_mean=[3.0])
    cert = compare_equilibrium_unrolling(scm, horizon=50)
    assert cert.kind is Kind.EMPIRICAL
    assert cert.hedge is not None and "non-contractive" in cert.hedge.reason
    assert cert.hedge.detail is not None
    assert cert.hedge.detail["stability_margin"] == pytest.approx(3.0)
    assert cert.hedge.detail["max_stable_learning_rate"] == pytest.approx(2.0 / 3.0)
    assert cert.witness is not None
    assert cert.witness.detail["stability_margin"] == pytest.approx(3.0)
    assert cert.witness.detail["spectral_radius"] == pytest.approx(2.0)


def test_learning_rate_certifies_stable_non_contractive_system() -> None:
    # The T1 case: naive unrolling diverges (rho = 2) yet gamma = 0.3 < gamma* = 2/3 makes the
    # damped adaptive dynamics converge to the equilibrium do() (x* = 3 / (1 + 2) = 1).
    scm = LinearCyclicSCM([[-2.0]], ["x0"], noise_mean=[3.0])
    cert = compare_equilibrium_unrolling(scm, horizon=400, tol=1e-6, learning_rate=0.3)
    assert cert.kind is Kind.IDENTIFIED
    assert cert.hedge is None
    assert cert.value is not None and cert.value < 1e-6
    assert cert.witness is not None and cert.witness.detail["learning_rate"] == 0.3
    np.testing.assert_allclose(cert.witness.detail["equilibrium_mean"], [1.0])


def test_learning_rate_above_threshold_hedges_with_the_threshold() -> None:
    scm = LinearCyclicSCM([[-2.0]], ["x0"], noise_mean=[3.0])
    cert = compare_equilibrium_unrolling(scm, horizon=100, learning_rate=0.9)
    assert cert.kind is Kind.EMPIRICAL
    assert cert.hedge is not None and cert.hedge.detail is not None
    assert cert.hedge.detail["max_stable_learning_rate"] == pytest.approx(2.0 / 3.0)


def test_learning_rate_cannot_rescue_unstable_mean_dynamics() -> None:
    scm = LinearCyclicSCM([[1.5]], ["x0"], noise_mean=[1.0])
    cert = compare_equilibrium_unrolling(scm, learning_rate=0.5)
    assert cert.kind is Kind.EMPIRICAL
    assert cert.hedge is not None and cert.hedge.detail is not None
    assert cert.hedge.detail["max_stable_learning_rate"] == 0.0


def test_learning_rate_one_reproduces_naive_dynamics() -> None:
    scm = _feedback_scm()
    naive = compare_equilibrium_unrolling(scm, horizon=100)
    damped = compare_equilibrium_unrolling(scm, horizon=100, learning_rate=1.0)
    assert damped.kind is naive.kind
    assert damped.value == pytest.approx(naive.value)


def test_invalid_learning_rate_raises() -> None:
    with pytest.raises(ValueError, match="learning_rate"):
        compare_equilibrium_unrolling(_feedback_scm(), learning_rate=0.0)
    with pytest.raises(ValueError, match="learning_rate"):
        compare_equilibrium_unrolling(_feedback_scm(), learning_rate=1.5)
