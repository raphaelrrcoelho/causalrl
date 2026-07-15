"""LinearCyclicSCM: equilibrium oracles, solvability, typed hedges (plan §11 acceptance)."""

from __future__ import annotations

import numpy as np
import pytest

from causalrl.exceptions import CausalRLError
from causalrl.experimental.cyclic import CyclicSolveError, LinearCyclicSCM


def _feedback_scm() -> LinearCyclicSCM:
    # x0 = 0.5 x1 + u0,  x1 = 0.5 x0 + u1 ;  u ~ N([1, 2], I). A stable 2-cycle.
    return LinearCyclicSCM([[0.0, 0.5], [0.5, 0.0]], ["x0", "x1"], noise_mean=[1.0, 2.0])


def test_equilibrium_matches_closed_form() -> None:
    scm = _feedback_scm()
    sol = scm.solve()
    assert sol.solved
    # (I - B)^{-1} = (1/0.75) [[1, 0.5], [0.5, 1]];  mean = inv @ [1, 2] = [8/3, 10/3].
    np.testing.assert_allclose(sol.mean, [8.0 / 3.0, 10.0 / 3.0], rtol=1e-9)
    inv = np.linalg.inv(np.eye(2) - scm.coefficients)
    np.testing.assert_allclose(sol.cov, inv @ inv.T, rtol=1e-9)


def test_spectral_radius_and_flags() -> None:
    scm = _feedback_scm()
    assert scm.spectral_radius() == pytest.approx(0.5)
    assert scm.is_contractive()
    assert scm.is_uniquely_solvable()
    assert scm.solve().contractive


def test_samples_recover_the_equilibrium_moments() -> None:
    scm = _feedback_scm()
    sol = scm.solve()
    draws = scm.sample(200_000, seed=7)
    assert sol.mean is not None and sol.cov is not None
    np.testing.assert_allclose(draws.mean(axis=0), sol.mean, atol=0.02)
    np.testing.assert_allclose(np.cov(draws, rowvar=False), sol.cov, atol=0.03)


def test_unrolling_converges_to_the_equilibrium_when_contractive() -> None:
    scm = _feedback_scm()
    b, mean_u = scm.coefficients, scm.noise_mean
    x = np.zeros(2)
    for _ in range(200):  # x_{k+1} = B x_k + E[u]
        x = b @ x + mean_u
    np.testing.assert_allclose(x, scm.solve().mean, atol=1e-6)


def test_intervention_pins_the_node_and_cuts_incoming_edges() -> None:
    scm = _feedback_scm()
    sol = scm.solve(do={"x0": 5.0})
    # x0 fixed at 5; x1 = 0.5 * 5 + E[u1] = 2.5 + 2 = 4.5.
    assert sol.solved
    np.testing.assert_allclose(sol.mean, [5.0, 4.5], rtol=1e-9)


def test_context_pins_exogenous_noise_deterministically() -> None:
    scm = _feedback_scm()
    sol = scm.solve(context={"x0": 0.0, "x1": 0.0})
    inv = np.linalg.inv(np.eye(2) - scm.coefficients)
    np.testing.assert_allclose(sol.mean, inv @ np.zeros(2), atol=1e-12)
    draws = scm.sample(1000, context={"x0": 0.0, "x1": 0.0}, seed=1)
    np.testing.assert_allclose(draws.var(axis=0), [0.0, 0.0], atol=1e-12)


def test_self_loop_is_solved_and_reported_cyclic() -> None:
    # x0 = 0.5 x0 + u0  ->  equilibrium x0 = 2 u0 ; graph has a self-loop (not acyclic).
    scm = LinearCyclicSCM([[0.5]], ["x0"], noise_mean=[3.0])
    np.testing.assert_allclose(scm.solve().mean, [6.0], rtol=1e-9)
    assert not scm.graph().is_acyclic()


def test_induced_graph_is_the_two_cycle() -> None:
    scm = _feedback_scm()
    graph = scm.graph()
    assert not graph.is_acyclic()
    sccs = {frozenset(s) for s in graph.strongly_connected_components()}
    assert sccs == {frozenset({"x0", "x1"})}


def test_non_unique_system_hedges_instead_of_inventing_a_solution() -> None:
    # x0 = x1, x1 = x0  ->  I - B singular (det 0): a continuum of equilibria.
    scm = LinearCyclicSCM([[0.0, 1.0], [1.0, 0.0]], ["x0", "x1"])
    sol = scm.solve()
    assert not sol.solved
    assert sol.mean is None and sol.cov is None
    assert not sol.unique
    assert sol.hedge is not None and "not uniquely solvable" in sol.hedge.reason


def test_non_unique_system_refuses_to_sample() -> None:
    scm = LinearCyclicSCM([[0.0, 1.0], [1.0, 0.0]], ["x0", "x1"])
    with pytest.raises(CyclicSolveError):
        scm.sample(10, seed=0)
    with pytest.raises(CausalRLError):  # CyclicSolveError is part of the library error hierarchy
        scm.solve().mean_dict()


def test_shape_validation() -> None:
    with pytest.raises(ValueError, match="square"):
        LinearCyclicSCM([[1.0, 2.0]], ["x0"])
    with pytest.raises(ValueError, match="must match"):
        LinearCyclicSCM([[0.0, 0.0], [0.0, 0.0]], ["only_one"])


def test_noise_shape_validation() -> None:
    with pytest.raises(ValueError, match="noise_mean"):
        LinearCyclicSCM([[0.0]], ["x0"], noise_mean=[1.0, 2.0])
    with pytest.raises(ValueError, match="noise_cov"):
        LinearCyclicSCM([[0.0]], ["x0"], noise_cov=[[1.0, 0.0], [0.0, 1.0]])


def test_mean_dict_on_solved_equilibrium() -> None:
    md = _feedback_scm().solve().mean_dict()
    assert md["x0"] == pytest.approx(8.0 / 3.0)
    assert md["x1"] == pytest.approx(10.0 / 3.0)


def test_intervene_unknown_variable_raises() -> None:
    with pytest.raises(KeyError):
        _feedback_scm().intervene({"nope": 1.0})


def test_stability_margin_weaker_than_contractivity() -> None:
    # B = [[-2]]: spectral radius 2 (non-contractive) but eigenvalue real part -2 < 1: the mean
    # dynamics x' = (B - I)x + u are stable with margin 1 - (-2) = 3.
    scm = LinearCyclicSCM([[-2.0]], ["x0"])
    assert scm.spectral_radius() == pytest.approx(2.0)
    assert not scm.is_contractive()
    assert scm.spectral_abscissa() == pytest.approx(-3.0)
    assert scm.stability_margin() == pytest.approx(3.0)


def test_stability_margin_negative_when_unstable() -> None:
    scm = LinearCyclicSCM([[1.5]], ["x0"])
    assert scm.stability_margin() == pytest.approx(-0.5)
    assert scm.max_stable_learning_rate() == 0.0


def test_stability_margin_nilpotent_feedforward() -> None:
    scm = LinearCyclicSCM([[0.0, 2.0], [0.0, 0.0]], ["x0", "x1"])
    assert scm.stability_margin() == pytest.approx(1.0)
    # nu = -1: gamma* = -2 Re(nu) / |nu|^2 = 2 (any admissible learning rate <= 1 converges).
    assert scm.max_stable_learning_rate() == pytest.approx(2.0)


def test_max_stable_learning_rate_matches_eigenvalue_algebra() -> None:
    # B = [[-2]]: nu = -3, gamma* = -2(-3)/9 = 2/3; the damped map 1 + gamma(nu) crosses |.|=1 there.
    scm = LinearCyclicSCM([[-2.0]], ["x0"])
    gamma_star = scm.max_stable_learning_rate()
    assert gamma_star == pytest.approx(2.0 / 3.0)
    inside = abs(1.0 + 0.99 * gamma_star * (-3.0))
    outside = abs(1.0 + 1.01 * gamma_star * (-3.0))
    assert inside < 1.0 < outside


def test_stability_margin_complex_pair_rotation() -> None:
    # Rotation scaled to modulus 1.2: eigenvalues 1.2 e^{+-i 2pi/3} with real part -0.6 < 1:
    # non-contractive yet stable mean dynamics.
    angle = 2.0 * np.pi / 3.0
    rot = 1.2 * np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    scm = LinearCyclicSCM(rot, ["x0", "x1"])
    assert scm.spectral_radius() == pytest.approx(1.2)
    assert scm.stability_margin() == pytest.approx(1.6)
    assert 0.0 < scm.max_stable_learning_rate() <= 2.0
