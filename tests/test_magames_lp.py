"""Pure-numpy two-phase simplex (plan: CCE polytope bounds need an LP, core ships no scipy)."""

from __future__ import annotations

import numpy as np
import pytest

from causalrl.magames._lp import LPResult, solve_lp


def test_min_over_simplex_puts_mass_on_smallest_coefficient() -> None:
    res = solve_lp([3.0, 1.0, 2.0], a_eq=[[1.0, 1.0, 1.0]], b_eq=[1.0])
    assert res.status == "optimal"
    assert res.value == pytest.approx(1.0)
    assert res.x is not None
    np.testing.assert_allclose(res.x, [0.0, 1.0, 0.0], atol=1e-9)


def test_max_via_negated_objective() -> None:
    res = solve_lp([-3.0, -1.0, -2.0], a_eq=[[1.0, 1.0, 1.0]], b_eq=[1.0])
    assert res.status == "optimal"
    assert res.value is not None and -res.value == pytest.approx(3.0)


def test_two_binding_inequalities() -> None:
    # min -x1 - x2  s.t.  x1 + 2 x2 <= 4,  3 x1 + x2 <= 6  ->  x = (8/5, 6/5), value -14/5
    res = solve_lp([-1.0, -1.0], a_ub=[[1.0, 2.0], [3.0, 1.0]], b_ub=[4.0, 6.0])
    assert res.status == "optimal"
    assert res.value == pytest.approx(-14.0 / 5.0)
    assert res.x is not None
    np.testing.assert_allclose(res.x, [8.0 / 5.0, 6.0 / 5.0], atol=1e-8)


def test_infeasible_detected() -> None:
    res = solve_lp([1.0, 1.0], a_eq=[[1.0, 1.0]], b_eq=[-1.0])
    assert res == LPResult("infeasible", None, None)


def test_unbounded_detected() -> None:
    res = solve_lp([-1.0])
    assert res.status == "unbounded"


def test_degenerate_face_terminates() -> None:
    # The objective is constant on the optimal face; Bland's rule must still terminate.
    res = solve_lp([1.0, 1.0], a_eq=[[1.0, 1.0]], b_eq=[1.0])
    assert res.status == "optimal"
    assert res.value == pytest.approx(1.0)


def test_negative_rhs_row_is_flipped() -> None:
    # x1 >= 1 encoded as -x1 <= -1; minimum of x1 is 1.
    res = solve_lp([1.0], a_ub=[[-1.0]], b_ub=[-1.0])
    assert res.status == "optimal"
    assert res.value == pytest.approx(1.0)


def test_mixed_equality_and_inequality() -> None:
    # Probability vector with x1 capped at 0.25: max 3 x1 + 2 x3 fills the cap first,
    # then puts the rest on x3 -> x = (0.25, 0, 0.75), value 3(0.25) + 2(0.75) = 2.25.
    res = solve_lp(
        [-3.0, 0.0, -2.0],
        a_ub=[[1.0, 0.0, 0.0]],
        b_ub=[0.25],
        a_eq=[[1.0, 1.0, 1.0]],
        b_eq=[1.0],
    )
    assert res.status == "optimal"
    assert res.value == pytest.approx(-2.25)
    assert res.x is not None
    np.testing.assert_allclose(res.x, [0.25, 0.0, 0.75], atol=1e-8)
