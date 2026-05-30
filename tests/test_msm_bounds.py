"""Tests for the Interval return type and migrated MSM/partial-ID bounds."""

import numpy as np

from causalrl.identification.bounds import (
    Interval,
    ipw_sensitivity_bounds,
    manski_bounds,
    msm_per_step_bounds,
    msm_stratified_bounds,
)


def test_interval_is_tuple_compatible():
    iv = Interval(0.2, 0.8)
    lo, hi = iv  # unpacks like a tuple
    assert (lo, hi) == (0.2, 0.8)
    assert iv[0] == 0.2 and iv[1] == 0.8
    assert iv.lower == 0.2 and iv.upper == 0.8


def test_manski_returns_interval():
    data = {"t": [1, 1, 0, 0], "y": [1.0, 0.0, 1.0, 0.0]}
    iv = manski_bounds(data, treatment="t", outcome="y", action=1)
    assert isinstance(iv, Interval)
    assert iv.lower <= iv.upper


def test_per_step_widens_with_gamma_and_returns_interval():
    rng = np.random.default_rng(0)
    rewards = [rng.uniform(0, 1, size=200) for _ in range(3)]
    props = [rng.uniform(0.2, 0.8, size=200) for _ in range(3)]
    iv1 = msm_per_step_bounds(rewards, props, gamma=1.0)
    iv2 = msm_per_step_bounds(rewards, props, gamma=2.0)
    assert isinstance(iv1, Interval)
    assert iv1.lower <= iv1.upper
    assert (iv2.upper - iv2.lower) >= (iv1.upper - iv1.lower)  # monotone in gamma


def test_per_step_gamma1_collapses_to_point():
    rng = np.random.default_rng(1)
    r = [rng.uniform(0, 1, size=100)]
    p = [rng.uniform(0.3, 0.7, size=100)]
    iv = msm_per_step_bounds(r, p, gamma=1.0)
    assert iv.upper - iv.lower < 1e-6


def test_stratified_never_wider_than_pooled():
    # Prop 1 (THEORY.md): stratified MSM is never wider than pooled MSM.
    rng = np.random.default_rng(2)
    v = rng.uniform(0, 1, size=300)
    p = rng.uniform(0.2, 0.8, size=300)
    strata = rng.integers(0, 3, size=300)
    weights = {s: 1 / 3 for s in range(3)}
    strat = msm_stratified_bounds(v, p, strata, weights, gamma=2.0)
    pooled = ipw_sensitivity_bounds(v.tolist(), p.tolist(), gamma=2.0)
    assert isinstance(strat, Interval)
    assert (strat.upper - strat.lower) <= (pooled.upper - pooled.lower) + 1e-9
