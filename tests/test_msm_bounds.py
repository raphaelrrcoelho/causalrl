"""Tests for the Interval return type and migrated MSM/partial-ID bounds."""

import numpy as np

from causalrl.identification.bounds import (
    Interval,
    ipw_sensitivity_bounds,
    manski_bounds,
    msm_contribution_bounds,
    msm_per_step_bounds,
    msm_policy_value_bounds,
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


def test_policy_value_reduces_to_ipw_for_constant_target():
    # A target that picks every logged action with the SAME probability is the treated/uniform
    # mean: the constant cancels in the self-normalised ratio, so the policy-value MSM bound must
    # equal ipw_sensitivity_bounds (the existing kernel) at every gamma.
    rng = np.random.default_rng(3)
    y = rng.uniform(0, 1, size=400)
    e0 = rng.uniform(0.1, 0.9, size=400)
    pt = np.full(400, 0.017)  # arbitrary constant target propensity
    for g in (1.0, 1.5, 3.0):
        pv = msm_policy_value_bounds(y.tolist(), e0.tolist(), pt.tolist(), gamma=g)
        ipw = ipw_sensitivity_bounds(y.tolist(), e0.tolist(), gamma=g)
        assert abs(pv.lower - ipw.lower) < 1e-9
        assert abs(pv.upper - ipw.upper) < 1e-9


def test_policy_value_gamma1_is_self_normalised_ips_point():
    # At gamma=1 the band collapses to the Hájek IPS point V_hat = sum(w Y)/sum(w), w = pi_t/e0.
    rng = np.random.default_rng(4)
    y = rng.uniform(0, 1, size=300)
    e0 = rng.uniform(0.2, 0.8, size=300)
    pt = rng.uniform(0.0, 0.05, size=300)
    w = pt / e0
    v_hat = float((w * y).sum() / w.sum())
    iv = msm_policy_value_bounds(y.tolist(), e0.tolist(), pt.tolist(), gamma=1.0)
    assert iv.upper - iv.lower < 1e-9
    assert abs(iv.lower - v_hat) < 1e-9


def test_policy_value_widens_with_gamma_and_brackets_point():
    rng = np.random.default_rng(5)
    y = rng.uniform(0, 1, size=300)
    e0 = rng.uniform(0.2, 0.8, size=300)
    pt = rng.uniform(0.0, 0.05, size=300)
    w = pt / e0
    v_hat = float((w * y).sum() / w.sum())
    iv1 = msm_policy_value_bounds(y.tolist(), e0.tolist(), pt.tolist(), gamma=1.5)
    iv2 = msm_policy_value_bounds(y.tolist(), e0.tolist(), pt.tolist(), gamma=3.0)
    assert iv1.lower <= v_hat <= iv1.upper
    assert (iv2.upper - iv2.lower) >= (iv1.upper - iv1.lower)


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


def _disjoint_arms(seed: int, n: int = 300):
    rng = np.random.default_rng(seed)
    y = rng.uniform(0, 1, size=n)
    e0 = rng.uniform(0.2, 0.8, size=n)
    f = rng.integers(0, 2, size=n).astype(float)
    return y.tolist(), e0.tolist(), f.tolist(), (1.0 - f).tolist()  # one-hot on / off, disjoint


def test_contribution_gamma1_is_point_difference():
    # At gamma=1 the contribution interval collapses to the difference of the two arms'
    # self-normalised IPS points.
    y, e0, on, off = _disjoint_arms(7)
    on_iv = msm_policy_value_bounds(y, e0, on, gamma=1.0)
    off_iv = msm_policy_value_bounds(y, e0, off, gamma=1.0)
    iv = msm_contribution_bounds(y, e0, on, off, gamma=1.0)
    assert iv.upper - iv.lower < 1e-9
    assert abs(iv.lower - (on_iv.lower - off_iv.lower)) < 1e-9


def test_contribution_widens_and_brackets_point():
    y, e0, on, off = _disjoint_arms(8)
    d_hat = msm_contribution_bounds(y, e0, on, off, gamma=1.0).lower
    iv15 = msm_contribution_bounds(y, e0, on, off, gamma=1.5)
    iv30 = msm_contribution_bounds(y, e0, on, off, gamma=3.0)
    assert iv15.lower <= d_hat <= iv15.upper  # brackets the point estimate
    assert (iv30.upper - iv30.lower) >= (iv15.upper - iv15.lower)  # monotone in gamma


def test_contribution_equals_arm_interval_difference():
    # Definitional: the contribution interval IS [on.lo - off.hi, on.hi - off.lo].
    y, e0, on, off = _disjoint_arms(9)
    a = msm_policy_value_bounds(y, e0, on, gamma=2.0)
    b = msm_policy_value_bounds(y, e0, off, gamma=2.0)
    iv = msm_contribution_bounds(y, e0, on, off, gamma=2.0)
    assert isinstance(iv, Interval)
    assert abs(iv.lower - (a.lower - b.upper)) < 1e-12
    assert abs(iv.upper - (a.upper - b.lower)) < 1e-12
