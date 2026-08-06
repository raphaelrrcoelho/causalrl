"""Tests for the factored_advantage primitive and FactoredAdvantageConfig."""

from __future__ import annotations

import numpy as np
import pytest

from causalrl.agents.factored_advantage import (
    FactoredAdvantageConfig,
    blend_advantages,
    factor_gae,
    factor_rewards,
    factored_advantage,
)

# ---------------------------------------------------------------------------
# Single-factor: must reduce to standard advantage
# ---------------------------------------------------------------------------


def test_single_factor_reduces_to_standard_advantage() -> None:
    """With K=1 factor, factored_advantage == V - baseline (standard advantage)."""
    V = np.array([[2.0], [3.0], [1.0]])
    b = np.array([1.5, 2.5, 0.5])
    result = factored_advantage(V, b)
    expected = np.array([0.5, 0.5, 0.5])
    np.testing.assert_array_almost_equal(result, expected)


def test_single_factor_negative_advantage() -> None:
    V = np.array([[0.5]])
    b = np.array([1.5])
    result = factored_advantage(V, b)
    np.testing.assert_array_almost_equal(result, np.array([-1.0]))


# ---------------------------------------------------------------------------
# Two-factor sum (hand-computed)
# ---------------------------------------------------------------------------


def test_two_factor_sum_hand_computed() -> None:
    """A_i = V_i - b; combined = A_0 + A_1."""
    # Step 0: V=[2, 1], b=1.5 -> A=[0.5, -0.5] -> sum=0.0
    # Step 1: V=[3, 0.5], b=2.0 -> A=[1.0, -1.5] -> sum=-0.5
    V = np.array([[2.0, 1.0], [3.0, 0.5]])
    b = np.array([1.5, 2.0])
    result = factored_advantage(V, b)
    expected = np.array([0.0, -0.5])
    np.testing.assert_array_almost_equal(result, expected)


def test_two_factor_mean_aggregation() -> None:
    """aggregation='mean' should equal sum / K."""
    V = np.array([[2.0, 1.0], [3.0, 0.5]])
    b = np.array([1.5, 2.0])
    result_sum = factored_advantage(V, b, aggregation="sum")
    result_mean = factored_advantage(V, b, aggregation="mean")
    np.testing.assert_array_almost_equal(result_mean, result_sum / 2.0)


# ---------------------------------------------------------------------------
# Weighted combination
# ---------------------------------------------------------------------------


def test_weighted_two_factors() -> None:
    """With weights [2, 1]: combined = 2*A_0 + 1*A_1."""
    V = np.array([[3.0, 1.0]])
    b = np.array([2.0])
    # A_0 = 1.0, A_1 = -1.0 -> 2*1 + 1*(-1) = 1.0
    w = np.array([2.0, 1.0])
    result = factored_advantage(V, b, weights=w)
    np.testing.assert_array_almost_equal(result, np.array([1.0]))


def test_weighted_mean_aggregation() -> None:
    """Weighted mean: Σ(w_i * A_i) / Σ(w_i)."""
    V = np.array([[3.0, 1.0]])
    b = np.array([2.0])
    w = np.array([2.0, 1.0])
    result = factored_advantage(V, b, weights=w, aggregation="mean")
    # Σ(w_i * A_i) = 2*1 + 1*(-1) = 1.0; Σw = 3 -> 1/3
    np.testing.assert_array_almost_equal(result, np.array([1.0 / 3.0]))


# ---------------------------------------------------------------------------
# FactoredAdvantageConfig path
# ---------------------------------------------------------------------------


def test_config_sum_matches_direct_call() -> None:
    V = np.array([[2.0, 1.0], [3.0, 0.5]])
    b = np.array([1.5, 2.0])
    config = FactoredAdvantageConfig(factor_nodes=["X3", "U"], aggregation="sum")
    result_config = factored_advantage(V, b, config=config)
    result_direct = factored_advantage(V, b, aggregation="sum")
    np.testing.assert_array_almost_equal(result_config, result_direct)


def test_config_mean_matches_direct_call() -> None:
    V = np.array([[2.0, 1.0], [3.0, 0.5]])
    b = np.array([1.5, 2.0])
    config = FactoredAdvantageConfig(factor_nodes=["X3", "U"], aggregation="mean")
    result_config = factored_advantage(V, b, config=config)
    result_direct = factored_advantage(V, b, aggregation="mean")
    np.testing.assert_array_almost_equal(result_config, result_direct)


def test_config_with_weights() -> None:
    V = np.array([[3.0, 1.0]])
    b = np.array([2.0])
    config = FactoredAdvantageConfig(factor_nodes=["A", "B"], weights=[2.0, 1.0])
    result = factored_advantage(V, b, config=config)
    np.testing.assert_array_almost_equal(result, np.array([1.0]))


def test_config_overrides_kwargs() -> None:
    """When config is given, aggregation/weights kwargs must be ignored."""
    V = np.array([[2.0, 1.0]])
    b = np.array([1.5])
    config = FactoredAdvantageConfig(factor_nodes=["A", "B"], aggregation="mean")
    # Pass aggregation="sum" as kwarg — config's "mean" should win.
    result_config = factored_advantage(V, b, config=config, aggregation="sum")
    result_mean = factored_advantage(V, b, aggregation="mean")
    np.testing.assert_array_almost_equal(result_config, result_mean)


# ---------------------------------------------------------------------------
# Return shape
# ---------------------------------------------------------------------------


def test_output_shape_T() -> None:
    T, K = 10, 4
    V = np.random.default_rng(0).standard_normal((T, K))
    b = np.random.default_rng(1).standard_normal(T)
    result = factored_advantage(V, b)
    assert result.shape == (T,)


def test_single_step_single_factor() -> None:
    V = np.array([[5.0]])
    b = np.array([3.0])
    result = factored_advantage(V, b)
    assert result.shape == (1,)
    np.testing.assert_array_almost_equal(result, np.array([2.0]))


# ---------------------------------------------------------------------------
# Integration with CausalEnvWrapper.reward_parents
# ---------------------------------------------------------------------------


def test_factor_nodes_match_reward_parents() -> None:
    """FactoredAdvantageConfig.factor_nodes == wrapper.reward_parents — end-to-end wiring."""
    from causalrl.envs.suite.scbandit import make_confounded_chain_env
    from causalrl.envs.wrapper import CausalEnvWrapper

    env = make_confounded_chain_env(n_mc=10, seed=0)
    wrapper = CausalEnvWrapper(env, reward_node="Y")
    parents = wrapper.reward_parents  # e.g. ["X3", "U"]
    config = FactoredAdvantageConfig(factor_nodes=parents)
    K = len(parents)
    T = 5
    V = np.ones((T, K)) * 2.0
    b = np.ones(T) * 1.5
    result = factored_advantage(V, b, config=config)
    # With all V_i=2, b=1.5: A_i=0.5 for each factor; sum = 0.5*K
    assert result.shape == (T,)
    np.testing.assert_array_almost_equal(result, np.full(T, 0.5 * K))


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_raises_on_1d_factor_values() -> None:
    with pytest.raises(ValueError, match="2-D"):
        factored_advantage(np.array([1.0, 2.0]), np.array([0.5, 1.0]))


def test_raises_on_baselines_shape_mismatch() -> None:
    V = np.array([[1.0, 2.0], [3.0, 4.0]])
    b = np.array([1.0])  # wrong length
    with pytest.raises(ValueError, match="baselines must have shape"):
        factored_advantage(V, b)


def test_raises_on_weights_shape_mismatch() -> None:
    V = np.array([[1.0, 2.0]])
    b = np.array([0.0])
    wrong_weights = np.array([1.0, 2.0, 3.0])  # K=3 but we have K=2
    with pytest.raises(ValueError, match="weights must have shape"):
        factored_advantage(V, b, weights=wrong_weights)


def test_config_raises_on_wrong_weights_length() -> None:
    with pytest.raises(ValueError, match="weights length"):
        FactoredAdvantageConfig(factor_nodes=["A", "B"], weights=[1.0, 2.0, 3.0])


def test_config_raises_on_factor_columns_mismatch() -> None:
    config = FactoredAdvantageConfig(factor_nodes=["A", "B"])  # K=2
    V = np.array([[1.0, 2.0, 3.0]])  # K=3
    b = np.array([0.0])
    with pytest.raises(ValueError, match="FactoredAdvantageConfig has"):
        factored_advantage(V, b, config=config)


# ---------------------------------------------------------------------------
# Top-level export
# ---------------------------------------------------------------------------


def test_factored_advantage_importable_from_top_level() -> None:
    import causalrl

    assert hasattr(causalrl, "factored_advantage")
    assert hasattr(causalrl, "FactoredAdvantageConfig")


def test_causal_env_wrapper_importable_from_top_level() -> None:
    import causalrl

    assert hasattr(causalrl, "CausalEnvWrapper")


# ---------------------------------------------------------------------------
# factor_rewards — the CGFA wrapper's per-step quantity (arXiv:2605.06066 §E.1)
# ---------------------------------------------------------------------------


def test_factor_rewards_is_the_first_difference_of_the_trace() -> None:
    """r^factor_{k,t} = phi_k(s_{t+1}) - phi_k(s_t), independently per factor."""
    phi = np.array([[0.0, 1.0], [1.0, 1.0], [1.0, 4.0]])  # (T+1=3, K=2)
    np.testing.assert_array_almost_equal(factor_rewards(phi), np.array([[1.0, 0.0], [0.0, 3.0]]))


def test_factor_rewards_keeps_factors_separate() -> None:
    """A factor that never changes must contribute exactly zero reward at every step."""
    phi = np.array([[0.0, 5.0], [2.0, 5.0], [7.0, 5.0]])
    rewards = factor_rewards(phi)
    assert np.all(rewards[:, 1] == 0.0)
    np.testing.assert_array_almost_equal(rewards[:, 0], np.array([2.0, 5.0]))


def test_factor_rewards_raises_on_1d() -> None:
    with pytest.raises(ValueError, match="2-D"):
        factor_rewards(np.array([1.0, 2.0]))


def test_factor_rewards_raises_on_single_row() -> None:
    with pytest.raises(ValueError, match="at least 2 rows"):
        factor_rewards(np.array([[1.0, 2.0]]))


# ---------------------------------------------------------------------------
# factor_gae — Eq. 8 (per-factor return) and Eq. 10 (per-factor advantage)
# ---------------------------------------------------------------------------


def test_factor_gae_at_lambda_one_is_the_monte_carlo_return() -> None:
    """lam=1, zero bootstrap: G_{k,t} = sum_i gamma^i r_{k,t+i} exactly (Eq. 8)."""
    r = np.array([[1.0, 0.0], [0.0, 1.0], [2.0, 0.0]])
    V = np.zeros((3, 2))
    _, returns = factor_gae(r, V, gamma=0.5)
    expected = np.array(
        [
            [1.0 + 0.5 * 0.0 + 0.25 * 2.0, 0.0 + 0.5 * 1.0 + 0.25 * 0.0],
            [0.0 + 0.5 * 2.0, 1.0 + 0.5 * 0.0],
            [2.0, 0.0],
        ]
    )
    np.testing.assert_array_almost_equal(returns, expected)


def test_factor_gae_advantage_is_return_minus_the_per_factor_value() -> None:
    """Eq. 10: A_{k,t} = G_{k,t} - V_k(s_t) — a PER-FACTOR baseline, not a shared scalar."""
    rng = np.random.default_rng(0)
    r = rng.standard_normal((6, 3))
    V = rng.standard_normal((6, 3))
    adv, returns = factor_gae(r, V, gamma=0.9)
    np.testing.assert_array_almost_equal(adv, returns - V)


def test_factor_gae_factors_do_not_leak_into_each_other() -> None:
    """Changing factor 0's rewards must leave factor 1's advantage bit-identical."""
    r = np.array([[1.0, 1.0], [1.0, 1.0]])
    V = np.zeros((2, 2))
    base, _ = factor_gae(r, V, gamma=0.9)
    perturbed_r = r.copy()
    perturbed_r[:, 0] += 100.0
    perturbed, _ = factor_gae(perturbed_r, V, gamma=0.9)
    np.testing.assert_array_almost_equal(perturbed[:, 1], base[:, 1])
    assert not np.allclose(perturbed[:, 0], base[:, 0])


def test_factor_gae_bootstrap_extends_the_truncated_return() -> None:
    """A truncated rollout picks up gamma * V_k(s_T) at the last step."""
    r = np.zeros((1, 2))
    V = np.zeros((1, 2))
    _, returns = factor_gae(r, V, gamma=0.5, bootstrap_values=np.array([4.0, 8.0]))
    np.testing.assert_array_almost_equal(returns, np.array([[2.0, 4.0]]))


def test_factor_gae_done_flag_cuts_the_bootstrap() -> None:
    """dones[t] true zeroes the gamma * V(s_{t+1}) term at step t."""
    r = np.zeros((1, 2))
    V = np.zeros((1, 2))
    _, returns = factor_gae(
        r, V, gamma=0.5, bootstrap_values=np.array([4.0, 8.0]), dones=np.array([True])
    )
    np.testing.assert_array_almost_equal(returns, np.zeros((1, 2)))


def test_factor_gae_lambda_below_one_shortens_the_credit_horizon() -> None:
    """lam < 1 down-weights distant per-factor rewards relative to the lam=1 return."""
    r = np.array([[0.0], [1.0]])
    V = np.zeros((2, 1))
    mc, _ = factor_gae(r, V, gamma=1.0, lam=1.0)
    truncated, _ = factor_gae(r, V, gamma=1.0, lam=0.5)
    assert mc[0, 0] == pytest.approx(1.0)
    assert truncated[0, 0] == pytest.approx(0.5)


def test_factor_gae_raises_on_1d_rewards() -> None:
    with pytest.raises(ValueError, match="rewards must be 2-D"):
        factor_gae(np.array([1.0, 2.0]), np.zeros((2, 1)), gamma=0.9)


def test_factor_gae_raises_on_value_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="same shape as rewards"):
        factor_gae(np.zeros((3, 2)), np.zeros((3, 1)), gamma=0.9)


def test_factor_gae_raises_on_bootstrap_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="bootstrap_values must have shape"):
        factor_gae(np.zeros((3, 2)), np.zeros((3, 2)), gamma=0.9, bootstrap_values=np.zeros(3))


def test_factor_gae_raises_on_dones_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="dones must have shape"):
        factor_gae(np.zeros((3, 2)), np.zeros((3, 2)), gamma=0.9, dones=np.zeros(2, dtype=bool))


# ---------------------------------------------------------------------------
# blend_advantages — the Eq. 11 residual blend
# ---------------------------------------------------------------------------


def test_blend_gate_zero_is_exactly_vanilla_ppo() -> None:
    """g -> 0 must reproduce the scalar advantage bit-for-bit, whatever the factors say."""
    a_scalar = np.array([1.0, -2.0])
    a_factor = np.array([[100.0, -100.0], [50.0, 50.0]])
    np.testing.assert_array_almost_equal(blend_advantages(a_scalar, a_factor, gate=0.0), a_scalar)


def test_blend_gate_one_is_exactly_the_weighted_factor_sum() -> None:
    """g -> 1 must discard the scalar advantage entirely."""
    a_scalar = np.array([999.0])
    a_factor = np.array([[4.0, 0.0]])
    result = blend_advantages(a_scalar, a_factor, gate=1.0, weights=np.array([0.25, 0.75]))
    np.testing.assert_array_almost_equal(result, np.array([1.0]))


def test_blend_interpolates_per_step_with_a_state_conditional_gate() -> None:
    """A per-step gate mixes each step independently."""
    a_scalar = np.array([0.0, 0.0, 0.0])
    a_factor = np.array([[4.0, 0.0], [4.0, 0.0], [4.0, 0.0]])
    result = blend_advantages(a_scalar, a_factor, gate=np.array([0.0, 0.5, 1.0]))
    np.testing.assert_array_almost_equal(result, np.array([0.0, 1.0, 2.0]))


def test_blend_weights_reweight_the_factors() -> None:
    """Mixture weights select which causal factor drives the update."""
    a_scalar = np.zeros(1)
    a_factor = np.array([[1.0, -1.0]])
    toward_first = blend_advantages(a_scalar, a_factor, gate=1.0, weights=np.array([0.9, 0.1]))
    toward_second = blend_advantages(a_scalar, a_factor, gate=1.0, weights=np.array([0.1, 0.9]))
    assert toward_first[0] == pytest.approx(0.8)
    assert toward_second[0] == pytest.approx(-0.8)


def test_blend_raises_on_out_of_range_gate() -> None:
    with pytest.raises(ValueError, match=r"gate must lie in \[0, 1\]"):
        blend_advantages(np.zeros(1), np.zeros((1, 2)), gate=1.5)


def test_blend_raises_on_1d_factor_advantages() -> None:
    with pytest.raises(ValueError, match="factor_advantages must be 2-D"):
        blend_advantages(np.zeros(2), np.zeros(2), gate=0.5)


def test_blend_raises_on_scalar_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="scalar_advantages must have shape"):
        blend_advantages(np.zeros(3), np.zeros((2, 2)), gate=0.5)


def test_blend_raises_on_gate_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="gate must be a scalar"):
        blend_advantages(np.zeros(2), np.zeros((2, 2)), gate=np.zeros(3))


def test_blend_raises_on_weights_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="weights must have shape"):
        blend_advantages(np.zeros(2), np.zeros((2, 2)), gate=0.5, weights=np.zeros(3))


# ---------------------------------------------------------------------------
# The pure-NumPy half must stay pure: importing it must never pull in torch.
# ---------------------------------------------------------------------------


def test_factored_advantage_module_imports_without_torch() -> None:
    """causalrl.agents.factored_advantage is framework-agnostic — no torch, at any depth."""
    import subprocess
    import sys

    source = """
import builtins

original_import = builtins.__import__

def import_without_torch(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "torch" or name.startswith("torch."):
        raise ModuleNotFoundError("No module named 'torch'", name="torch")
    return original_import(name, globals, locals, fromlist, level)

builtins.__import__ = import_without_torch

import numpy as np
from causalrl.agents.factored_advantage import (
    blend_advantages,
    factor_gae,
    factor_rewards,
    factored_advantage,
)

phi = np.array([[0.0, 0.0], [1.0, 2.0]])
r = factor_rewards(phi)
adv, ret = factor_gae(r, np.zeros_like(r), gamma=0.9)
out = blend_advantages(np.zeros(1), adv, gate=0.5)
assert out.shape == (1,)
assert factored_advantage(np.ones((1, 2)), np.zeros(1)).shape == (1,)

import sys
assert "torch" not in sys.modules, "importing the CGFA arithmetic pulled in torch"
"""
    result = subprocess.run(
        [sys.executable, "-c", source], check=False, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
