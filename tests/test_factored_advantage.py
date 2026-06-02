"""Tests for the factored_advantage primitive and FactoredAdvantageConfig."""

from __future__ import annotations

import numpy as np
import pytest

from causalrl.agents.factored_advantage import FactoredAdvantageConfig, factored_advantage

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
