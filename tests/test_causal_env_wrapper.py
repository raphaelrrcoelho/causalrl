"""Tests for CausalEnvWrapper: gymnasium API conformance and causal interface."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from causalrl.envs.base import CausalEnv
from causalrl.envs.suite.scbandit import make_confounded_chain_env
from causalrl.envs.wrapper import CausalEnvWrapper

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chain_wrapper(n_mc: int = 10) -> CausalEnvWrapper:
    """Wrap the small confounded chain env for testing."""
    env = make_confounded_chain_env(n_mc=n_mc, seed=0)
    return CausalEnvWrapper(env, reward_node="Y")


# ---------------------------------------------------------------------------
# Gymnasium API conformance
# ---------------------------------------------------------------------------


def test_wrapper_is_gymnasium_env() -> None:
    wrapper = _make_chain_wrapper()
    assert isinstance(wrapper, gym.Env)


def test_wrapper_passes_gymnasium_checker() -> None:
    wrapper = _make_chain_wrapper()
    check_env(wrapper, skip_render_check=True)


def test_wrapper_observation_space_matches_inner() -> None:
    inner = make_confounded_chain_env(n_mc=10, seed=0)
    wrapper = CausalEnvWrapper(inner, reward_node="Y")
    assert wrapper.observation_space == inner.observation_space


def test_wrapper_action_space_matches_inner() -> None:
    inner = make_confounded_chain_env(n_mc=10, seed=0)
    wrapper = CausalEnvWrapper(inner, reward_node="Y")
    assert wrapper.action_space == inner.action_space


def test_wrapper_reset_returns_5_tuple_shape() -> None:
    wrapper = _make_chain_wrapper()
    result = wrapper.reset(seed=0)
    obs, info = result
    assert isinstance(obs, dict)
    assert isinstance(info, dict)


def test_wrapper_step_returns_5_tuple() -> None:
    wrapper = _make_chain_wrapper()
    wrapper.reset(seed=0)
    _obs, reward, terminated, truncated, _info = wrapper.step(0)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)


def test_wrapper_multiple_episodes_consistent() -> None:
    """Wrapper can be reset and stepped across multiple episodes without error."""
    wrapper = _make_chain_wrapper()
    for _ in range(3):
        wrapper.reset()
        _obs, _r, terminated, truncated, _info = wrapper.step(0)
        assert terminated or truncated  # bandit env ends after one step


# ---------------------------------------------------------------------------
# Causal interface
# ---------------------------------------------------------------------------


def test_reward_parents_are_list_of_strings() -> None:
    wrapper = _make_chain_wrapper()
    parents = wrapper.reward_parents
    assert isinstance(parents, list)
    assert all(isinstance(p, str) for p in parents)


def test_reward_parents_are_graph_parents_of_reward_node() -> None:
    """reward_parents must match the SCM graph's direct parents of the reward node."""
    wrapper = _make_chain_wrapper()
    expected = wrapper.scm.graph.parents("Y")
    assert wrapper.reward_parents == expected


def test_reward_parents_nonempty_for_chain_env() -> None:
    """The chain env Y = f(X3, U) so Y has parents [X3, U] (at least one)."""
    wrapper = _make_chain_wrapper()
    assert len(wrapper.reward_parents) >= 1


def test_do_returns_mutilated_scm_not_same_object() -> None:
    """do() must return a NEW mutilated SCM, not the original."""
    wrapper = _make_chain_wrapper()
    original_scm = wrapper.scm
    mutilated = wrapper.do({"X1": 0.0})
    assert mutilated is not original_scm


def test_do_does_not_mutate_original_scm() -> None:
    """do() is a pure query — the env's live SCM is unchanged."""
    wrapper = _make_chain_wrapper()
    original_nodes = list(wrapper.scm.graph.nodes)
    wrapper.do({"X1": 1.0})
    assert list(wrapper.scm.graph.nodes) == original_nodes


def test_intervene_agrees_with_do() -> None:
    """intervene(node, value) must be identical to do({node: value})."""
    wrapper = _make_chain_wrapper()
    via_do = wrapper.do({"X1": 1.0})
    via_intervene = wrapper.intervene("X1", 1.0)
    # Both SCMs should produce the same samples under the same seed.
    samples_do = via_do.see(50, seed=42)
    samples_intervene = via_intervene.see(50, seed=42)
    for node in samples_do:
        np.testing.assert_array_almost_equal(
            samples_do[node].numpy(),
            samples_intervene[node].numpy(),
            err_msg=f"Mismatch at node {node!r}",
        )


def test_intervene_on_reward_node_fixes_Y() -> None:
    """do(Y=0.0) should clamp Y to 0 in samples from the mutilated SCM."""
    wrapper = _make_chain_wrapper()
    mutilated = wrapper.intervene("Y", 0.0)
    samples = mutilated.see(100, seed=0)
    np.testing.assert_array_almost_equal(samples["Y"].numpy(), np.zeros(100), decimal=5)


def test_scm_property_proxies_to_inner_env() -> None:
    inner = make_confounded_chain_env(n_mc=10, seed=0)
    wrapper = CausalEnvWrapper(inner, reward_node="Y")
    assert wrapper.scm is inner.scm


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_raises_if_env_has_no_scm_attribute() -> None:
    class _NoSCMEnv(gym.Env):  # type: ignore[type-arg]
        observation_space = gym.spaces.Discrete(1)
        action_space = gym.spaces.Discrete(1)

        def reset(
            self, *, seed: int | None = None, options: dict[str, Any] | None = None
        ) -> tuple[int, dict[str, Any]]:  # type: ignore[override]
            return 0, {}

        def step(self, action: int) -> tuple[int, float, bool, bool, dict[str, Any]]:  # type: ignore[override]
            return 0, 0.0, True, False, {}

    with pytest.raises(ValueError, match=r"non-None \.scm"):
        CausalEnvWrapper(_NoSCMEnv(), reward_node="Y")


def test_raises_if_reward_node_not_in_scm() -> None:
    inner = make_confounded_chain_env(n_mc=10, seed=0)
    with pytest.raises(ValueError, match="reward_node"):
        CausalEnvWrapper(inner, reward_node="DOES_NOT_EXIST")


def test_raises_if_env_scm_is_none() -> None:
    """ConfoundedMDP has scm=None; wrapping it should raise clearly."""
    from causalrl.envs.suite.gridworld import ConfoundedGridworld

    env = ConfoundedGridworld(size=2, seed=0)
    with pytest.raises(ValueError, match=r"non-None \.scm"):
        CausalEnvWrapper(env, reward_node="R")


# ---------------------------------------------------------------------------
# CausalEnv base-class: ensure demo SCM envs conform to gymnasium API
# ---------------------------------------------------------------------------


def test_structural_causal_bandit_env_is_causal_env() -> None:
    env = make_confounded_chain_env(n_mc=10, seed=0)
    assert isinstance(env, CausalEnv)
    assert isinstance(env, gym.Env)
