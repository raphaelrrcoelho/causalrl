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
from causalrl.exceptions import CausalInterfaceUnavailableError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chain_wrapper(n_mc: int = 10) -> CausalEnvWrapper:
    """Wrap the small confounded chain env for testing."""
    env = make_confounded_chain_env(n_mc=n_mc, seed=0)
    return CausalEnvWrapper(env, reward_node="Y")


class _NoSCMEnv(gym.Env[Any, Any]):
    """Minimal env with no .scm attribute at all."""

    observation_space: gym.Space[Any] = gym.spaces.Discrete(1)
    action_space: gym.Space[Any] = gym.spaces.Discrete(1)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[int, dict[str, Any]]:  # type: ignore[override]
        return 0, {}

    def step(self, action: int) -> tuple[int, float, bool, bool, dict[str, Any]]:  # type: ignore[override]
        return 0, 0.0, True, False, {}


# ---------------------------------------------------------------------------
# Gymnasium API conformance
# ---------------------------------------------------------------------------


def test_wrapper_is_gymnasium_env() -> None:
    wrapper = _make_chain_wrapper()
    assert isinstance(wrapper, gym.Env)


def test_wrapper_passes_gymnasium_checker() -> None:
    wrapper = _make_chain_wrapper()
    # Checking the *wrapper* is the point of this test, so gymnasium's "you passed a wrapped env"
    # advisory is expected. Asserting it fires keeps the output pristine and pins the intent: if
    # this ever became check_env(wrapper.unwrapped), the warning would stop and this would fail.
    with pytest.warns(UserWarning, match="different from the unwrapped version"):
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
    expected = wrapper.scm.graph.parents("Y")  # type: ignore[union-attr]
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
    original_nodes = list(wrapper.scm.graph.nodes)  # type: ignore[union-attr]
    wrapper.do({"X1": 1.0})
    assert list(wrapper.scm.graph.nodes) == original_nodes  # type: ignore[union-attr]


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
# has_causal_interface
# ---------------------------------------------------------------------------


def test_has_causal_interface_true_when_scm_and_reward_node() -> None:
    wrapper = _make_chain_wrapper()
    assert wrapper.has_causal_interface is True


def test_has_causal_interface_false_when_no_reward_node() -> None:
    inner = make_confounded_chain_env(n_mc=10, seed=0)
    wrapper = CausalEnvWrapper(inner)  # no reward_node
    assert wrapper.has_causal_interface is False


def test_has_causal_interface_false_when_scm_is_none() -> None:
    from causalrl.envs.suite.gridworld import ConfoundedGridworld

    env = ConfoundedGridworld(size=2, seed=0)
    wrapper = CausalEnvWrapper(env, reward_node="R")
    assert wrapper.has_causal_interface is False


def test_has_causal_interface_false_for_no_scm_env() -> None:
    wrapper = CausalEnvWrapper(_NoSCMEnv())
    assert wrapper.has_causal_interface is False


# ---------------------------------------------------------------------------
# Pass-through mode: scm=None construction succeeds and interface is disabled
# ---------------------------------------------------------------------------


def test_construction_succeeds_when_scm_is_none() -> None:
    """ConfoundedMDP has scm=None; wrapping it must now SUCCEED in pass-through mode."""
    from causalrl.envs.suite.gridworld import ConfoundedGridworld

    env = ConfoundedGridworld(size=2, seed=0)
    wrapper = CausalEnvWrapper(env, reward_node="R")  # no exception
    assert isinstance(wrapper, gym.Env)


def test_construction_succeeds_when_no_reward_node() -> None:
    inner = make_confounded_chain_env(n_mc=10, seed=0)
    wrapper = CausalEnvWrapper(inner)  # no reward_node — still succeeds
    assert isinstance(wrapper, gym.Env)


def test_construction_succeeds_for_env_without_scm_attr() -> None:
    wrapper = CausalEnvWrapper(_NoSCMEnv())
    assert isinstance(wrapper, gym.Env)


def test_reward_parents_raises_when_no_causal_interface() -> None:
    inner = make_confounded_chain_env(n_mc=10, seed=0)
    wrapper = CausalEnvWrapper(inner)  # no reward_node
    with pytest.raises(CausalInterfaceUnavailableError, match="reward_node"):
        _ = wrapper.reward_parents


def test_do_raises_when_scm_is_none() -> None:
    from causalrl.envs.suite.gridworld import ConfoundedGridworld

    env = ConfoundedGridworld(size=2, seed=0)
    wrapper = CausalEnvWrapper(env, reward_node="R")
    with pytest.raises(CausalInterfaceUnavailableError, match="scm=None"):
        wrapper.do({"R": 1.0})


def test_intervene_raises_when_no_causal_interface() -> None:
    inner = make_confounded_chain_env(n_mc=10, seed=0)
    wrapper = CausalEnvWrapper(inner)
    with pytest.raises(CausalInterfaceUnavailableError):
        wrapper.intervene("X1", 1.0)


def test_set_intervention_raises_when_no_causal_interface() -> None:
    from causalrl.envs.suite.gridworld import ConfoundedGridworld

    env = ConfoundedGridworld(size=2, seed=0)
    wrapper = CausalEnvWrapper(env)
    with pytest.raises(CausalInterfaceUnavailableError):
        wrapper.set_intervention({"X": 1.0})


def test_scm_property_returns_none_for_no_scm_env() -> None:
    wrapper = CausalEnvWrapper(_NoSCMEnv())
    assert wrapper.scm is None


def test_passthrough_reset_and_step_work_when_scm_is_none() -> None:
    """In pass-through mode, reset/step still work normally."""
    from causalrl.envs.suite.gridworld import ConfoundedGridworld

    env = ConfoundedGridworld(size=2, seed=0)
    wrapper = CausalEnvWrapper(env)
    obs, info = wrapper.reset(seed=0)
    assert obs is not None
    assert isinstance(info, dict)


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_raises_if_reward_node_not_in_scm() -> None:
    inner = make_confounded_chain_env(n_mc=10, seed=0)
    with pytest.raises(ValueError, match="reward_node"):
        CausalEnvWrapper(inner, reward_node="DOES_NOT_EXIST")


# ---------------------------------------------------------------------------
# Persistent interventional rollouts
# ---------------------------------------------------------------------------


def test_active_interventions_none_by_default() -> None:
    wrapper = _make_chain_wrapper()
    assert wrapper.active_interventions is None


def test_set_intervention_stores_mapping() -> None:
    wrapper = _make_chain_wrapper()
    wrapper.set_intervention({"X3": 1.0})
    assert wrapper.active_interventions is not None
    assert wrapper.active_interventions["X3"] == 1.0


def test_clear_intervention_resets_to_none() -> None:
    wrapper = _make_chain_wrapper()
    wrapper.set_intervention({"X3": 1.0})
    wrapper.clear_intervention()
    assert wrapper.active_interventions is None


def test_interventional_rollout_shifts_reward_distribution() -> None:
    """Under do(X3=1), Y=[1==U] with U~Bernoulli(0.5) -> E[Y]~0.5.

    The observational empty arm scores ~1.0 because X3 == U always.
    A persistent intervention do(X3=1) breaks this coupling and drives E[Y] to ~0.5.
    We run enough steps (N=200, seed-fixed) that the oracle 0.5 vs 1.0 gap is clear.
    """
    rng = np.random.default_rng(42)
    n_samples = 200

    # Build wrapper with a large enough n_mc for stable arm_values.
    inner = make_confounded_chain_env(n_mc=200, seed=0)
    wrapper = CausalEnvWrapper(inner, reward_node="Y")

    # --- observational arm (action 0 = empty intervention) ---
    obs_rewards: list[float] = []
    for _ in range(n_samples):
        wrapper.reset(seed=int(rng.integers(0, 2**31)))
        _, r, _, _, _ = wrapper.step(0)
        obs_rewards.append(r)
    obs_mean = float(np.mean(obs_rewards))

    # --- interventional rollout: persistent do(X3=1) ---
    # Find the arm index corresponding to do(X3=1).
    arm_x3_1_idx = next(i for i, a in enumerate(inner.arms) if a == {"X3": 1})
    wrapper.set_intervention({"X3": 1.0})
    int_rewards: list[float] = []
    for _ in range(n_samples):
        wrapper.reset(seed=int(rng.integers(0, 2**31)))
        # Under the mutilated SCM step still samples from it regardless of action.
        _, r, _, _, _ = wrapper.step(arm_x3_1_idx)
        int_rewards.append(r)
    wrapper.clear_intervention()
    int_mean = float(np.mean(int_rewards))

    # Oracle: observational ~1.0, interventional ~0.5.
    assert obs_mean > 0.85, f"Expected observational mean > 0.85, got {obs_mean:.3f}"
    assert int_mean < 0.65, f"Expected interventional mean < 0.65, got {int_mean:.3f}"


def test_clear_intervention_restores_original_reward_distribution() -> None:
    """After clear_intervention, reward distribution returns to ~1.0."""
    rng = np.random.default_rng(7)
    n_samples = 100

    inner = make_confounded_chain_env(n_mc=200, seed=0)
    wrapper = CausalEnvWrapper(inner, reward_node="Y")

    # Activate and then immediately clear the intervention.
    wrapper.set_intervention({"X3": 1.0})
    wrapper.clear_intervention()

    rewards: list[float] = []
    for _ in range(n_samples):
        wrapper.reset(seed=int(rng.integers(0, 2**31)))
        _, r, _, _, _ = wrapper.step(0)
        rewards.append(r)

    mean_reward = float(np.mean(rewards))
    assert mean_reward > 0.85, f"Expected restored mean > 0.85, got {mean_reward:.3f}"


def test_scm_is_restored_after_intervention_step() -> None:
    """After each step, the env's SCM must be the original, not the mutilated one."""
    inner = make_confounded_chain_env(n_mc=10, seed=0)
    wrapper = CausalEnvWrapper(inner, reward_node="Y")
    original_scm = inner.scm

    wrapper.set_intervention({"X3": 1.0})
    wrapper.reset(seed=0)
    wrapper.step(0)

    # After step, inner.scm must be the original (not the mutilated copy).
    assert inner.scm is original_scm


def test_scm_is_restored_even_if_step_raises() -> None:
    """try/finally must restore SCM even when step itself raises an exception."""
    inner = make_confounded_chain_env(n_mc=10, seed=0)
    original_scm = inner.scm

    wrapper = CausalEnvWrapper(inner, reward_node="Y")
    wrapper.set_intervention({"X3": 1.0})

    # Patch step to raise so the finally block is exercised.
    original_step = inner.step

    def _raising_step(action: Any) -> Any:
        raise RuntimeError("injected failure")

    inner.step = _raising_step  # type: ignore[method-assign]
    try:
        wrapper.step(0)
    except RuntimeError:
        pass
    finally:
        inner.step = original_step  # type: ignore[method-assign]

    assert inner.scm is original_scm


# ---------------------------------------------------------------------------
# CausalEnv base-class: ensure demo SCM envs conform to gymnasium API
# ---------------------------------------------------------------------------


def test_structural_causal_bandit_env_is_causal_env() -> None:
    env = make_confounded_chain_env(n_mc=10, seed=0)
    assert isinstance(env, CausalEnv)
    assert isinstance(env, gym.Env)
