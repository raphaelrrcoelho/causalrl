"""Tests for Gymnasium env registration (Items 3 and 4).

Covers:
- gymnasium.make("causalrl/...") works after importing causalrl
- gymnasium.make_vec vectorized smoke test
- register_envs() is idempotent
- new IDs appear in the registry
"""

from __future__ import annotations

import gymnasium as gym


def test_import_causalrl_triggers_registration() -> None:
    """Importing causalrl should register the demo environments."""
    import causalrl  # noqa: F401

    assert "causalrl/StructuralCausalBandit-v0" in gym.registry
    assert "causalrl/FrontdoorBandit-v0" in gym.registry


def test_register_envs_is_idempotent() -> None:
    """register_envs() called multiple times must not raise or produce warnings."""
    import causalrl

    # Already called at import time; calling again must be silent.
    causalrl.register_envs()
    causalrl.register_envs()
    # Still in registry.
    assert "causalrl/StructuralCausalBandit-v0" in gym.registry


def test_gym_make_structural_causal_bandit() -> None:
    """gymnasium.make should construct a working StructuralCausalBanditEnv."""
    import causalrl  # noqa: F401

    env = gym.make("causalrl/StructuralCausalBandit-v0", n_mc=10)
    obs, info = env.reset()
    assert obs is not None
    assert isinstance(info, dict)
    env.close()


def test_gym_make_frontdoor_bandit() -> None:
    """gymnasium.make should construct a working frontdoor bandit env."""
    import causalrl  # noqa: F401

    env = gym.make("causalrl/FrontdoorBandit-v0", n_mc=10)
    obs, _info = env.reset()
    assert obs is not None
    env.close()


def test_gym_make_bandit_step() -> None:
    """gymnasium.make bandit: reset + step returns the expected tuple shape."""
    import causalrl  # noqa: F401

    env = gym.make("causalrl/StructuralCausalBandit-v0", n_mc=10)
    env.reset(seed=0)
    action = env.action_space.sample()
    _obs, reward, terminated, _truncated, _info = env.step(action)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    env.close()


def test_register_envs_top_level_export() -> None:
    """register_envs must be importable from the top-level package."""
    import causalrl

    assert callable(causalrl.register_envs)


# ---------------------------------------------------------------------------
# Item 4: Vectorized-env smoke test
# ---------------------------------------------------------------------------


def test_make_vec_sync_constructs_and_steps() -> None:
    """gymnasium.make_vec with vectorization_mode='sync' must construct and step."""
    import causalrl  # noqa: F401

    vec_env = gym.make_vec(
        "causalrl/StructuralCausalBandit-v0",
        num_envs=2,
        vectorization_mode="sync",
        n_mc=10,  # forwarded to the env creator via **kwargs
    )
    try:
        obs, _info = vec_env.reset()
        assert obs is not None
        actions = vec_env.action_space.sample()
        _obs2, rewards, _terms, _truncs, _infos = vec_env.step(actions)
        assert rewards.shape == (2,)
    finally:
        vec_env.close()


def test_sync_vector_env_direct_construction() -> None:
    """gymnasium.vector.SyncVectorEnv constructs and runs two bandit envs."""
    import causalrl  # noqa: F401

    vec_env = gym.vector.SyncVectorEnv(
        [lambda: gym.make("causalrl/StructuralCausalBandit-v0", n_mc=10) for _ in range(2)]
    )
    try:
        obs, _info = vec_env.reset()
        assert obs is not None
        actions = vec_env.action_space.sample()
        _obs2, rewards, _terms, _truncs, _infos = vec_env.step(actions)
        assert rewards.shape == (2,)
    finally:
        vec_env.close()
