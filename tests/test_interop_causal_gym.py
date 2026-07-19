"""from_causal_gym: the data-level Causal-Gymnasium adapter, exercised with fake Gymnasium envs."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from causalrl.data.dataset import ConfoundedTrajectoryDataset
from causalrl.interop.causal_gym import from_causal_gym


class _FakeConfoundedGym:
    """Minimal Gymnasium-API confounded bandit: a hidden U in {0, 1} exposed in obs/info; the reward
    is 1.0 when the action matches U. A policy that plays ``a = U`` is strongly confounded."""

    n_states = 1
    n_actions = 2

    def __init__(self, seed: int = 0) -> None:
        self._rng = np.random.default_rng(seed)
        self._u = 0

    def reset(self, *, seed: int | None = None) -> tuple[dict[str, int], dict[str, int]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._u = int(self._rng.random() < 0.5)
        return {"state": 0, "u": self._u}, {"u": self._u}

    def step(self, action: int) -> tuple[dict[str, int], float, bool, bool, dict[str, int]]:
        reward = 1.0 if action == self._u else 0.0
        return {"state": 0, "u": self._u}, reward, True, False, {"u": self._u}


def test_builds_a_confounded_dataset_from_rollouts() -> None:
    env = _FakeConfoundedGym(seed=0)
    dataset = from_causal_gym(env, lambda obs: obs["u"], n_episodes=2000, seed=0)
    assert isinstance(dataset, ConfoundedTrajectoryDataset)
    assert len(dataset) == 2000
    assert dataset.n_states == 1
    assert dataset.n_actions == 2
    # Behavior plays a = U, U ~ Bernoulli(0.5), so each action is played about half the time.
    assert 0.4 < dataset.behavior_propensity(0, 0) < 0.6
    # a = U always matches, so every logged reward is 1.0.
    assert dataset.mean_reward(0, 0) == 1.0
    assert dataset.mean_reward(0, 1) == 1.0


def test_infers_dims_and_supports_int_observations() -> None:
    class _IntEnv:
        def reset(self, *, seed: int | None = None) -> tuple[int, dict[str, Any]]:
            return 0, {}

        def step(self, action: int) -> tuple[int, float, bool, bool, dict[str, Any]]:
            return 0, float(action), True, False, {}

    dataset = from_causal_gym(_IntEnv(), lambda obs: 1, n_episodes=5, seed=1)
    assert dataset.n_states == 1  # only state 0 seen -> max_state + 1
    assert dataset.n_actions == 2  # action 1 seen -> max_action + 1
    assert dataset.mean_reward(0, 1) == 1.0


def test_opaque_observation_requires_a_state_fn() -> None:
    class _ArrayEnv:
        def reset(self, *, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
            return np.array([0.1, 0.2, 0.3]), {}

        def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
            return np.array([0.1, 0.2, 0.3]), 0.0, True, False, {}

    with pytest.raises(ValueError):
        from_causal_gym(_ArrayEnv(), lambda obs: 0, n_episodes=1)
    # An explicit state_fn resolves the mapping.
    dataset = from_causal_gym(_ArrayEnv(), lambda obs: 0, n_episodes=3, state_fn=lambda obs: 0)
    assert len(dataset) == 3
