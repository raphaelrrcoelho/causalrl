"""Plan §8.3 acceptance (a): a PettingZoo ParallelEnv reproduced end-to-end through the adapter.

The adapter is duck-typed (PettingZoo is never imported), so a lightweight ParallelEnv-shaped mock
exercises the full path a real benchmark would take — logging every agent-step into a TrajectoryLog
keyed by ``entity_id`` = agent. numpy; fully local.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from causalrl.interop.pettingzoo import pettingzoo_to_trajectory_log


class MockParallelEnv:
    """A minimal 2-agent, 3-step ParallelEnv: each agent's reward is its own action."""

    def __init__(self) -> None:
        self.possible_agents = ["p0", "p1"]
        self.agents: list[str] = []
        self._t = 0

    def reset(self, seed: int | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        self.agents = list(self.possible_agents)
        self._t = 0
        return {a: np.zeros(1, dtype=float) for a in self.agents}, {}

    def step(
        self, actions: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, float], dict[str, bool], dict[str, bool], dict[str, Any]]:
        self._t += 1
        obs = {a: np.array([float(self._t)]) for a in self.agents}
        rewards = {a: float(actions[a]) for a in self.agents}
        done = self._t >= 3
        terms = {a: done for a in self.agents}
        truncs = {a: False for a in self.agents}
        if done:
            self.agents = []
        return obs, rewards, terms, truncs, {}


def test_adapter_logs_every_agent_step() -> None:
    log = pettingzoo_to_trajectory_log(
        MockParallelEnv(),
        {"p0": lambda _o: 1, "p1": lambda _o: 0},
        n_episodes=2,
        max_steps=10,
        seed=0,
    )
    ents = log.column("entity_id")
    names = log.column("name")
    values = log.column("value")

    assert set(ents.tolist()) == {0, 1}  # entity_id = agent
    assert set(log.column("episode_id").tolist()) == {0, 1}
    assert {"obs", "action", "reward"} <= set(names.tolist())

    reward = names == "reward"
    p0 = values[reward & (ents == 0)]
    p1 = values[reward & (ents == 1)]
    assert len(p0) == 6 and len(p1) == 6  # 2 episodes x 3 steps
    assert bool((p0 == 1.0).all())  # p0 reward = its constant action 1
    assert bool((p1 == 0.0).all())  # p1 reward = its constant action 0


def test_adapter_raises_on_missing_policy() -> None:
    with pytest.raises(KeyError):
        pettingzoo_to_trajectory_log(MockParallelEnv(), {"p0": lambda _o: 1}, n_episodes=1)
