from collections.abc import Callable
from typing import Any

import pytest
from gymnasium.utils.env_checker import check_env

from causalrl.agents.base import Agent
from causalrl.data.dataset import generate_logs
from causalrl.envs.suite.dtr import DTREnv
from causalrl.envs.suite.gridworld import ConfoundedGridworld
from causalrl.envs.suite.mabuc import MABUCEnv
from causalrl.envs.suite.scbandit import make_confounded_chain_env
from causalrl.envs.suite.seq_dtr import SequentialDTREnv
from causalrl.envs.suite.seq_mabuc import SequentialMABUCEnv
from causalrl.eval.harness import run_episodes


@pytest.mark.parametrize(
    "factory",
    [
        DTREnv,
        SequentialDTREnv,
        SequentialMABUCEnv,
        ConfoundedGridworld,
        MABUCEnv,
        lambda: make_confounded_chain_env(n_mc=10),
    ],
)
def test_public_environments_pass_gymnasium_checker(factory: Callable[[], Any]) -> None:
    check_env(factory(), skip_render_check=True)


class _TruncatingEnv:
    n_states = 2
    n_actions = 1

    def __init__(self) -> None:
        self.calls = 0

    def reset(self, *, seed: int | None = None) -> tuple[dict[str, int], dict[str, Any]]:
        return {"state": 0}, {}

    def behavior_policy(self, observation: dict[str, int]) -> int:
        return 0

    def step(self, action: int) -> tuple[dict[str, int], float, bool, bool, dict[str, Any]]:
        self.calls += 1
        if self.calls > 1:
            raise AssertionError("environment stepped after truncation")
        return {"state": 1}, 0.0, False, True, {}


class _RecordingAgent(Agent):
    def __init__(self) -> None:
        self.transitions: list[tuple[int, int, int, bool]] = []

    def act(self, observation: dict[str, int]) -> int:
        return 0

    def update(self, observation: dict[str, int], action: int, reward: float) -> None:
        return None

    def observe_transition(self, state: int, action: int, next_state: int, done: bool) -> None:
        self.transitions.append((state, action, next_state, done))


def test_generate_logs_ends_an_episode_on_truncation() -> None:
    dataset = generate_logs(_TruncatingEnv(), n_episodes=1, seed=0)
    assert len(dataset) == 1
    assert dataset.transitions[0].done is True


def test_run_episodes_ends_an_episode_on_truncation() -> None:
    agent = _RecordingAgent()
    returns = run_episodes(agent, _TruncatingEnv(), n_episodes=1, seed=0)
    assert returns == [0.0]
    assert agent.transitions == [(0, 0, 1, True)]
