from dataclasses import dataclass
from typing import Any, Protocol


class RolloutEnv(Protocol):
    """Minimal interface required by :func:`generate_logs`."""

    n_states: int
    n_actions: int

    def reset(self, *, seed: int | None = None) -> tuple[Any, Any]: ...

    def step(self, action: int) -> tuple[Any, float, bool, bool, Any]: ...

    def behavior_policy(self, observation: Any) -> int: ...


@dataclass(frozen=True)
class Transition:
    """A single logged transition. `reward` is the terminal return (0 except on done)."""

    state: int
    action: int
    reward: float
    next_state: int
    done: bool


class ConfoundedTrajectoryDataset:
    """Immutable offline log plus empirical behavior statistics per (state, action)."""

    def __init__(self, transitions: list[Transition], n_states: int, n_actions: int) -> None:
        self._transitions = list(transitions)
        self.n_states = n_states
        self.n_actions = n_actions
        # counts[s][a] = number of times action a was taken in state s
        self._counts = [[0 for _ in range(n_actions)] for _ in range(n_states)]
        self._reward_sums = [[0.0 for _ in range(n_actions)] for _ in range(n_states)]
        for tr in self._transitions:
            self._counts[tr.state][tr.action] += 1
            self._reward_sums[tr.state][tr.action] += tr.reward

    def __len__(self) -> int:
        return len(self._transitions)

    @property
    def transitions(self) -> list[Transition]:
        return list(self._transitions)

    def _state_total(self, state: int) -> int:
        return sum(self._counts[state])

    def behavior_propensity(self, state: int, action: int) -> float:
        """Empirical P(action | state). 0.0 if the state was never visited."""
        total = self._state_total(state)
        if total == 0:
            return 0.0
        return self._counts[state][action] / total

    def mean_reward(self, state: int, action: int) -> float:
        """Empirical E[return | state, action]. 0.0 if (state, action) never logged."""
        n = self._counts[state][action]
        if n == 0:
            return 0.0
        return self._reward_sums[state][action] / n


def generate_logs(env: RolloutEnv, n_episodes: int, seed: int) -> ConfoundedTrajectoryDataset:
    """Roll out an env's confounded behavior_policy to build an offline dataset.

    The env must expose ``n_states``, ``n_actions``, ``reset(seed=...)``, ``step(action)``,
    and ``behavior_policy(observation)``. Rewards are recorded as the per-step reward; for
    the finite-horizon envs here the terminal step carries the return.
    """
    transitions: list[Transition] = []
    obs, _ = env.reset(seed=seed)
    for ep in range(n_episodes):
        if ep > 0:
            obs, _ = env.reset()
        done = False
        while not done:
            state = int(obs["state"])
            action = int(env.behavior_policy(obs))
            next_obs, reward, done, _trunc, _info = env.step(action)
            transitions.append(
                Transition(state, action, float(reward), int(next_obs["state"]), bool(done))
            )
            obs = next_obs
    return ConfoundedTrajectoryDataset(transitions, n_states=env.n_states, n_actions=env.n_actions)
