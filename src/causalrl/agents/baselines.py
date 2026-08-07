import math
from typing import Any

import numpy as np

from causalrl.agents.base import Agent, BatchAgent
from causalrl.data.dataset import ConfoundedTrajectoryDataset


class OnlineOnlyUCB(Agent):
    """UCB1 per state, ignoring any offline data. The 'learn from scratch online' baseline."""

    def __init__(self, n_states: int, n_actions: int, seed: int | None = None) -> None:
        self.n_states = n_states
        self.n_actions = n_actions
        self._counts = np.zeros((n_states, n_actions))
        self._sums = np.zeros((n_states, n_actions))
        self._t = 1
        self._rng = np.random.default_rng(seed)

    def ingest_offline(self, dataset: ConfoundedTrajectoryDataset) -> None:
        """No-op: this baseline ignores offline data by design."""

    def act(self, observation: dict[str, Any]) -> int:
        s = int(observation["state"])
        untried = [a for a in range(self.n_actions) if self._counts[s, a] == 0]
        if untried:
            return int(self._rng.choice(untried))
        means = self._sums[s] / self._counts[s]
        bonus = np.sqrt(2.0 * math.log(self._t) / self._counts[s])
        return int(np.argmax(np.asarray(means + bonus, dtype=np.float64)))

    def update(self, observation: dict[str, Any], action: int, reward: float) -> None:
        s = int(observation["state"])
        self._counts[s, action] += 1.0
        self._sums[s, action] += reward
        self._t += 1


class NaiveOffline(BatchAgent):
    """Fits E[R|s,a] from the logs as if unconfounded and acts greedily. Provably biased
    under confounding — the cautionary baseline."""

    def __init__(self, n_states: int, n_actions: int) -> None:
        self.n_states = n_states
        self.n_actions = n_actions
        self._mean = np.zeros((n_states, n_actions))

    def ingest_offline(self, dataset: ConfoundedTrajectoryDataset) -> None:
        for s in range(self.n_states):
            for a in range(self.n_actions):
                self._mean[s, a] = dataset.mean_reward(s, a)

    def act(self, observation: dict[str, Any]) -> int:
        s = int(observation["state"])
        return int(np.argmax(np.asarray(self._mean[s], dtype=np.float64)))
