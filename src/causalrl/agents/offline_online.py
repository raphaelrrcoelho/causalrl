import math
from typing import Any

import numpy as np

from causalrl.agents.base import Agent
from causalrl.agents.primitives import non_dominated_actions
from causalrl.data.dataset import ConfoundedTrajectoryDataset


class UCDTR(Agent):
    """Causal offline-to-online learner (Zhang & Bareinboim 2019).

    Offline: prune each state's action set to the non-dominated actions implied by the
    Manski causal bounds (a safety mechanism; with natural bounds it is a no-op). Online:
    run UCB1 over only the surviving actions. Unlike NaiveOffline, it never trusts the
    confounded offline point estimates, so it corrects to the true optimum online.
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        seed: int | None = None,
        *,
        require_identified: bool = False,
    ) -> None:
        self.n_states = n_states
        self.n_actions = n_actions
        self._counts = np.zeros((n_states, n_actions))
        self._sums = np.zeros((n_states, n_actions))
        self._t = 1
        self._rng = np.random.default_rng(seed)
        self._allowed: dict[int, list[int]] = {s: list(range(n_actions)) for s in range(n_states)}
        self._require_identified = require_identified

    def ingest_offline(self, dataset: ConfoundedTrajectoryDataset) -> None:
        if self._require_identified:
            from causalrl.identification.bounds import causal_q_bounds

            for s in range(self.n_states):
                for a in range(self.n_actions):
                    causal_q_bounds(dataset, s, a, require_identified=True)
        for s in range(self.n_states):
            self._allowed[s] = non_dominated_actions(dataset, s) or list(range(self.n_actions))

    def act(self, observation: dict[str, Any]) -> int:
        s = int(observation["state"])
        allowed = self._allowed[s]
        untried = [a for a in allowed if self._counts[s, a] == 0]
        if untried:
            return int(self._rng.choice(untried))
        means = self._sums[s] / np.maximum(self._counts[s], 1.0)
        bonus = np.sqrt(2.0 * math.log(self._t) / np.maximum(self._counts[s], 1.0))
        scores = np.asarray(means + bonus, dtype=np.float64)
        best = max(allowed, key=lambda a: float(scores[a]))
        return int(best)

    def update(self, observation: dict[str, Any], action: int, reward: float) -> None:
        s = int(observation["state"])
        self._counts[s, action] += 1.0
        self._sums[s, action] += reward
        self._t += 1
