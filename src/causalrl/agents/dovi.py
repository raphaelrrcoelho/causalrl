import math
from typing import Any

import numpy as np

from causalrl.agents.base import Agent
from causalrl.agents.primitives import bounds_table
from causalrl.data.dataset import ConfoundedTrajectoryDataset


class DOVI(Agent):
    """Deconfounded optimistic learning (after Wang et al. 2021), tabular form.

    The Manski UPPER causal bound on each (state, action) caps an optimistic estimate, so
    online optimism never exceeds the causally-valid ceiling. Online, the estimate is the
    running mean plus a UCB bonus, clamped to that ceiling — deconfounding the optimism that
    pure UCB would otherwise overstate.

    SCOPE (v0.2): the ceiling is built from the *immediate per-step reward* (Manski bounds
    over ``dataset.mean_reward``), so the deconfounding guarantee is exact for **contextual /
    horizon-1** problems (e.g. the DTR). On multi-step environments this agent performs
    bound-capped online learning per state but does NOT yet bootstrap next-state values — a
    full horizon-indexed Bellman backup with bounds on the *return* Q is a v0.3 refinement
    (see the v0.2 spec backlog). The ``horizon`` argument is accepted for that future use.
    """

    def __init__(
        self, n_states: int, n_actions: int, horizon: int, seed: int | None = None
    ) -> None:
        self.n_states = n_states
        self.n_actions = n_actions
        self.horizon = horizon
        self._counts = np.zeros((n_states, n_actions))
        self._sums = np.zeros((n_states, n_actions))
        self._t = 1
        self._rng = np.random.default_rng(seed)
        self._ceiling = np.ones((n_states, n_actions))

    def ingest_offline(self, dataset: ConfoundedTrajectoryDataset) -> None:
        for (s, a), (_lo, hi) in bounds_table(dataset).items():
            self._ceiling[s, a] = hi

    def optimistic_q(self, state: int, action: int) -> float:
        n = self._counts[state, action]
        mean = self._sums[state, action] / n if n > 0 else 0.0
        bonus = math.sqrt(2.0 * math.log(self._t) / n) if n > 0 else 1.0
        return float(min(mean + bonus, self._ceiling[state, action]))

    def act(self, observation: dict[str, Any]) -> int:
        s = int(observation["state"])
        scores = np.asarray(
            [self.optimistic_q(s, a) for a in range(self.n_actions)], dtype=np.float64
        )
        best = float(np.max(scores))
        winners = [a for a in range(self.n_actions) if scores[a] >= best - 1e-12]
        return int(self._rng.choice(winners))

    def update(self, observation: dict[str, Any], action: int, reward: float) -> None:
        s = int(observation["state"])
        self._counts[s, action] += 1.0
        self._sums[s, action] += reward
        self._t += 1
