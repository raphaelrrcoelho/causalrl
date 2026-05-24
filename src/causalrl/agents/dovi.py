import math
from typing import Any

import numpy as np

from causalrl.agents.base import Agent
from causalrl.agents.primitives import bounds_table
from causalrl.data.dataset import ConfoundedTrajectoryDataset


class DOVI(Agent):
    """Deconfounded Optimistic Value Iteration (Wang, Yang, Wang, Bareinboim 2021), tabular.

    Finite-horizon backward induction (LSVI form) whose optimism is capped by the Manski
    upper causal bound on each (state, action)'s immediate reward, deconfounding the value
    estimate. With horizon ``H``::

        r̃(s,a)   = min( mean_online(s,a) + ucb_bonus(s,a),  hi_Manski(s,a) )
        Q_h(s,a) = r̃(s,a) + Σ_s' P̂(s'|s,a) · V_{h+1}(s')
        V_h(s)   = max_a Q_h(s,a),    V_{H+1} ≡ 0,    h = H, H-1, …, 1

    ``P̂`` is the empirical next-state distribution pooled from offline logs and online steps;
    transitions that ended an episode (``done``) carry zero future value (the absorbing
    terminal). At ``H = 1`` the backup reduces exactly to v0.2's immediate Manski ceiling, so
    the horizon-1 DTR result is preserved.

    Bound validity: the backup is a certified optimistic bound on the return only when the
    dynamics do not depend on the hidden confounder (true of ``SequentialDTREnv``, whose
    transitions are U-independent). On ``ConfoundedGridworld`` the behavior policy steers
    around the hidden hazard, so ``P̂`` is confounded and the backup is heuristic value
    propagation, not a certified bound.
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        horizon: int,
        seed: int | None = None,
        *,
        reward_max: float = 1.0,
    ) -> None:
        self.n_states = n_states
        self.n_actions = n_actions
        self.horizon = horizon
        self.reward_max = reward_max
        self._counts = np.zeros((n_states, n_actions))
        self._sums = np.zeros((n_states, n_actions))
        # Transition model: non-terminal next-state counts, and total transitions seen.
        self._trans = np.zeros((n_states, n_actions, n_states))
        self._trans_n = np.zeros((n_states, n_actions))
        self._t = 1
        self._rng = np.random.default_rng(seed)
        self._ceiling = np.ones((n_states, n_actions))  # Manski upper bound on immediate reward
        self._q: np.ndarray | None = None  # cached plan, shape (H+1, S, A), indexed h = 1..H

    def ingest_offline(self, dataset: ConfoundedTrajectoryDataset) -> None:
        for (s, a), (_lo, hi) in bounds_table(dataset).items():
            self._ceiling[s, a] = hi
        for tr in dataset.transitions:
            self._trans_n[tr.state, tr.action] += 1.0
            if not tr.done:
                self._trans[tr.state, tr.action, tr.next_state] += 1.0
        self._q = None

    def _r_tilde_table(self) -> np.ndarray:
        n = self._counts
        safe_n = np.maximum(n, 1.0)
        mean = np.where(n > 0, self._sums / safe_n, 0.0)
        bonus = np.where(
            n > 0,
            self.reward_max * np.sqrt(2.0 * math.log(self._t) / safe_n),
            self.reward_max,
        )
        return np.minimum(mean + bonus, self._ceiling)

    def _plan(self) -> np.ndarray:
        h_max, n_states, n_actions = self.horizon, self.n_states, self.n_actions
        rtil = self._r_tilde_table()
        denom = np.maximum(self._trans_n, 1.0)  # (S, A)
        q = np.zeros((h_max + 1, n_states, n_actions))
        v_next = np.zeros(n_states)  # V_{H+1}
        for h in range(h_max, 0, -1):
            # boot[s,a] = (Σ_s' trans[s,a,s'] · V_next[s']) / total transitions seen at (s,a)
            boot = (self._trans @ v_next) / denom
            q[h] = rtil + boot
            v_next = q[h].max(axis=1)
        return q

    def optimistic_q(self, state: int, action: int, h: int = 1) -> float:
        if self._q is None:
            self._q = self._plan()
        return float(self._q[h, state, action])

    def act(self, observation: dict[str, Any]) -> int:
        s = int(observation["state"])
        h = min(int(observation.get("t", 0)) + 1, self.horizon)
        if self._q is None:
            self._q = self._plan()
        scores = self._q[h, s]
        best = float(scores.max())
        winners = [a for a in range(self.n_actions) if scores[a] >= best - 1e-12]
        return int(self._rng.choice(winners))

    def update(self, observation: dict[str, Any], action: int, reward: float) -> None:
        s = int(observation["state"])
        self._counts[s, action] += 1.0
        self._sums[s, action] += reward
        self._t += 1
        self._q = None

    def observe_transition(self, state: int, action: int, next_state: int, done: bool) -> None:
        self._trans_n[state, action] += 1.0
        if not done:
            self._trans[state, action, next_state] += 1.0
        self._q = None
