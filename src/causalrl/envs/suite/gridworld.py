from typing import Any, ClassVar

import gymnasium as gym
import numpy as np

from causalrl.envs.base import ConfoundedMDP


class ConfoundedGridworld(ConfoundedMDP):
    """A size x size gridworld with a hidden confounder that biases the logging policy.

    State = row * size + col (0-indexed), plus a terminal state = size*size. Start at
    top-left (state 0), goal at bottom-right. Actions: 0=up, 1=down, 2=left, 3=right.
    Reaching the goal yields terminal return 1.0; hitting the per-episode 'hazard' cell
    (chosen by the hidden confounder U) yields 0.0 and ends the episode. Horizon = 2*size.
    The behavior policy, knowing U, avoids the hazard — so the logs are confounded.
    """

    n_actions = 4

    metadata: ClassVar[dict[str, list[str]]] = {"render_modes": []}  # type: ignore[misc]  # gymnasium Env.metadata is an instance var in the base

    def __init__(self, size: int = 3, seed: int | None = None) -> None:
        super().__init__()
        self.size = size
        self.n_states = size * size + 1
        self.horizon = 2 * size
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Discrete(self.n_states),
                "t": gym.spaces.Discrete(self.horizon + 1),
            }
        )
        self._rng = np.random.default_rng(seed)
        self._terminal = size * size
        self._row = 0
        self._col = 0
        self._t = 0
        self._u = 0

    def _state_index(self) -> int:
        return self._row * self.size + self._col

    def reset(  # type: ignore[override]
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, int], dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._row = 0
        self._col = 0
        self._t = 0
        self._u = int(self._rng.integers(1, self.size * self.size - 1))
        return {"state": 0, "t": 0}, {}

    def step(  # type: ignore[override]
        self, action: int
    ) -> tuple[dict[str, int], float, bool, bool, dict[str, Any]]:
        if action == 0:
            self._row = max(0, self._row - 1)
        elif action == 1:
            self._row = min(self.size - 1, self._row + 1)
        elif action == 2:
            self._col = max(0, self._col - 1)
        elif action == 3:
            self._col = min(self.size - 1, self._col + 1)
        self._t += 1
        idx = self._state_index()
        if idx == self._u:
            return {"state": self._terminal, "t": self._t}, 0.0, True, False, {"hazard": True}
        if self._row == self.size - 1 and self._col == self.size - 1:
            return {"state": self._terminal, "t": self._t}, 1.0, True, False, {}
        if self._t >= self.horizon:
            return {"state": self._terminal, "t": self._t}, 0.0, True, False, {}
        return {"state": idx, "t": self._t}, 0.0, False, False, {}

    def behavior_policy(self, observation: dict[str, int]) -> int:
        """Knows the hazard U: biases toward down/right but avoids stepping onto the hazard."""
        candidates = [1, 3]  # down, right (toward goal)
        safe: list[int] = []
        for a in candidates:
            r, c = self._row, self._col
            if a == 1:
                r = min(self.size - 1, r + 1)
            else:
                c = min(self.size - 1, c + 1)
            if r * self.size + c != self._u:
                safe.append(a)
        choices = safe or candidates
        return int(self._rng.choice(choices))
