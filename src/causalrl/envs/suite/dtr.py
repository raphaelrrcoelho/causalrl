from typing import Any, ClassVar

import gymnasium as gym
import numpy as np

from causalrl.envs.base import ConfoundedMDP

_TERMINAL = 4


class DTREnv(ConfoundedMDP):
    """2-stage confounded dynamic treatment regime (see plan Task A3 for the formalization).

    state = 0           : stage 0, context 0
    state = 2 + c1      : stage 1, context c1 = a0 XOR U   (states 2, 3)
    state = 4           : terminal
    Terminal return R = 1.0 if a1 == U else 0.0.
    """

    n_states = 5
    n_actions = 2
    horizon = 2

    metadata: ClassVar[dict[str, list[str]]] = {"render_modes": []}  # type: ignore[misc]  # gymnasium Env.metadata is an instance var in the base

    def __init__(self, seed: int | None = None) -> None:
        super().__init__()
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Dict(
            {"state": gym.spaces.Discrete(5), "t": gym.spaces.Discrete(3)}
        )
        self._rng = np.random.default_rng(seed)
        self._u = 0
        self._a0 = 0
        self._t = 0
        self._state = 0

    def reset(  # type: ignore[override]
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, int], dict[str, Any]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._u = int(self._rng.integers(0, 2))
        self._a0 = 0
        self._t = 0
        self._state = 0
        return {"state": 0, "t": 0}, {}

    def step(  # type: ignore[override]
        self, action: int
    ) -> tuple[dict[str, int], float, bool, bool, dict[str, Any]]:
        if self._t == 0:
            self._a0 = action
            c1 = action ^ self._u
            self._state = 2 + c1
            self._t = 1
            return {"state": self._state, "t": 1}, 0.0, False, False, {}
        reward = 1.0 if action == self._u else 0.0
        self._state = _TERMINAL
        self._t = 2
        return {"state": _TERMINAL, "t": 2}, reward, True, False, {"u": self._u}

    def behavior_policy(self, observation: dict[str, int]) -> int:
        """Confounded logging policy.

        Stage 0: clinicians observe the hidden severity U and prescribe it with prob 0.9
        (else uniform). This creates confounding in the offline log.

        Stage 1: clinicians observe only the stage-1 context (state) and pick uniformly,
        so stage-1 mean rewards are driven purely by the U-induced selection bias from
        stage 0 — making it easy to verify confounding in offline data.
        """
        if observation["t"] == 0:
            # Confounded at stage 0: correlate action with hidden U
            if self._rng.random() < 0.9:
                return self._u
            return int(self._rng.integers(0, 2))
        # Stage 1: uniform random — stage-1 rewards are confounded by stage-0 selection
        return int(self._rng.integers(0, 2))
