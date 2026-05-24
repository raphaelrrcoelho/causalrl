from typing import Any, ClassVar

import gymnasium as gym
import numpy as np

from causalrl.envs.base import ConfoundedMDP


class SequentialMABUCEnv(ConfoundedMDP):
    """A horizon-H sequential bandit with an observed intuition signal.

    Each step draws D, B ~ Bernoulli(0.5); the intuition I = D xor B is OBSERVED (it is the
    state, step*2 + I); the lucky arm equals I; reward is 0.75 if the chosen arm matches the
    lucky arm else 0.25 (Bernoulli). The behavior policy plays I.

    SCOPE (v0.2) — honesty note: because the intuition fully reveals the optimal action and
    is observed, this env is NOT confounded in the hidden-U sense (conditional on the state,
    E[Y|s,a] = E[Y|do(a),s]); the behavior policy simply leaves the off-arm unexplored
    (vacuous bound) rather than biased. It is used to exercise the deep agent's online
    learning + bound-clamping, not to demonstrate a confounding effect. A genuinely
    confounded sequential variant (a hidden component not recoverable from the state) is a
    v0.3 refinement (see the v0.2 spec backlog).
    """

    n_actions = 2

    metadata: ClassVar[dict[str, list[str]]] = {"render_modes": []}  # type: ignore[misc]  # gymnasium Env.metadata is an instance var in the base

    def __init__(self, horizon: int = 3, seed: int | None = None) -> None:
        super().__init__()
        self.horizon = horizon
        self.n_states = horizon * 2 + 1
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Dict(
            {"state": gym.spaces.Discrete(self.n_states), "t": gym.spaces.Discrete(horizon + 1)}
        )
        self._rng = np.random.default_rng(seed)
        self._terminal = horizon * 2
        self._t = 0
        self._intuition = 0

    def _draw_intuition(self) -> int:
        d = int(self._rng.integers(0, 2))
        b = int(self._rng.integers(0, 2))
        return d ^ b

    def reset(  # type: ignore[override]
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, int], dict[str, Any]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._t = 0
        self._intuition = self._draw_intuition()
        return {"state": self._intuition, "t": 0}, {}

    def step(  # type: ignore[override]
        self, action: int
    ) -> tuple[dict[str, int], float, bool, bool, dict[str, Any]]:
        lucky = self._intuition
        p = 0.75 if action == lucky else 0.25
        reward = float(self._rng.random() < p)
        self._t += 1
        if self._t >= self.horizon:
            return {"state": self._terminal, "t": self._t}, reward, True, False, {}
        self._intuition = self._draw_intuition()
        state = self._t * 2 + self._intuition
        return {"state": state, "t": self._t}, reward, False, False, {}

    def behavior_policy(self, observation: dict[str, int]) -> int:
        return int(observation["state"] % 2)  # play the intuition
