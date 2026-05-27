from typing import Any, ClassVar

import gymnasium as gym
import numpy as np

from causalrl.envs.base import ConfoundedMDP


class SequentialMABUCEnv(ConfoundedMDP):
    """Horizon-H contextual bandit with an unobserved confounder (a genuine MABUC).

    Each step is an independent confounded contextual-bandit decision: an observed context
    ``C ~ Bernoulli(0.5)`` (the state, ``stage*2 + C``; terminal = ``2H``) and a HIDDEN
    ``U ~ Bernoulli(0.5)`` are drawn, then ``reward ~ Bernoulli(p(C, a, U))``. The logging
    policy follows the hidden ``U`` (plays ``a = U`` with prob 0.85), so conditional on the
    observed context the logged action is tied to ``U`` — which also drives the reward.
    Naive-offline therefore over-prescribes treatment 1 in every context (its apparent reward
    is inflated by ``U``), exactly as in ``DTREnv``, but here repeated across the horizon.

    There are no state transitions: this is the bandit member of the suite. The
    sequential-credit-assignment lesson lives in ``SequentialDTREnv``.
    """

    n_actions = 2

    # P(R = 1 | C, a, U). do-optimal a = C (value 0.75/step); U inflates treatment 1.
    _REWARD_PROB: ClassVar[dict[tuple[int, int, int], float]] = {
        (0, 0, 0): 0.70,
        (0, 0, 1): 0.70,
        (0, 1, 0): 0.20,
        (0, 1, 1): 0.90,
        (1, 0, 0): 0.20,
        (1, 0, 1): 0.20,
        (1, 1, 0): 0.70,
        (1, 1, 1): 0.90,
    }
    _BEHAVIOR_FOLLOW_U = 0.85

    metadata: ClassVar[dict[str, list[str]]] = {"render_modes": []}  # type: ignore[misc]  # gymnasium Env.metadata is an instance var in the base

    def __init__(self, horizon: int = 3, seed: int | None = None) -> None:
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}")
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
        self._c = 0
        self._u = 0

    def do_value(self, c: int, a: int) -> float:
        """Per-step interventional value E[R | do(a), C=c], averaged over U ~ Bernoulli(0.5)."""
        return 0.5 * self._REWARD_PROB[(c, a, 0)] + 0.5 * self._REWARD_PROB[(c, a, 1)]

    @property
    def optimal_value(self) -> float:
        """Per-step value of the do-optimal policy a*(C) = argmax_a do_value(C, a)."""
        return 0.5 * sum(max(self.do_value(c, 0), self.do_value(c, 1)) for c in (0, 1))

    def _draw(self) -> None:
        self._c = int(self._rng.integers(0, 2))
        self._u = int(self._rng.integers(0, 2))

    def reset(  # type: ignore[override]
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, int], dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._t = 0
        self._draw()
        return {"state": self._c, "t": 0}, {}

    def step(  # type: ignore[override]
        self, action: int
    ) -> tuple[dict[str, int], float, bool, bool, dict[str, Any]]:
        p = self._REWARD_PROB[(self._c, action, self._u)]
        reward = float(self._rng.random() < p)
        self._t += 1
        if self._t >= self.horizon:
            return {"state": self._terminal, "t": self._t}, reward, True, False, {}
        self._draw()
        state = self._t * 2 + self._c
        return {"state": state, "t": self._t}, reward, False, False, {}

    def behavior_policy(self, observation: dict[str, int]) -> int:
        """Confounded logging: follows the hidden U (plays a = U with prob 0.85)."""
        if self._rng.random() < self._BEHAVIOR_FOLLOW_U:
            return self._u
        return 1 - self._u
