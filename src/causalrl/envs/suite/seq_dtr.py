from typing import Any, ClassVar

import gymnasium as gym
import numpy as np

from causalrl.envs.base import ConfoundedMDP


class SequentialDTREnv(ConfoundedMDP):
    """Multi-stage confounded dynamic treatment regime — genuinely confounded AND sequential.

    Per episode a persistent hidden comorbidity ``U ~ Bernoulli(0.5)`` is drawn. At each of
    ``H`` stages the agent observes a subtype ``Z`` (the state, encoded ``stage*2 + Z`` so the
    stage is baked into the index; terminal = ``2H``), picks a treatment ``a in {0,1}`` and
    receives ``reward ~ Bernoulli(p(Z, a, U))``. The subtype then transitions deterministically
    by ``Z' = a`` — a function of the chosen action and observed state only, NOT of ``U`` — so
    the transition dynamics are unconfounded and DOVI's value-iteration bound stays certified.

    The reward table is tuned so the immediately-greedy action differs from the lookahead-
    optimal one (a genuine foresight gap), and ``U`` adds +0.2 to every cell when ``U = 1``
    while the logging clinician plays ``a = U`` (prob 0.85): so naive-offline over-prescribes
    treatment 1 in the logs AND, being myopic, also misorders the sequential trade-off.
    """

    n_actions = 2

    # P(R = 1 | Z, a, U). dv(Z,a) = mean over U: dv(0,0)=.5 dv(0,1)=.65 dv(1,0)=.3 dv(1,1)=.35.
    _REWARD_PROB: ClassVar[dict[tuple[int, int, int], float]] = {
        (0, 0, 0): 0.30,
        (0, 0, 1): 0.70,
        (0, 1, 0): 0.45,
        (0, 1, 1): 0.85,
        (1, 0, 0): 0.10,
        (1, 0, 1): 0.50,
        (1, 1, 0): 0.15,
        (1, 1, 1): 0.55,
    }
    _BEHAVIOR_FOLLOW_U = 0.85

    metadata: ClassVar[dict[str, list[str]]] = {"render_modes": []}  # type: ignore[misc]  # gymnasium Env.metadata is an instance var in the base

    def __init__(self, horizon: int = 2, seed: int | None = None) -> None:
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
        self._z = 0
        self._u = 0

    def dv(self, z: int, a: int) -> float:
        """Per-stage interventional value E[R | do(a), Z=z], averaged over U ~ Bernoulli(0.5)."""
        return 0.5 * self._REWARD_PROB[(z, a, 0)] + 0.5 * self._REWARD_PROB[(z, a, 1)]

    def do_value(self, z: int, a: int, stage: int) -> float:
        """Interventional value of taking ``a`` at ``stage`` in subtype ``z``, then optimal."""
        if stage >= self.horizon - 1:
            return self.dv(z, a)
        z_next = a  # transition Z' = a
        v_next = max(self.do_value(z_next, a2, stage + 1) for a2 in (0, 1))
        return self.dv(z, a) + v_next

    @property
    def optimal_value(self) -> float:
        """Expected return of the do-optimal policy from a random initial subtype Z."""
        return 0.5 * sum(max(self.do_value(z, a, 0) for a in (0, 1)) for z in (0, 1))

    def reset(  # type: ignore[override]
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, int], dict[str, Any]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._t = 0
        self._u = int(self._rng.integers(0, 2))
        self._z = int(self._rng.integers(0, 2))
        return {"state": self._z, "t": 0}, {}

    def step(  # type: ignore[override]
        self, action: int
    ) -> tuple[dict[str, int], float, bool, bool, dict[str, Any]]:
        p = self._REWARD_PROB[(self._z, action, self._u)]
        reward = float(self._rng.random() < p)
        self._t += 1
        if self._t >= self.horizon:
            return {"state": self._terminal, "t": self._t}, reward, True, False, {"u": self._u}
        self._z = action  # transition Z' = a (U-independent)
        state = self._t * 2 + self._z
        return {"state": state, "t": self._t}, reward, False, False, {}

    def behavior_policy(self, observation: dict[str, int]) -> int:
        """Confounded logging: the clinician follows the hidden comorbidity U (plays a = U
        with prob 0.85), tying the logged action to U, which also drives the reward."""
        if self._rng.random() < self._BEHAVIOR_FOLLOW_U:
            return self._u
        return 1 - self._u
