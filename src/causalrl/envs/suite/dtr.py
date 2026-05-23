from typing import Any, ClassVar

import gymnasium as gym
import numpy as np

from causalrl.envs.base import ConfoundedMDP

_TERMINAL = 2


class DTREnv(ConfoundedMDP):
    """Confounded (single-stage) dynamic treatment regime.

    A patient subtype ``Z ~ Bernoulli(0.5)`` is OBSERVED by the agent (the state). An
    unobserved comorbidity ``U ~ Bernoulli(0.5)`` is hidden from the agent but drives the
    logging clinician's treatment choice. The agent picks a treatment ``a in {0, 1}`` and
    receives a Bernoulli reward whose success probability depends on ``(Z, a, U)``.

    state = 0 / 1 : observed subtype Z
    state = 2     : terminal
    Horizon 1 (one treatment decision per episode).

    The reward table is chosen so that:
    - the do-optimal policy is a = Z (matched treatment), with average value 0.75;
    - the confounder U inflates the apparent reward of treatment 1, so a clinician who
      follows U over-prescribes treatment 1. In the resulting confounded logs, treatment 1
      looks best for BOTH subtypes, so a naive offline learner picks a = 1 everywhere and is
      WRONG for subtype Z = 0 (true value 0.55 vs the optimal 0.70) — average value 0.675.

    This is the canonical causal-RL lesson: naively trusting confounded logs yields a
    confidently biased, suboptimal policy.
    """

    n_states = 3
    n_actions = 2
    horizon = 1

    # P(R = 1 | Z, a, U). do-value(Z, a) = mean over U ~ Bernoulli(0.5).
    # do-values: (0,0)=0.70 (0,1)=0.55 (1,0)=0.20 (1,1)=0.80 -> optimal a = Z, value 0.75.
    _REWARD_PROB: ClassVar[dict[tuple[int, int, int], float]] = {
        (0, 0, 0): 0.70, (0, 0, 1): 0.70,
        (0, 1, 0): 0.20, (0, 1, 1): 0.90,
        (1, 0, 0): 0.20, (1, 0, 1): 0.20,
        (1, 1, 0): 0.70, (1, 1, 1): 0.90,
    }
    _BEHAVIOR_FOLLOW_U = 0.85

    metadata: ClassVar[dict[str, list[str]]] = {"render_modes": []}  # type: ignore[misc]  # gymnasium Env.metadata is an instance var in the base

    def __init__(self, seed: int | None = None) -> None:
        super().__init__()
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Dict(
            {"state": gym.spaces.Discrete(3), "t": gym.spaces.Discrete(2)}
        )
        self._rng = np.random.default_rng(seed)
        self._z = 0
        self._u = 0

    def do_value(self, z: int, a: int) -> float:
        """The interventional value E[R | do(a), Z=z], averaged over U ~ Bernoulli(0.5)."""
        return 0.5 * self._REWARD_PROB[(z, a, 0)] + 0.5 * self._REWARD_PROB[(z, a, 1)]

    @property
    def optimal_value(self) -> float:
        """Average value of the do-optimal policy a*(Z) = argmax_a do_value(Z, a)."""
        return 0.5 * sum(max(self.do_value(z, 0), self.do_value(z, 1)) for z in (0, 1))

    def reset(  # type: ignore[override]
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, int], dict[str, Any]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._z = int(self._rng.integers(0, 2))
        self._u = int(self._rng.integers(0, 2))
        return {"state": self._z, "t": 0}, {}

    def step(  # type: ignore[override]
        self, action: int
    ) -> tuple[dict[str, int], float, bool, bool, dict[str, Any]]:
        p = self._REWARD_PROB[(self._z, action, self._u)]
        reward = float(self._rng.random() < p)
        return {"state": _TERMINAL, "t": 1}, reward, True, False, {"u": self._u}

    def behavior_policy(self, observation: dict[str, int]) -> int:
        """Confounded logging: the clinician observes the hidden comorbidity U and prescribes
        it with probability 0.85 (else the opposite), independent of the subtype Z. This ties
        the logged action to U, which also drives the reward — the source of confounding."""
        if self._rng.random() < self._BEHAVIOR_FOLLOW_U:
            return self._u
        return 1 - self._u
