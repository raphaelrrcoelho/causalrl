"""Oracle confounded contextual bandit for the causal-MBRL M0 kill-gate.

A 2-context, 2-action bandit with a hidden confounder ``U`` on a backdoor path ``A <- U -> Y``.
The optimal action is ``c_opt(c)`` (``= c``, or ``1 - c`` under ``shift``). The behavior policy is
confounded (its action tracks ``U`` with probability ``gamma``), so the naive logged mean
over-values the ``U``-aligned action. ``true_policy_value`` is the exact interventional value,
marginalizing ``U`` — the ground truth for the kill-gate.

The SCM is carried on ``self.scm`` for the later (M1) discovery loop; the M0 experiment uses the
env's ``behavior_policy`` / ``step`` / ``true_policy_value`` directly (pure NumPy), not the SCM.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
import torch
from torch.distributions import Bernoulli, Distribution, Uniform

from causalrl.envs.base import CausalEnv
from causalrl.scm.graph import CausalGraph
from causalrl.scm.mechanisms import FunctionalMechanism, Mechanism
from causalrl.scm.scm import StructuralCausalModel


def _build_scm(shift: bool) -> StructuralCausalModel:
    """Confounded-bandit SCM: ``A <- U`` (confounded logging) and ``C, A, U -> Y`` (reward).

    Structurally honest for M1 discovery: the backdoor path ``A <- U -> Y`` is the confounding,
    and ``C -> Y`` is the (observed) causal feature. ``A``'s in-SCM mechanism is the deterministic
    confounded-logging view (``A = U``); the env's ``behavior_policy`` adds the ``gamma`` mixing.
    """

    def _reward(pa: dict[str, torch.Tensor], u: torch.Tensor) -> torch.Tensor:
        c_opt = (1.0 - pa["C"]) if shift else pa["C"]
        p = (0.05 + 0.20 * (pa["A"] == c_opt).float() + 0.60 * pa["U"]).clamp(0.0, 1.0)
        return (u < p).float()

    graph = CausalGraph(directed_edges=[("U", "A"), ("C", "Y"), ("A", "Y"), ("U", "Y")])
    mechanisms: dict[str, Mechanism] = {
        "C": FunctionalMechanism([], lambda pa, u: u),
        "U": FunctionalMechanism([], lambda pa, u: u),
        "A": FunctionalMechanism(["U"], lambda pa, u: pa["U"]),
        "Y": FunctionalMechanism(["C", "A", "U"], _reward),
    }
    exogenous: dict[str, Distribution] = {
        "C": Bernoulli(0.5),
        "U": Bernoulli(0.5),
        "A": Uniform(0.0, 1.0),
        "Y": Uniform(0.0, 1.0),
    }
    return StructuralCausalModel(graph, mechanisms, exogenous)


class ConfoundedContextualBandit(CausalEnv):
    """A 2-context, 2-action bandit with a hidden confounder ``U`` on a backdoor ``A <- U -> Y``."""

    metadata: ClassVar[dict[str, list[str]]] = {"render_modes": []}  # type: ignore[misc]
    n_states: int = 2
    n_actions: int = 2

    def __init__(self, gamma: float = 0.9, shift: bool = False, *, seed: int | None = None) -> None:
        super().__init__(_build_scm(shift))
        self.gamma = float(gamma)
        self.shift = bool(shift)
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Dict({"state": gym.spaces.Discrete(2)})
        self._rng = np.random.default_rng(seed)
        self._c = 0
        self._u = 0

    def reset(  # type: ignore[override]
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._c = int(self._rng.integers(0, 2))
        self._u = int(self._rng.integers(0, 2))
        return {"state": self._c}, {}

    def behavior_policy(self, observation: dict[str, Any]) -> int:
        """Confounded logging policy: play ``a = U`` with prob ``gamma``, else uniform."""
        if self._rng.random() < self.gamma:
            return self._u
        return int(self._rng.integers(0, 2))

    def _reward_prob(self, c: int, a: int, u: int) -> float:
        c_opt = (1 - c) if self.shift else c
        return min(1.0, max(0.0, 0.05 + 0.20 * float(a == c_opt) + 0.60 * float(u)))

    def step(  # type: ignore[override]
        self, action: int
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        p = self._reward_prob(self._c, int(action), self._u)
        reward = float(self._rng.random() < p)
        return {"state": self._c}, reward, True, False, {"u": self._u}

    def true_policy_value(self, policy: Sequence[int]) -> float:
        """Exact interventional value of a per-context deterministic ``policy`` (marginalizing U)."""
        vals = [
            0.5 * (self._reward_prob(c, int(policy[c]), 0) + self._reward_prob(c, int(policy[c]), 1))
            for c in range(self.n_states)
        ]
        return float(np.mean(vals))


def make_confounded_context_env(
    gamma: float = 0.9, shift: bool = False, seed: int | None = None
) -> ConfoundedContextualBandit:
    """Factory for the M0 oracle confounded contextual bandit."""
    return ConfoundedContextualBandit(gamma=gamma, shift=shift, seed=seed)
