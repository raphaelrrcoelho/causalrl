"""Confounded one-step imitation environment (taxonomy Task 6).

A hidden-but-observed confounder ``W ~ Bernoulli(0.5)`` drives the lucky action; the reward is
``Bernoulli(0.9)`` if the played action equals ``W`` else ``Bernoulli(0.1)``. The expert plays
``A = W`` (reward ~0.9). A causal imitator conditioning on ``W`` matches the expert; a behavioral
cloner copying the marginal ``P(A)`` plays ~0.5 — the imitation analog of the MABUC gap.
"""

from __future__ import annotations

from typing import Any, ClassVar

import gymnasium as gym
import numpy as np

from causalrl.scm.graph import CausalGraph

_P_WIN = 0.9
_P_LOSE = 0.1


def make_imitation_diagram() -> tuple[CausalGraph, frozenset[str]]:
    """The imitable diagram: an *observed* confounder ``W -> A, W -> Y, A -> Y``."""
    graph = CausalGraph(directed_edges=[("W", "A"), ("W", "Y"), ("A", "Y")])
    return graph, frozenset({"W", "A", "Y"})


def make_unconfounded_observed_diagram() -> tuple[CausalGraph, frozenset[str]]:
    """The infeasible diagram: a *latent* confounder ``A <-> Y`` with only ``A, Y`` observed."""
    graph = CausalGraph(directed_edges=[("A", "Y")], bidirected_edges=[("A", "Y")])
    return graph, frozenset({"A", "Y"})


def expert_policy(observation: dict[str, int]) -> int:
    """The expert plays the lucky action ``A = W``."""
    return int(observation["W"])


class ImitationEnv(gym.Env):  # type: ignore[type-arg]
    """One-step confounded bandit: observe ``W``, play ``A``, reward ``Bernoulli(0.9 if A==W)``."""

    metadata: ClassVar[dict[str, list[str]]] = {"render_modes": []}  # type: ignore[misc]

    def __init__(self, seed: int | None = None) -> None:
        super().__init__()
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Dict({"W": gym.spaces.Discrete(2)})
        self._rng = np.random.default_rng(seed)
        self._w = 0

    def reset(  # type: ignore[override]
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, int], dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._w = int(self._rng.integers(0, 2))
        return {"W": self._w}, {}

    def step(  # type: ignore[override]
        self, action: int
    ) -> tuple[dict[str, int], float, bool, bool, dict[str, Any]]:
        p = _P_WIN if action == self._w else _P_LOSE
        reward = float(self._rng.random() < p)
        return {"W": self._w}, reward, True, False, {"w": self._w}


def generate_demonstrations(
    env: ImitationEnv, n: int = 2000, seed: int = 0
) -> dict[str, np.ndarray]:
    """Roll the expert out for ``n`` episodes; return integer columns ``{W, A, Y}``."""
    ws: list[int] = []
    actions: list[int] = []
    ys: list[int] = []
    for t in range(n):
        observation, _ = env.reset(seed=seed + t)
        action = expert_policy(observation)
        _, reward, _, _, _ = env.step(action)
        ws.append(observation["W"])
        actions.append(action)
        ys.append(int(reward))
    return {"W": np.array(ws), "A": np.array(actions), "Y": np.array(ys)}
