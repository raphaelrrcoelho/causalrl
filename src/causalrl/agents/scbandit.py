from collections.abc import Iterable
from typing import Any

import numpy as np

from causalrl.agents.base import Agent
from causalrl.identification.intervention_sets import pomis
from causalrl.scm.graph import CausalGraph


class _ArmSubsetThompsonSampling(Agent):
    """Beta-Bernoulli Thompson sampling restricted to a fixed subset of global arm indices.

    Posteriors are indexed locally; `act` returns the GLOBAL arm index. Reward is assumed in
    [0, 1] (Bernoulli-like), matching the bandit slice in ``agents/bandits.py``.
    """

    def __init__(self, allowed: list[int], seed: int | None = None) -> None:
        if not allowed:
            raise ValueError("agent must be given at least one allowed arm")
        self.allowed = allowed
        self._local = {global_i: local_i for local_i, global_i in enumerate(allowed)}
        self._alpha = np.ones(len(allowed))
        self._beta = np.ones(len(allowed))
        self._rng = np.random.default_rng(seed)

    def act(self, observation: dict[str, Any]) -> int:
        samples = np.asarray(self._rng.beta(self._alpha, self._beta), dtype=np.float64)
        return self.allowed[int(np.argmax(samples))]

    def update(self, observation: dict[str, Any], action: int, reward: float) -> None:
        local = self._local[action]
        if reward > 0:
            self._alpha[local] += 1.0
        else:
            self._beta[local] += 1.0


class POMISThompsonSampling(_ArmSubsetThompsonSampling):
    """Thompson sampling over only the arms whose intervened-variable set is a POMIS."""

    def __init__(
        self, graph: CausalGraph, reward: str, arms: list[dict[str, int]], seed: int | None = None
    ) -> None:
        pomis_sets = set(pomis(graph, reward))
        allowed = [i for i, arm in enumerate(arms) if frozenset(arm.keys()) in pomis_sets]
        super().__init__(allowed, seed)


class BruteForceInterventionTS(_ArmSubsetThompsonSampling):
    """Thompson sampling over every arm (no pruning) — the unguided baseline."""

    def __init__(self, arms: list[dict[str, int]], seed: int | None = None) -> None:
        super().__init__(list(range(len(arms))), seed)


class FixedSetThompsonSampling(_ArmSubsetThompsonSampling):
    """Thompson sampling restricted to a single fixed intervention set (the naive baseline)."""

    def __init__(
        self, arms: list[dict[str, int]], intervention_set: Iterable[str], seed: int | None = None
    ) -> None:
        target = frozenset(intervention_set)
        allowed = [i for i, arm in enumerate(arms) if frozenset(arm.keys()) == target]
        super().__init__(allowed, seed)
