import warnings
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
        if not 0.0 <= reward <= 1.0:
            raise ValueError(f"reward must lie in [0, 1], got {reward}")
        local = self._local[action]
        self._alpha[local] += reward
        self._beta[local] += 1.0 - reward


class POMISThompsonSampling(_ArmSubsetThompsonSampling):
    """Thompson sampling over only the arms whose intervened-variable set is a POMIS.

    The manipulable set is inferred from the arms (the variables any arm intervenes on), so
    non-manipulable variables are handled by the POMIS engine via latent projection (r40).
    When every non-reward variable is manipulable this matches the unconstrained POMIS.
    """

    def __init__(
        self,
        graph: CausalGraph,
        reward: str,
        arms: list[dict[str, int]],
        seed: int | None = None,
        *,
        manipulable: Iterable[str] | None = None,
    ) -> None:
        arm_variables = {v for arm in arms for v in arm}
        if manipulable is None:
            warnings.warn(
                "inferring manipulable variables from arms is deprecated; "
                "pass manipulable= explicitly",
                DeprecationWarning,
                stacklevel=2,
            )
            permitted = arm_variables
        else:
            permitted = set(manipulable)
            unexpected = arm_variables - permitted
            if unexpected:
                raise ValueError(
                    f"arms intervene on non-manipulable variables: {sorted(unexpected)}"
                )
        pomis_sets = set(pomis(graph, reward, manipulable=permitted))
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


class NaivePOMISThompsonSampling(_ArmSubsetThompsonSampling):
    """Naive baseline that ignores manipulability: it computes the UNCONSTRAINED POMIS and
    keeps only the arms that happen to exist. When the optimal lever lies outside the
    unconstrained POMIS — as when a non-manipulable variable dominates — it cannot reach the
    optimum (r40 Prop. 1: filtering the unconstrained POMIS is insufficient)."""

    def __init__(
        self, graph: CausalGraph, reward: str, arms: list[dict[str, int]], seed: int | None = None
    ) -> None:
        pomis_sets = set(pomis(graph, reward))  # unconstrained — ignores manipulability
        allowed = [i for i, arm in enumerate(arms) if frozenset(arm.keys()) in pomis_sets]
        super().__init__(allowed, seed)
