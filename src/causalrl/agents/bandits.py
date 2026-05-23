from typing import Any

import numpy as np

from causalrl.agents.base import Agent


class NaiveThompsonSampling(Agent):
    """Beta-Bernoulli Thompson sampling, one posterior per arm. Ignores intuition,
    so it cannot distinguish the arms (their interventional means are equal).

    Sampling uses a per-instance ``numpy.random.Generator`` seeded from ``seed``, so two
    agents with different seeds draw independent action sequences and a fixed seed is
    reproducible regardless of global RNG state.
    """

    def __init__(self, n_arms: int, seed: int | None = None) -> None:
        self.n_arms = n_arms
        self._alpha = np.ones(n_arms)
        self._beta = np.ones(n_arms)
        self._rng = np.random.default_rng(seed)

    def act(self, observation: dict[str, Any]) -> int:
        samples = np.asarray(self._rng.beta(self._alpha, self._beta), dtype=np.float64)
        return int(np.argmax(samples))

    def update(self, observation: dict[str, Any], action: int, reward: float) -> None:
        if reward > 0:
            self._alpha[action] += 1.0
        else:
            self._beta[action] += 1.0


class CausalThompsonSampling(Agent):
    """Thompson sampling with one Beta posterior per (intuition, arm) cell.

    Conditioning on the observed confounder proxy `intuition` de-confounds the choice,
    letting the agent learn the lucky arm for each intuition value.

    Sampling uses a per-instance ``numpy.random.Generator`` seeded from ``seed`` (see
    :class:`NaiveThompsonSampling` for the reproducibility contract).
    """

    def __init__(self, n_arms: int, n_contexts: int, seed: int | None = None) -> None:
        self.n_arms = n_arms
        self.n_contexts = n_contexts
        self._alpha = np.ones((n_contexts, n_arms))
        self._beta = np.ones((n_contexts, n_arms))
        self._rng = np.random.default_rng(seed)

    def act(self, observation: dict[str, Any]) -> int:
        ctx = int(observation["intuition"])
        samples = np.asarray(self._rng.beta(self._alpha[ctx], self._beta[ctx]), dtype=np.float64)
        return int(np.argmax(samples))

    def update(self, observation: dict[str, Any], action: int, reward: float) -> None:
        ctx = int(observation["intuition"])
        if reward > 0:
            self._alpha[ctx, action] += 1.0
        else:
            self._beta[ctx, action] += 1.0
