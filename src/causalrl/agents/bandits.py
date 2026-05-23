from __future__ import annotations

from typing import Any

import torch

from causalrl.agents.base import Agent


class NaiveThompsonSampling(Agent):
    """Beta-Bernoulli Thompson sampling, one posterior per arm. Ignores intuition,
    so it cannot distinguish the arms (their interventional means are equal)."""

    def __init__(self, n_arms: int, seed: int | None = None) -> None:
        self.n_arms = n_arms
        self._alpha = torch.ones(n_arms)  # type: ignore[reportPrivateImportUsage]
        self._beta = torch.ones(n_arms)  # type: ignore[reportPrivateImportUsage]
        self._gen = torch.Generator()  # type: ignore[reportPrivateImportUsage]
        if seed is not None:
            self._gen.manual_seed(seed)

    def act(self, observation: dict[str, Any]) -> int:
        samples = torch.distributions.Beta(self._alpha, self._beta).sample()
        return int(torch.argmax(samples).item())  # type: ignore[reportPrivateImportUsage]

    def update(self, observation: dict[str, Any], action: int, reward: float) -> None:
        if reward > 0:
            self._alpha[action] += 1.0
        else:
            self._beta[action] += 1.0


class CausalThompsonSampling(Agent):
    """Thompson sampling with one Beta posterior per (intuition, arm) cell.

    Conditioning on the observed confounder proxy `intuition` de-confounds the choice,
    letting the agent learn the lucky arm for each intuition value."""

    def __init__(self, n_arms: int, n_contexts: int, seed: int | None = None) -> None:
        self.n_arms = n_arms
        self.n_contexts = n_contexts
        self._alpha = torch.ones(n_contexts, n_arms)  # type: ignore[reportPrivateImportUsage]
        self._beta = torch.ones(n_contexts, n_arms)  # type: ignore[reportPrivateImportUsage]
        self._gen = torch.Generator()  # type: ignore[reportPrivateImportUsage]
        if seed is not None:
            self._gen.manual_seed(seed)

    def act(self, observation: dict[str, Any]) -> int:
        ctx = int(observation["intuition"])
        samples = torch.distributions.Beta(self._alpha[ctx], self._beta[ctx]).sample()
        return int(torch.argmax(samples).item())  # type: ignore[reportPrivateImportUsage]

    def update(self, observation: dict[str, Any], action: int, reward: float) -> None:
        ctx = int(observation["intuition"])
        if reward > 0:
            self._alpha[ctx, action] += 1.0
        else:
            self._beta[ctx, action] += 1.0
