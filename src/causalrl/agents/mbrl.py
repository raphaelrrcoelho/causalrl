"""Model-based agents for the causal-MBRL probe.

:class:`CertifiedPolicyAgent` is the confounding-robust causal agent for the M0 kill-gate. It ships
the highest-contrast deterministic policy whose improvement over the behavior policy is *certified*
robust to hidden confounding by :func:`causalrl.certify_policy` (Tan's marginal sensitivity model),
and abstains to the empirical behavior policy when nothing certifies. The certificate is the
decision rule — the honest robust planner, since a naive Manski-lower-bound greedy does not correct
a backdoor ``A <- U -> Y``.
"""

from __future__ import annotations

import itertools
from typing import Any

from causalrl.agents.base import Agent
from causalrl.data.dataset import ConfoundedTrajectoryDataset
from causalrl.scale import certify_policy


class CertifiedPolicyAgent(Agent):
    """Ship the best deterministic policy whose improvement over behavior certifies robust to hidden
    confounding; abstain to the empirical behavior policy otherwise."""

    def __init__(self, n_states: int, n_actions: int, *, gamma_max: float = 5.0) -> None:
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma_max = gamma_max
        self.policy: list[int] = [0] * n_states

    def _behavior_policy(self, dataset: ConfoundedTrajectoryDataset) -> list[int]:
        """The empirical behavior policy: the most-logged action in each state (abstention target)."""
        return [
            max(range(self.n_actions), key=lambda a: dataset.behavior_propensity(s, a))
            for s in range(self.n_states)
        ]

    def ingest_offline(self, dataset: ConfoundedTrajectoryDataset) -> None:
        transitions = dataset.transitions
        best_policy = self._behavior_policy(dataset)  # abstention default
        best_contrast = 0.0
        for candidate in itertools.product(range(self.n_actions), repeat=self.n_states):
            target_actions = [candidate[tr.state] for tr in transitions]
            cert = certify_policy(dataset, target_actions, gamma_max=self.gamma_max)
            if cert.certified and cert.naive_contrast > best_contrast:
                best_contrast = cert.naive_contrast
                best_policy = list(candidate)
        self.policy = best_policy

    def act(self, observation: dict[str, Any]) -> int:
        return int(self.policy[int(observation["state"])])

    def update(self, observation: dict[str, Any], action: int, reward: float) -> None:
        """Fixed policy from the logs; no online update."""
