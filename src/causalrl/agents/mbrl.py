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
from collections.abc import Mapping
from typing import Any

import numpy as np

from causalrl.agents.base import Agent
from causalrl.data.dataset import ConfoundedTrajectoryDataset
from causalrl.identification.criteria import backdoor_adjustment_set
from causalrl.scale import certify_policy
from causalrl.scm.graph import CausalGraph


class CertifiedPolicyAgent(Agent):
    """Ship the best deterministic policy whose improvement over behavior certifies robust to hidden
    confounding; abstain to the empirical behavior policy otherwise."""

    def __init__(self, n_states: int, n_actions: int, *, gamma_max: float = 5.0) -> None:
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma_max = gamma_max
        self.policy: list[int] = [0] * n_states

    def _behavior_policy(self, dataset: ConfoundedTrajectoryDataset) -> list[int]:
        """Empirical behavior policy: the most-logged action per state (abstention target)."""
        return [
            max(range(self.n_actions), key=lambda a: dataset.behavior_propensity(s, a))
            for s in range(self.n_states)
        ]

    def ingest_offline(self, dataset: ConfoundedTrajectoryDataset) -> None:
        transitions = dataset.transitions
        best_policy = self._behavior_policy(dataset)  # abstention default
        best_contrast = 0.0
        for candidate in itertools.product(range(self.n_actions), repeat=self.n_states):
            # Skip a policy that assigns a never-logged action in some state: an unseen action's
            # value is not identified from the logs, and certify_policy has no support to bound it.
            if any(
                dataset.behavior_propensity(s, candidate[s]) == 0.0 for s in range(self.n_states)
            ):
                continue
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


class BackdoorAdjustedAgent(Agent):
    """Active deconfounded optimizer: pick the action with the highest back-door-adjusted value
    ``E[Y | do(A=a)] = Σ_z P(z) · E[Y | A=a, Z=z]``, with the adjustment set read from the graph via
    :func:`~causalrl.backdoor_adjustment_set`.

    Unlike the certify-gated agent (whose ceiling is the behavior policy), this *recovers* the
    interventional optimum from confounded logs, given an observed admissible adjustment set. It is
    fitted on columnar data ``{treatment, outcome, *adjustment}`` (equal-length arrays).
    """

    def __init__(
        self,
        n_actions: int,
        *,
        graph: CausalGraph,
        treatment: str = "A",
        outcome: str = "Y",
    ) -> None:
        self.n_actions = n_actions
        self.treatment = treatment
        self.outcome = outcome
        self.adjustment: tuple[str, ...] = tuple(
            sorted(backdoor_adjustment_set(graph, treatment, outcome))
        )
        self._best_action = 0
        self.values: list[float] = [0.0] * n_actions

    def fit(self, data: Mapping[str, np.ndarray]) -> None:
        """Estimate each action's back-door-adjusted value and select the argmax."""
        self.values = [self._adjusted_value(a, data) for a in range(self.n_actions)]
        self._best_action = int(np.argmax(np.asarray(self.values)))

    def _adjusted_value(self, action: int, data: Mapping[str, np.ndarray]) -> float:
        a = np.asarray(data[self.treatment])
        y = np.asarray(data[self.outcome], dtype=float)
        if not self.adjustment:
            sel = y[a == action]
            return float(sel.mean()) if sel.size else 0.0
        strata = np.stack([np.asarray(data[z]) for z in self.adjustment], axis=1)
        total = 0.0
        for stratum in np.unique(strata, axis=0):
            in_z = np.all(strata == stratum, axis=1)
            p_z = float(in_z.mean())
            in_az = in_z & (a == action)
            # Back-door term; on a positivity gap (action unseen in a stratum) fall back to the
            # stratum-marginal outcome so the sum stays a proper P(z)-weighted average.
            outcome_mean = float(y[in_az].mean()) if in_az.any() else float(y[in_z].mean())
            total += p_z * outcome_mean
        return total

    def act(self, observation: dict[str, Any]) -> int:
        return int(self._best_action)

    def update(self, observation: dict[str, Any], action: int, reward: float) -> None:
        """Fixed action from the fitted adjustment; no online update."""
