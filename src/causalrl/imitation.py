"""Causal imitation learning (taxonomy Task 6).

Imitate an expert from demonstrations when unobserved confounders drive both the expert's actions
and the outcome. A naive behavioral-cloning learner that clones the *marginal* action distribution
is biased: it acts independently of the confounding the expert used, so its value is the
interventional ``sum_a P(a) E[Y|do(a)]``, not the observational ``E[Y]``. A causal imitator that
clones ``P(A | Z)`` for a back-door-admissible *observed* set ``Z`` reproduces the expert's reward
when imitation is feasible. ``is_imitable`` decides feasibility and returns ``None`` (rather than a
biased policy) when no observed admissible set exists.

Faithful to J. Zhang, D. Kumor, E. Bareinboim, *Causal Imitation Learning with Unobserved
Confounders*, NeurIPS 2020. No external code is ported.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from itertools import combinations
from typing import Any

import numpy as np

from causalrl.agents.base import Agent
from causalrl.exceptions import CausalGraphError
from causalrl.identification.transport import is_backdoor_admissible
from causalrl.scm.graph import CausalGraph

__all__ = ["BehavioralCloning", "CausalImitator", "imitation_backdoor_set", "is_imitable"]


def imitation_backdoor_set(
    graph: CausalGraph,
    *,
    action: str,
    outcome: str,
    observable: Iterable[str],
    max_size: int = 3,
) -> frozenset[str] | None:
    """Smallest observed back-door-admissible set for ``action -> outcome``, or ``None``.

    A set ``Z ⊆ observable \\ {action, outcome}`` is admissible when it has no descendant of
    ``action`` and blocks every back-door path (``action ⊥ outcome | Z`` with ``action``'s outgoing
    edges removed). Cloning ``P(action | Z)`` then reproduces the expert's outcome distribution.
    """
    for name in (action, outcome):
        if name not in graph.nodes:
            raise CausalGraphError(f"unknown node: {name!r}")
    observed = set(observable)
    unknown = observed - set(graph.nodes)
    if unknown:
        raise CausalGraphError(f"unknown observable node(s): {sorted(unknown)}")
    candidates = sorted(observed - {action, outcome})
    for size in range(min(max_size, len(candidates)) + 1):
        for combo in combinations(candidates, size):
            z = set(combo)
            if is_backdoor_admissible(graph, action, outcome, z):
                return frozenset(z)
    return None


def is_imitable(
    graph: CausalGraph, *, action: str, outcome: str, observable: Iterable[str]
) -> bool:
    """Whether the expert is imitable: an observed back-door-admissible set exists."""
    return (
        imitation_backdoor_set(graph, action=action, outcome=outcome, observable=observable)
        is not None
    )


class BehavioralCloning(Agent):
    """Naive imitator: clones the marginal action distribution ``P(A)``, ignoring covariates."""

    def __init__(self, n_actions: int, seed: int | None = None) -> None:
        self.n_actions = n_actions
        self._rng = np.random.default_rng(seed)
        self._probs = np.ones(n_actions) / n_actions

    def fit(self, demonstrations: Mapping[str, np.ndarray], *, action: str) -> None:
        actions = np.asarray(demonstrations[action]).astype(int)
        counts = np.bincount(actions, minlength=self.n_actions).astype(float)
        self._probs = counts / counts.sum()

    def act(self, observation: dict[str, Any]) -> int:
        return int(self._rng.choice(self.n_actions, p=self._probs))

    def update(self, observation: dict[str, Any], action: int, reward: float) -> None:
        """No-op: imitation is fit offline from demonstrations."""


class CausalImitator(Agent):
    """Clones ``P(A | Z)`` for a back-door-admissible observed set ``Z`` (the adjustment set).

    Conditioning the cloned policy on ``Z`` reproduces the confounding the expert responded to, so
    deployment matches the expert's reward. Unseen ``Z`` values fall back to the marginal ``P(A)``.
    """

    def __init__(self, n_actions: int, adjustment: Sequence[str], seed: int | None = None) -> None:
        self.n_actions = n_actions
        self._adjustment = list(adjustment)
        self._rng = np.random.default_rng(seed)
        self._marginal = np.ones(n_actions) / n_actions
        self._conditional: dict[tuple[int, ...], np.ndarray] = {}

    def fit(self, demonstrations: Mapping[str, np.ndarray], *, action: str) -> None:
        actions = np.asarray(demonstrations[action]).astype(int)
        marginal = np.bincount(actions, minlength=self.n_actions).astype(float)
        self._marginal = marginal / marginal.sum()
        z_columns = [np.asarray(demonstrations[z]).astype(int) for z in self._adjustment]
        counts: dict[tuple[int, ...], np.ndarray] = {}
        for i in range(len(actions)):
            key = tuple(int(column[i]) for column in z_columns)
            if key not in counts:
                counts[key] = np.zeros(self.n_actions)
            counts[key][int(actions[i])] += 1.0
        self._conditional = {key: row / row.sum() for key, row in counts.items()}

    def act(self, observation: dict[str, Any]) -> int:
        key = tuple(int(observation[z]) for z in self._adjustment)
        probs = self._conditional.get(key, self._marginal)
        return int(self._rng.choice(self.n_actions, p=probs))

    def update(self, observation: dict[str, Any], action: int, reward: float) -> None:
        """No-op: imitation is fit offline from demonstrations."""
