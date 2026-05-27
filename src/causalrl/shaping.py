"""Causal reward shaping (taxonomy Task 8).

Potential-based reward shaping adds ``F(s, s') = gamma * Phi(s') - Phi(s)`` to the reward. For any
potential ``Phi`` this leaves the optimal policy unchanged (Ng, Harada & Russell 1999). Using the
causal optimal value ``V*`` as the potential turns a sparse reward dense, so a learner reaches the
optimal policy far faster, without changing what is optimal.

Faithful to A. Ng, D. Harada, S. Russell, *Policy Invariance Under Reward Transformations: Theory
and Application to Reward Shaping*, ICML 1999. No external code is ported.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace

import numpy as np

__all__ = [
    "TabularMDP",
    "apply_potential_shaping",
    "causal_potential",
    "q_learning",
    "value_iteration",
]


@dataclass(frozen=True)
class TabularMDP:
    """A finite deterministic MDP: ``transitions``/``rewards`` are keyed by ``(state, action)``."""

    n_states: int
    n_actions: int
    transitions: dict[tuple[int, int], int]
    rewards: dict[tuple[int, int], float]
    terminals: frozenset[int]
    gamma: float


def value_iteration(
    mdp: TabularMDP, *, tol: float = 1e-9, max_iter: int = 1000
) -> tuple[dict[int, float], dict[int, int]]:
    """Return ``(V*, greedy optimal policy)`` for the deterministic MDP."""
    values = dict.fromkeys(range(mdp.n_states), 0.0)
    for _ in range(max_iter):
        delta = 0.0
        for s in range(mdp.n_states):
            if s in mdp.terminals:
                continue
            best = max(
                mdp.rewards[(s, a)] + mdp.gamma * values[mdp.transitions[(s, a)]]
                for a in range(mdp.n_actions)
            )
            delta = max(delta, abs(best - values[s]))
            values[s] = best
        if delta < tol:
            break
    policy: dict[int, int] = {}
    for s in range(mdp.n_states):
        if s in mdp.terminals:
            continue
        q = [
            mdp.rewards[(s, a)] + mdp.gamma * values[mdp.transitions[(s, a)]]
            for a in range(mdp.n_actions)
        ]
        policy[s] = int(np.argmax(q))
    return values, policy


def causal_potential(mdp: TabularMDP) -> dict[int, float]:
    """The ideal (causal) shaping potential: ``V*`` with terminal states pinned to ``0`` (the
    condition under which potential shaping is policy-invariant for episodic tasks)."""
    values, _ = value_iteration(mdp)
    return {s: (0.0 if s in mdp.terminals else values[s]) for s in range(mdp.n_states)}


def apply_potential_shaping(mdp: TabularMDP, potential: Mapping[int, float]) -> TabularMDP:
    """Return a new MDP with ``rewards[s, a] += gamma * Phi(s') - Phi(s)`` (policy-invariant)."""
    shaped = dict(mdp.rewards)
    for (s, a), s_next in mdp.transitions.items():
        shaped[(s, a)] = mdp.rewards[(s, a)] + mdp.gamma * potential[s_next] - potential[s]
    return replace(mdp, rewards=shaped)


def q_learning(
    mdp: TabularMDP,
    *,
    episodes: int,
    potential: Mapping[int, float] | None = None,
    alpha: float = 0.5,
    epsilon: float = 0.1,
    max_steps: int | None = None,
    seed: int | None = None,
) -> dict[int, int]:
    """Tabular epsilon-greedy Q-learning from state ``0``. With ``potential`` given, the reward is
    shaped online by ``gamma * Phi(s') - Phi(s)``. Returns the greedy policy."""
    rng = np.random.default_rng(seed)
    q = np.zeros((mdp.n_states, mdp.n_actions))
    steps_cap = max_steps if max_steps is not None else 4 * mdp.n_states
    for _ in range(episodes):
        s = 0
        for _ in range(steps_cap):
            if s in mdp.terminals:
                break
            if rng.random() < epsilon:
                a = int(rng.integers(0, mdp.n_actions))
            else:
                a = int(np.argmax(q[s]))
            s_next = mdp.transitions[(s, a)]
            reward = mdp.rewards[(s, a)]
            if potential is not None:
                reward = reward + mdp.gamma * potential[s_next] - potential[s]
            bootstrap = 0.0 if s_next in mdp.terminals else float(np.max(q[s_next]))
            q[s, a] += alpha * (reward + mdp.gamma * bootstrap - q[s, a])
            s = s_next
    return {s: int(np.argmax(q[s])) for s in range(mdp.n_states) if s not in mdp.terminals}
