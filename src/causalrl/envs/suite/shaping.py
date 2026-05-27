"""Sparse-reward chain MDP for causal reward shaping (taxonomy Task 8)."""

from __future__ import annotations

from causalrl.shaping import TabularMDP


def make_sparse_chain_mdp(length: int = 10, gamma: float = 0.9) -> TabularMDP:
    """A deterministic chain ``0 -> 1 -> ... -> length-1``. Action ``1`` moves right (+1), action
    ``0`` moves left (-1), clamped to ``[0, length-1]``. Reward ``1.0`` only on entering the
    terminal goal, ``0`` elsewhere; the sparse-reward task where shaping matters. The optimal
    policy is "always right" and ``V*(s) = gamma ** (goal - 1 - s)``."""
    if length < 2:
        raise ValueError("chain length must be at least 2")
    goal = length - 1
    transitions: dict[tuple[int, int], int] = {}
    rewards: dict[tuple[int, int], float] = {}
    for s in range(length):
        if s == goal:
            continue
        left = max(0, s - 1)
        right = min(goal, s + 1)
        transitions[(s, 0)] = left
        transitions[(s, 1)] = right
        rewards[(s, 0)] = 1.0 if left == goal else 0.0
        rewards[(s, 1)] = 1.0 if right == goal else 0.0
    return TabularMDP(
        n_states=length,
        n_actions=2,
        transitions=transitions,
        rewards=rewards,
        terminals=frozenset({goal}),
        gamma=gamma,
    )
