"""Reward shaping: value iteration, the policy-invariance theorem, and the shaping transform."""

from __future__ import annotations

import numpy as np

from causalrl.envs.suite.shaping import make_sparse_chain_mdp
from causalrl.shaping import apply_potential_shaping, causal_potential, value_iteration


def test_value_iteration_solves_the_chain() -> None:
    mdp = make_sparse_chain_mdp(length=6, gamma=0.9)
    values, policy = value_iteration(mdp)
    goal = 5
    assert all(policy[s] == 1 for s in range(goal))  # always right
    for s in range(goal):
        assert abs(values[s] - 0.9 ** (goal - 1 - s)) < 1e-6


def test_causal_potential_pins_terminals() -> None:
    mdp = make_sparse_chain_mdp(length=5)
    assert causal_potential(mdp)[4] == 0.0


def test_potential_shaping_preserves_optimal_policy() -> None:
    mdp = make_sparse_chain_mdp(length=7, gamma=0.9)
    base_policy = value_iteration(mdp)[1]
    # The causal potential.
    assert value_iteration(apply_potential_shaping(mdp, causal_potential(mdp)))[1] == base_policy
    # An arbitrary potential (terminal pinned to 0, as episodic invariance requires).
    rng = np.random.default_rng(0)
    arbitrary = {s: float(rng.normal()) for s in range(mdp.n_states)}
    arbitrary[6] = 0.0
    assert value_iteration(apply_potential_shaping(mdp, arbitrary))[1] == base_policy


def test_apply_potential_shaping_adds_the_difference() -> None:
    mdp = make_sparse_chain_mdp(length=4, gamma=0.9)
    phi = {0: 1.0, 1: 2.0, 2: 3.0, 3: 0.0}
    shaped = apply_potential_shaping(mdp, phi)
    # (s=0, a=1) -> s'=1: F = 0.9 * phi[1] - phi[0] = 0.9 * 2 - 1 = 0.8
    assert abs(shaped.rewards[(0, 1)] - (mdp.rewards[(0, 1)] + 0.8)) < 1e-9
