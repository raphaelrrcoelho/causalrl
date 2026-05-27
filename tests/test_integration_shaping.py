"""Headline: causal-potential shaping reaches the optimum fast; sparse unshaped learning lags."""

from __future__ import annotations

from causalrl.envs.suite.shaping import make_sparse_chain_mdp
from causalrl.shaping import causal_potential, q_learning, value_iteration


def test_causal_shaping_speeds_learning_and_keeps_the_optimum() -> None:
    mdp = make_sparse_chain_mdp(length=12, gamma=0.9)
    optimal = value_iteration(mdp)[1]  # "always right"
    potential = causal_potential(mdp)

    shaped = q_learning(mdp, potential=potential, episodes=20, seed=0)
    unshaped = q_learning(mdp, episodes=20, seed=0)

    shaped_correct = sum(shaped[s] == optimal[s] for s in optimal)
    unshaped_correct = sum(unshaped[s] == optimal[s] for s in optimal)

    assert shaped == optimal  # causal shaping reaches the optimal policy within 20 episodes
    assert shaped_correct > unshaped_correct  # strictly ahead of sparse unshaped learning
