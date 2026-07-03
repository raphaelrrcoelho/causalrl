"""Causal potential-based reward shaping (taxonomy Task 8).

A sparse-reward chain MDP: the agent only sees a reward on reaching the far end, so an
epsilon-greedy learner has to stumble onto the goal before it learns anything. Using the causal
optimal value ``V*`` as a shaping potential (:func:`causalrl.causal_potential`) turns that sparse
signal into a dense one *without changing the optimal policy* (Ng, Harada & Russell 1999) — so the
same Q-learning budget converges to the optimal ("always move toward the goal") policy far faster.

Run:  python examples/reward_shaping_example.py
"""

from __future__ import annotations

from causalrl import causal_potential, q_learning, value_iteration
from causalrl.envs.suite.shaping import make_sparse_chain_mdp


def policy_accuracy(policy: dict[int, int], optimal: dict[int, int]) -> float:
    return sum(policy[s] == optimal[s] for s in optimal) / len(optimal)


def main() -> None:
    mdp = make_sparse_chain_mdp(length=15, gamma=0.9)
    optimal_policy = value_iteration(mdp)[1]
    episodes = 40

    unshaped = q_learning(mdp, episodes=episodes, seed=0)
    shaped = q_learning(mdp, episodes=episodes, potential=causal_potential(mdp), seed=0)

    unshaped_acc = policy_accuracy(unshaped, optimal_policy)
    shaped_acc = policy_accuracy(shaped, optimal_policy)
    print(f"Sparse-reward chain (length={mdp.n_states}), {episodes} Q-learning episodes each:")
    print(f"  without shaping: {unshaped_acc:.0%} optimal-policy match")
    print(f"  with causal_potential shaping: {shaped_acc:.0%} match")
    print("\nSame reward budget, same optimal policy — the causal potential just densifies credit.")


if __name__ == "__main__":
    main()
