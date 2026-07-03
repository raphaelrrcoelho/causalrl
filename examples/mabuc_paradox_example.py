"""Reproduce the MABUC-paradox launch figure (running-average reward, causal vs naive).

Both arms of :class:`~causalrl.envs.suite.mabuc.MABUCEnv` share the same interventional mean
``E[Y|do(X=a)] = 0.5``, so a policy that only reasons about interventions cannot tell them apart.
Conditioning on the observed *intuition* (the MABUC confounder proxy) reveals the lucky arm per
context: :class:`~causalrl.agents.bandits.CausalThompsonSampling` converges to ~0.75 average
reward, while :class:`~causalrl.agents.bandits.NaiveThompsonSampling` (ignores intuition) is
capped near ~0.50 — the running-average curves behind the README's launch figure.

Run:  python examples/mabuc_paradox_example.py
"""

from __future__ import annotations

import numpy as np

from causalrl.agents.bandits import CausalThompsonSampling, NaiveThompsonSampling
from causalrl.agents.base import Agent
from causalrl.envs.suite.mabuc import MABUCEnv


def running_average_rewards(agent: Agent, *, n_steps: int, seed: int) -> np.ndarray:
    """Play `agent` on a fresh :class:`MABUCEnv` for `n_steps` and return the running-average
    reward curve ``cumsum(rewards) / step`` (length `n_steps`) — the quantity plotted in the
    launch figure."""
    env = MABUCEnv(seed=seed)
    obs, _ = env.reset(seed=seed)
    rewards = np.empty(n_steps)
    for t in range(n_steps):
        action = agent.act(obs)
        _, reward, _, _, _ = env.step(action)
        agent.update(obs, action, reward)
        obs, _ = env.reset()
        rewards[t] = reward
    return np.cumsum(rewards) / np.arange(1, n_steps + 1)


def main() -> None:
    n_steps = 8000
    causal_curve = running_average_rewards(
        CausalThompsonSampling(n_arms=2, n_contexts=2, seed=0), n_steps=n_steps, seed=1
    )
    naive_curve = running_average_rewards(
        NaiveThompsonSampling(n_arms=2, seed=0), n_steps=n_steps, seed=1
    )

    print("MABUC running-average reward (causal vs naive), both do(X=a) means are 0.5:")
    for t in (1000, 2000, 4000, 8000):
        print(f"  step {t:>5}:  causal={causal_curve[t - 1]:.3f}  naive={naive_curve[t - 1]:.3f}")
    print(f"\nfinal: causal={causal_curve[-1]:.3f} (~0.75), naive={naive_curve[-1]:.3f} (~0.50)")


if __name__ == "__main__":
    main()
