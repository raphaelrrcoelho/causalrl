from typing import Any

from causalrl.agents.base import Agent


def run_episodes(agent: Agent, env: Any, n_episodes: int, seed: int) -> list[float]:
    """Run an agent on a finite-horizon env for n_episodes, returning per-episode returns.

    The agent acts each step and is updated with the step reward; the per-episode return is
    the sum of step rewards (terminal-reward envs put it all on the last step).
    """
    returns: list[float] = []
    obs, _ = env.reset(seed=seed)
    for ep in range(n_episodes):
        if ep > 0:
            obs, _ = env.reset()
        done = False
        ep_return = 0.0
        while not done:
            action = agent.act(obs)
            next_obs, reward, done, _trunc, _info = env.step(action)
            agent.update(obs, action, float(reward))
            obs = next_obs
            ep_return += float(reward)
        returns.append(ep_return)
    return returns
