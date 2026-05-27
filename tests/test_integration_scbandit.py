from causalrl.agents.base import Agent
from causalrl.agents.scbandit import (
    BruteForceInterventionTS,
    FixedSetThompsonSampling,
    POMISThompsonSampling,
)
from causalrl.envs.suite.scbandit import StructuralCausalBanditEnv, make_confounded_chain_env
from causalrl.eval.metrics import cumulative_regret


def _run(agent: Agent, env: StructuralCausalBanditEnv, n_steps: int, seed: int) -> list[float]:
    obs, _ = env.reset(seed=seed)
    rewards: list[float] = []
    for _ in range(n_steps):
        action = agent.act(obs)
        next_obs, reward, _term, _trunc, _info = env.step(action)
        agent.update(obs, action, reward)
        rewards.append(reward)
        obs = next_obs
    return rewards


def _tail_mean(rewards: list[float], window: int) -> float:
    return sum(rewards[-window:]) / window


def test_pomis_agent_beats_brute_force_and_naive():
    n_steps = 8000
    env = make_confounded_chain_env(seed=1)

    pomis_rewards = _run(
        POMISThompsonSampling(env.graph, env.reward, env.arms, seed=0), env, n_steps, seed=1
    )
    brute_rewards = _run(BruteForceInterventionTS(env.arms, seed=0), env, n_steps, seed=2)
    naive_rewards = _run(FixedSetThompsonSampling(env.arms, {"X3"}, seed=0), env, n_steps, seed=3)

    opt = env.optimal_value  # ~1.0 (the observational arm)
    window = 2000

    # 1. POMIS finds the optimal (observational) arm.
    assert _tail_mean(pomis_rewards, window) >= 0.9

    # 2. POMIS strictly beats the naive fixed-set agent, which is stuck near 0.5.
    assert _tail_mean(pomis_rewards, window) - _tail_mean(naive_rewards, window) >= 0.3

    # 3. POMIS converges faster than brute force: fewer arms => less exploration regret.
    pomis_regret = cumulative_regret(pomis_rewards, opt)
    brute_regret = cumulative_regret(brute_rewards, opt)
    assert pomis_regret < brute_regret
