from causalrl.agents.base import Agent
from causalrl.agents.scbandit import NaivePOMISThompsonSampling, POMISThompsonSampling
from causalrl.envs.suite.scbandit import StructuralCausalBanditEnv, make_frontdoor_env
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


def test_manipulability_aware_pomis_beats_naive():
    n_steps = 30000
    env = make_frontdoor_env(seed=1)

    aware = _run(
        POMISThompsonSampling(env.graph, env.reward, env.arms, seed=0, manipulable=env.manipulable),
        env,
        n_steps,
        seed=1,
    )
    naive = _run(
        NaivePOMISThompsonSampling(env.graph, env.reward, env.arms, seed=0), env, n_steps, seed=2
    )

    opt = env.optimal_value  # ~0.56 (do(X=1), reachable only by the manipulability-aware agent)
    window = 10000

    # 1. The M-aware agent steers through X and approaches the front-door optimum.
    assert _tail_mean(aware, window) >= 0.53
    # 2. The naive agent only has the observational arm and is stuck near 0.50.
    assert _tail_mean(naive, window) <= 0.52
    # 3. It strictly beats naive, both in tail reward and cumulative regret.
    assert _tail_mean(aware, window) - _tail_mean(naive, window) >= 0.025
    assert cumulative_regret(aware, opt) < cumulative_regret(naive, opt)
