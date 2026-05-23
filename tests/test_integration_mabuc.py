from causalrl.agents.bandits import CausalThompsonSampling, NaiveThompsonSampling
from causalrl.agents.base import Agent
from causalrl.envs.suite.mabuc import MABUCEnv
from causalrl.eval.metrics import cumulative_regret


def run_episode_loop(agent: Agent, env: MABUCEnv, n_rounds: int) -> float:
    total = 0.0
    obs, _ = env.reset(seed=123)
    for _ in range(n_rounds):
        action = agent.act(obs)
        _, reward, _, _, _ = env.step(action)
        agent.update(obs, action, reward)
        obs, _ = env.reset()
        total += reward
    return total


def test_causal_agent_beats_naive_on_mabuc():
    n = 8000
    causal = CausalThompsonSampling(n_arms=2, n_contexts=2, seed=0)
    naive = NaiveThompsonSampling(n_arms=2, seed=0)

    causal_total = run_episode_loop(causal, MABUCEnv(seed=1), n)
    naive_total = run_episode_loop(naive, MABUCEnv(seed=1), n)

    # Causal agent converges toward 0.75/round; naive is stuck near 0.5/round.
    assert causal_total / n > 0.68
    assert naive_total / n < 0.6
    assert causal_total > naive_total


def test_cumulative_regret_basic():
    # optimal 0.75/step, realized [1,0,1] -> regret = (0.75-1)+(0.75-0)+(0.75-1) = 0.25
    assert abs(cumulative_regret([1.0, 0.0, 1.0], optimal_per_step=0.75) - 0.25) < 1e-9
