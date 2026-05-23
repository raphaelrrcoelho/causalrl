from causalrl.agents.baselines import OnlineOnlyUCB
from causalrl.envs.suite.dtr import DTREnv
from causalrl.eval.harness import run_episodes
from causalrl.eval.metrics import finite_horizon_regret


def test_run_episodes_returns_returns_list():
    env = DTREnv(seed=0)
    agent = OnlineOnlyUCB(n_states=3, n_actions=2, seed=0)
    returns = run_episodes(agent, env, n_episodes=50, seed=0)
    assert len(returns) == 50
    assert all(r in (0.0, 1.0) for r in returns)


def test_finite_horizon_regret():
    # optimal 0.75/episode, realized [0.75, 0.0, 0.75] -> regret = 0 + 0.75 + 0 = 0.75
    assert abs(finite_horizon_regret([0.75, 0.0, 0.75], optimal_return=0.75) - 0.75) < 1e-9
