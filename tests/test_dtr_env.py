import pytest

from causalrl.envs.suite.dtr import DTREnv


def test_env_shape_and_horizon():
    env = DTREnv(seed=0)
    assert env.n_states == 5
    assert env.n_actions == 2
    obs, _ = env.reset(seed=0)
    assert obs["state"] == 0
    assert obs["t"] == 0


def test_optimal_policy_reaches_reward_one():
    env = DTREnv(seed=1)
    wins = 0
    n = 4000
    for _ in range(n):
        obs, _ = env.reset()
        a0 = 0
        obs, r, done, _, _ = env.step(a0)
        c1 = obs["state"] - 2  # state = 2 + c1 at stage 1
        u_inferred = a0 ^ c1
        _, r, done, _, _ = env.step(u_inferred)
        assert done
        wins += int(r == 1.0)
    assert wins / n > 0.99


def test_behavior_logs_are_confounded():
    try:
        from causalrl.data.dataset import generate_logs
    except ImportError:
        pytest.skip("generate_logs lands in Task A4")
    env = DTREnv(seed=2)
    d = generate_logs(env, n_episodes=5000, seed=2)
    means = [d.mean_reward(s, a) for s in (2, 3) for a in (0, 1)]
    assert min(means) < 0.6
