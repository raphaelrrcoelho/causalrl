from causalrl.data.dataset import generate_logs
from causalrl.envs.suite.seq_mabuc import SequentialMABUCEnv


def test_shapes_and_horizon():
    env = SequentialMABUCEnv(horizon=3, seed=0)
    assert env.horizon == 3
    assert env.n_actions == 2
    assert env.n_states == 3 * 2 + 1  # 2 per step + terminal
    obs, _ = env.reset(seed=0)
    assert obs["t"] == 0


def test_following_intuition_scores_well():
    env = SequentialMABUCEnv(horizon=3, seed=1)
    total = 0.0
    n = 2000
    for _ in range(n):
        obs, _ = env.reset()
        done = False
        while not done:
            intuition = obs["state"] % 2  # state = step*2 + intuition
            obs, r, done, _, _ = env.step(intuition)
            total += r
    assert total / n > 0.6 * 3  # ~0.75/step over horizon 3


def test_logs_generate():
    env = SequentialMABUCEnv(horizon=3, seed=0)
    d = generate_logs(env, n_episodes=300, seed=0)
    assert len(d) == 900  # 3 steps per episode
    assert d.n_actions == 2
