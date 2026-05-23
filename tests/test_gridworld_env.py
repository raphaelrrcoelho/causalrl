from causalrl.data.dataset import generate_logs
from causalrl.envs.suite.gridworld import ConfoundedGridworld


def test_shapes_and_horizon():
    env = ConfoundedGridworld(size=3, seed=0)
    assert env.n_states == 9 + 1  # grid cells + terminal
    assert env.n_actions == 4
    obs, _ = env.reset(seed=0)
    assert obs["state"] == 0


def test_rewards_bounded_non_negative():
    env = ConfoundedGridworld(size=3, seed=0)
    total = 0.0
    for _ in range(200):
        _obs, _ = env.reset()
        done = False
        while not done:
            _, r, done, _, _ = env.step(1)
            total += r
    assert total >= 0.0


def test_logs_generate():
    env = ConfoundedGridworld(size=3, seed=0)
    d = generate_logs(env, n_episodes=200, seed=0)
    assert len(d) > 0
    assert d.n_actions == 4
