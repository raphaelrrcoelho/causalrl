from causalrl.data.dataset import generate_logs
from causalrl.envs.suite.dtr import DTREnv


def test_env_shape_and_horizon():
    env = DTREnv(seed=0)
    assert env.n_states == 3
    assert env.n_actions == 2
    assert env.horizon == 1
    obs, _ = env.reset(seed=0)
    assert obs["state"] in (0, 1)  # observed subtype Z
    assert obs["t"] == 0


def test_do_values_and_optimal_policy():
    env = DTREnv()
    # do-optimal action is the matched treatment a = Z.
    assert env.do_value(0, 0) > env.do_value(0, 1)  # subtype 0 -> treatment 0
    assert env.do_value(1, 1) > env.do_value(1, 0)  # subtype 1 -> treatment 1
    assert abs(env.optimal_value - 0.75) < 1e-9


def test_optimal_policy_reaches_its_value_empirically():
    env = DTREnv(seed=1)
    n = 20000
    total = 0.0
    for _ in range(n):
        obs, _ = env.reset()
        z = obs["state"]
        _, r, done, _, _ = env.step(z)  # matched treatment a = Z is optimal
        assert done
        total += r
    assert abs(total / n - 0.75) < 0.02


def test_naive_offline_estimate_is_confounded_toward_treatment_one():
    # In the confounded logs, treatment 1 has a HIGHER apparent mean reward than its true
    # do-value for subtype 0 (U inflates it), so a naive learner would prefer it wrongly.
    env = DTREnv(seed=2)
    d = generate_logs(env, n_episodes=20000, seed=2)
    obs_mean_z0_a1 = d.mean_reward(0, 1)
    true_do_z0_a1 = env.do_value(0, 1)
    assert obs_mean_z0_a1 > true_do_z0_a1 + 0.1  # confounding inflates apparent reward
    # and naive would pick treatment 1 for subtype 0 even though treatment 0 is optimal
    assert d.mean_reward(0, 1) > d.mean_reward(0, 0)
    assert env.do_value(0, 0) > env.do_value(0, 1)
