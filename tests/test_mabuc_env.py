from causalrl.envs.suite.mabuc import MABUCEnv, build_mabuc_scm


def test_do_means_are_equal_across_arms():
    # The defining MABUC property: marginal interventional means are identical.
    scm = build_mabuc_scm()
    mean0 = scm.do({"X": 0.0}).see(20000, seed=0)["Y"].mean().item()
    mean1 = scm.do({"X": 1.0}).see(20000, seed=1)["Y"].mean().item()
    assert abs(mean0 - 0.5) < 0.02
    assert abs(mean1 - 0.5) < 0.02


def test_intuition_reveals_lucky_arm():
    scm = build_mabuc_scm()
    s = scm.see(20000, seed=2)
    # behavior plays X = I, and lucky arm == I, so observed reward concentrates at 0.75
    assert abs(s["Y"].mean().item() - 0.75) < 0.02


def test_env_step_returns_intuition_and_reward():
    env = MABUCEnv(seed=0)
    obs, _ = env.reset(seed=0)
    assert "intuition" in obs
    _, reward, terminated, _truncated, _ = env.step(int(obs["intuition"]))
    assert reward in (0.0, 1.0)
    assert terminated is True  # bandit: one-step episodes


def test_optimal_intuition_policy_beats_half():
    env = MABUCEnv(seed=1)
    total = 0.0
    n = 5000
    obs, _ = env.reset(seed=1)
    for _ in range(n):
        action = int(obs["intuition"])  # play arm == intuition (== lucky)
        _, r, _, _, _ = env.step(action)
        obs, _ = env.reset()
        total += r
    assert total / n > 0.7


def test_env_reward_matches_scm_interventional_distribution():
    # Guard against the env's step() logic drifting from its backing SCM: rolling out a
    # fixed arm should reproduce E[Y|do(X=a)] from the SCM (~0.5 for either arm).
    env = MABUCEnv(seed=4)
    scm_mean = env.scm.do({"X": 0.0}).see(20000, seed=4)["Y"].mean().item()
    n = 20000
    total = 0.0
    env.reset(seed=4)
    for _ in range(n):
        _, r, _, _, _ = env.step(0)
        env.reset()
        total += r
    env_mean = total / n
    assert abs(env_mean - scm_mean) < 0.02
    assert abs(env_mean - 0.5) < 0.02
