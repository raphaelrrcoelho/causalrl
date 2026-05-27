from causalrl.envs.suite.scbandit import enumerate_arms, make_confounded_chain_env


def test_enumerate_arms_counts_all_interventions():
    arms = enumerate_arms(["X1", "X2", "X3"], {"X1": [0, 1], "X2": [0, 1], "X3": [0, 1]})
    assert arms[0] == {}                       # the observational arm comes first
    assert len(arms) == 27                     # 3^3 for three binary variables
    assert {"X3": 1} in arms
    assert {"X1": 0, "X2": 1, "X3": 0} in arms


def test_env_exposes_admg_and_manipulable():
    env = make_confounded_chain_env(seed=0)
    assert set(env.graph.nodes) == {"X1", "X2", "X3", "Y"}   # abstracted ADMG (no U)
    assert env.graph.is_confounded("X1", "Y") is True
    assert env.manipulable == ["X1", "X2", "X3"]
    assert env.action_space.n == 27


def test_observational_arm_is_optimal_and_near_one():
    env = make_confounded_chain_env(seed=0)
    empty_index = env.arms.index({})
    assert env.arm_values[empty_index] > 0.95          # observe: X3==U always -> ~1.0
    assert abs(env.optimal_value - env.arm_values[empty_index]) < 1e-9
    # any do(X3=c) breaks the confounding-exploit -> ~0.5
    x3_index = env.arms.index({"X3": 1})
    assert abs(env.arm_values[x3_index] - 0.5) < 0.05


def test_step_returns_binary_reward():
    env = make_confounded_chain_env(seed=0)
    env.reset(seed=0)
    _obs, reward, terminated, truncated, info = env.step(env.arms.index({}))
    assert reward in (0.0, 1.0)
    assert terminated is True and truncated is False
    assert "optimal_value" in info
