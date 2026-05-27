from causalrl.envs.suite.scbandit import make_frontdoor_env


def test_frontdoor_env_structure():
    env = make_frontdoor_env(seed=0)
    assert set(env.graph.nodes) == {"X", "Z", "Y"}  # abstracted ADMG (latent U_XY dropped)
    assert env.graph.is_confounded("X", "Y") is True
    assert env.manipulable == ["X"]                 # Z is non-manipulable
    assert env.action_space.n == 3                  # {}, do(X=0), do(X=1)


def test_frontdoor_optimal_is_do_x1():
    env = make_frontdoor_env(seed=0)
    v_empty = env.arm_values[env.arms.index({})]
    v_x0 = env.arm_values[env.arms.index({"X": 0})]
    v_x1 = env.arm_values[env.arms.index({"X": 1})]
    assert abs(v_empty - 0.50) < 0.03   # observation
    assert abs(v_x0 - 0.44) < 0.03      # do(X=0)
    assert abs(v_x1 - 0.56) < 0.03      # do(X=1) -- optimal
    assert v_x1 > v_empty > v_x0
    assert abs(env.optimal_value - 0.56) < 0.03
