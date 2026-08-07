from causalrl.ope.ipw import ipw_value


def test_ipw_recovers_value_with_known_propensities():
    # target policy always picks action 0; behavior picked 0 half the time with prob 0.5
    actions = [0, 1, 0, 1]
    rewards = [1.0, 0.0, 1.0, 0.0]
    behavior_probs = [0.5, 0.5, 0.5, 0.5]
    target_probs = [1.0, 0.0, 1.0, 0.0]
    v = ipw_value(actions, rewards, behavior_probs, target_probs)
    assert abs(v - 1.0) < 1e-9  # (1/0.5*1 + 0 + 1/0.5*1 + 0)/4 = 1.0
