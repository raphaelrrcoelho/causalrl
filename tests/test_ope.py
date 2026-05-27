import importlib

import causalrl
from causalrl.eval.ope import ipw_value


def test_ipw_recovers_value_with_known_propensities():
    # target policy always picks action 0; behavior picked 0 half the time with prob 0.5
    actions = [0, 1, 0, 1]
    rewards = [1.0, 0.0, 1.0, 0.0]
    behavior_probs = [0.5, 0.5, 0.5, 0.5]
    target_probs = [1.0, 0.0, 1.0, 0.0]
    v = ipw_value(actions, rewards, behavior_probs, target_probs)
    assert abs(v - 1.0) < 1e-9  # (1/0.5*1 + 0 + 1/0.5*1 + 0)/4 = 1.0


def test_sensitivity_bounds_widen_with_gamma():
    experimental_ope = importlib.import_module("causalrl.experimental.ope")
    lo1, hi1 = experimental_ope.confounding_sensitivity_bounds(point=0.5, gamma=1.0)
    assert lo1 == hi1 == 0.5
    lo2, hi2 = experimental_ope.confounding_sensitivity_bounds(point=0.5, gamma=2.0)
    assert lo2 < 0.5 < hi2


def test_sensitivity_bounds_are_not_stable_top_level_api():
    assert "confounding_sensitivity_bounds" not in causalrl.__all__
    assert not hasattr(causalrl, "confounding_sensitivity_bounds")
