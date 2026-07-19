"""ContinuousConfoundedBandit: oracle correctness for the M3 function-approximation tier."""

from __future__ import annotations

import numpy as np
import pytest

from causalrl.envs.suite.continuous_confounded import ContinuousConfoundedBandit
from causalrl.identification.criteria import backdoor_adjustment_set


def test_safe_arm_is_optimal_and_arm1_is_a_thin_bump() -> None:
    env = ContinuousConfoundedBandit(gamma=1.0)
    assert env.true_action_value(0) == pytest.approx(0.5, abs=1e-6)
    # The narrow high-z bump averages well below 0.5 over Uniform(0, 1).
    assert env.true_action_value(1) == pytest.approx(0.381, abs=0.01)
    assert env.optimal_action() == 0
    assert env.optimal_value() == pytest.approx(0.5, abs=1e-6)


def test_backdoor_set_is_the_single_continuous_confounder() -> None:
    env = ContinuousConfoundedBandit()
    assert set(env.graph.nodes) == {"Z", "A", "Y"}
    assert set(backdoor_adjustment_set(env.graph, "A", "Y")) == {"Z"}


def test_confounded_marginal_is_fooled_at_high_gamma() -> None:
    env = ContinuousConfoundedBandit(gamma=1.0, seed=0)
    data = env.sample(50_000, seed=0)
    a, y = data["A"], data["Y"]
    # The behavior over-samples arm 1 near the bump, so its confounded marginal exceeds arm 0's 0.5.
    assert y[a == 1].mean() > 0.5
    assert y[a == 0].mean() == pytest.approx(0.5, abs=0.02)


def test_sample_columns_and_continuity() -> None:
    env = ContinuousConfoundedBandit(gamma=0.5, seed=1)
    data = env.sample(1000, seed=1)
    assert set(data) == {"Z", "A", "Y"}
    assert data["Z"].min() >= 0.0
    assert data["Z"].max() <= 1.0
    assert set(np.unique(data["A"]).tolist()) <= {0, 1}


def test_rejects_out_of_range_gamma() -> None:
    with pytest.raises(ValueError):
        ContinuousConfoundedBandit(gamma=1.5)
