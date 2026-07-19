"""Simpson's-paradox bandit: oracle interventional values and the naive-marginal reversal."""

from __future__ import annotations

import numpy as np
import pytest

from causalrl.envs.suite.simpson_bandit import SimpsonBandit


def test_interventional_oracle_prefers_action_1() -> None:
    env = SimpsonBandit(seed=0)
    assert env.true_action_value(0) == pytest.approx(0.40)
    assert env.true_action_value(1) == pytest.approx(0.50)
    assert env.optimal_value == pytest.approx(0.50)


def test_naive_marginal_reverses_to_action_0() -> None:
    # The confounded logs make E[Y|A=0] > E[Y|A=1] even though do(A=1) is truly better.
    env = SimpsonBandit(seed=1)
    data = env.sample(200_000, seed=1)
    a, y = data["A"], data["Y"]
    assert y[a == 0].mean() > y[a == 1].mean()


def test_sample_shapes_and_support() -> None:
    env = SimpsonBandit(seed=2)
    data = env.sample(500, seed=2)
    assert set(data) == {"Z", "A", "Y"}
    for key in ("Z", "A", "Y"):
        assert len(data[key]) == 500
    assert set(np.unique(data["A"]).tolist()) <= {0, 1}
    assert set(np.unique(data["Z"]).tolist()) <= {0, 1}
