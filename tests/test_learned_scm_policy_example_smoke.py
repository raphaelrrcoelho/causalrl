"""Smoke test for examples/learned_scm_policy.py.

Loads the example as a module and runs ``main()`` on a tiny budget, so the whole
fit -> world-model env -> plan -> score path executes without the full run's cost. The
assertions pin the example's *point*: on a confounded log the two world models disagree about
which arm is best, and only the causal one agrees with the truth -- and, separately, that the
reported score is the chosen arm's value in the TRUE world rather than in the model that chose it.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from causalrl.envs.suite.simpson_bandit import SimpsonBandit


def _true_arm_values() -> tuple[float, ...]:
    """SimpsonBandit's exact ground truth for the example's three arms.

    ``do(A=0)`` = 0.40 and ``do(A=1)`` = 0.50 are closed-form; the "observe" arm is ``E[Y]`` under
    the logging policy (0.45 analytically), recomputed here from the *same* seeded 200k draw the
    example uses so the comparison below is exact rather than tolerance-bounded. Every score the
    example reports must land on this grid, because it scores the chosen arm in the TRUE world --
    never in the model that chose it.
    """
    bandit = SimpsonBandit(seed=0)
    status_quo = float(np.asarray(bandit.sample(200_000, seed=99)["Y"], dtype=float).mean())
    return (bandit.true_action_value(0), status_quo, bandit.true_action_value(1))


def _load_example_module() -> Any:
    """Load examples/learned_scm_policy.py without executing its __main__ block."""
    example_path = Path(__file__).parent.parent / "examples" / "learned_scm_policy.py"
    spec = importlib.util.spec_from_file_location("learned_scm_policy", example_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_learned_scm_policy_example_smoke() -> None:
    module = _load_example_module()
    results = module.main(n_logs=4_000, steps=40, seeds=(0,), n_mc=400)

    assert len(results) == 2
    confounded, randomized = results
    assert confounded.regime == "confounded log"
    assert randomized.regime.startswith("randomized log")

    # The logs set the trap: the naive conditional prefers action 0.
    assert confounded.empirical_conditional[0] > confounded.empirical_conditional[1]

    causal, blind = confounded.models
    # sorted(): reward_parents comes from networkx predecessor order, which is insertion order and
    # not a documented contract -- the same fix tests/test_learned_scm_as_env.py already applies.
    assert sorted(causal.reward_parents) == ["A", "Z"]
    assert sorted(blind.reward_parents) == ["A"]
    # Same rows, same fitter -- opposite verdicts, because of structure alone.
    assert causal.arm_values[2] > causal.arm_values[1]  # do(A=1) beats do(A=0): correct
    assert blind.arm_values[1] > blind.arm_values[2]  # ...and the blind model reverses it

    # On a randomized log the trap is gone and both models get the ordering right.
    for model in randomized.models:
        assert model.arm_values[2] > model.arm_values[1]

    # The example's headline claim: policies are scored in the TRUE world, never in the model that
    # chose them. Pin the reported value to the ground-truth grid rather than to a particular arm
    # -- 40 Thompson steps is too small a budget to fix which arm gets picked, but whichever one it
    # is, its true value is 0.40, 0.45 or 0.50 and its regret is in [0, 0.10]. Scoring in-model
    # instead puts the value off the grid and can make regret negative, which is impossible against
    # the true optimum.
    true_values = _true_arm_values()
    optimal = max(true_values)
    for regime in (confounded, randomized):
        for model in regime.models:
            # The comparison below is exact (not approximate) because the smoke budget runs a
            # single seed, so the mean is one arm's value; this len() keeps that premise honest.
            assert len(model.chosen_arms) == 1
            assert model.chosen_arms[0] in {"observe", "do(A=0)", "do(A=1)"}
            nearest = min(abs(model.mean_true_value - value) for value in true_values)
            assert nearest < 1e-9, (regime.regime, model.name, model.mean_true_value)
            assert 0.0 <= model.mean_regret <= 0.10, (regime.regime, model.name, model.mean_regret)
            # ...and regret is measured against the true optimum, not the model's own best arm.
            assert model.mean_regret == pytest.approx(optimal - model.mean_true_value)
