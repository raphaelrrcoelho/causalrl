"""The three-arm counterfactual bandit: observational, interventional, and per-intent means."""

from __future__ import annotations

from causalrl.envs.suite.counterfactual_bandit import (
    build_counterfactual_scm,
    make_counterfactual_bandit_env,
)
from causalrl.identification.counterfactual import counterfactual_expectation

_DO_MEAN = (1 / 3) * 0.8 + (2 / 3) * 0.15  # = 0.36667, any fixed do(X=a)


def test_playing_intent_is_optimal() -> None:
    env = make_counterfactual_bandit_env(seed=0)
    total = 0.0
    n = 4000
    for t in range(n):
        obs, _ = env.reset(seed=t)
        _, reward, terminated, _, _ = env.step(obs["intuition"])  # play arm = intent
        assert terminated
        total += reward
    assert abs(total / n - 0.8) < 0.03


def test_fixed_arm_is_suboptimal() -> None:
    env = make_counterfactual_bandit_env(seed=1)
    total = 0.0
    n = 4000
    for t in range(n):
        env.reset(seed=1000 + t)
        _, reward, _, _, _ = env.step(0)  # always arm 0
        total += reward
    assert abs(total / n - _DO_MEAN) < 0.03


def test_scm_observational_and_interventional_means() -> None:
    scm = build_counterfactual_scm()
    # Behavior policy X = I = U is implicitly optimal: observational mean is 0.8.
    observational = scm.see(60_000, seed=0)["Y"].float().mean().item()
    assert abs(observational - 0.8) < 0.02
    # Every fixed intervention severs X from U and averages ~0.367.
    for arm in (0, 1, 2):
        do_mean = counterfactual_expectation(
            scm, outcome="Y", intervention={"X": float(arm)}, evidence={}, n=60_000, seed=arm
        )
        assert abs(do_mean - _DO_MEAN) < 0.02


def test_per_intent_counterfactual_reward() -> None:
    scm = build_counterfactual_scm()
    for intent in (0, 1, 2):
        for arm in (0, 1, 2):
            value = counterfactual_expectation(
                scm,
                outcome="Y",
                intervention={"X": float(arm)},
                evidence={"I": float(intent)},
                n=40_000,
                seed=10 * intent + arm,
            )
            expected = 0.8 if arm == intent else 0.15
            assert abs(value - expected) < 0.03
