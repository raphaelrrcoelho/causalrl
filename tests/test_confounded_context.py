"""Oracle confounded contextual bandit (causal-MBRL M0 kill-gate substrate)."""

from __future__ import annotations

import pytest

from causalrl.envs.suite.confounded_context import (
    ConfoundedContextualBandit,
    make_confounded_context_env,
)


def test_oracle_optimal_policy_value_source() -> None:
    env = make_confounded_context_env(gamma=0.9, shift=False, seed=0)
    # Optimal policy plays a == c in each context: 0.05 + 0.20 + 0.30 = 0.55.
    assert env.true_policy_value([0, 1]) == pytest.approx(0.55)
    # The confounder-aligned "always action 1" policy is worse in expectation.
    assert env.true_policy_value([1, 1]) < env.true_policy_value([0, 1])


def test_shift_flips_the_optimal_action() -> None:
    src = make_confounded_context_env(shift=False, seed=0)
    tgt = make_confounded_context_env(shift=True, seed=0)
    assert src.true_policy_value([0, 1]) == pytest.approx(0.55)
    assert tgt.true_policy_value([1, 0]) == pytest.approx(0.55)
    assert tgt.true_policy_value([0, 1]) < tgt.true_policy_value([1, 0])


def test_behavior_policy_tracks_the_confounder() -> None:
    env = make_confounded_context_env(gamma=1.0, seed=1)
    obs, _ = env.reset(seed=1)
    # With gamma=1 the logged action equals the (hidden) confounder u for that episode.
    assert env.behavior_policy(obs) == env._u  # noqa: SLF001 - white-box check of confounding


def test_env_exposes_scm_and_shapes() -> None:
    env = ConfoundedContextualBandit(seed=0)
    assert env.n_states == 2
    assert env.n_actions == 2
    assert env.scm is not None
    assert "Y" in env.scm.graph.nodes
