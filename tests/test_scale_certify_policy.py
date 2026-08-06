"""Tests for certify_policy — certify a learned policy's improvement over the behaviour policy
against hidden confounding. No d3rlpy needed: the learned policy is supplied as its chosen actions.
"""

from __future__ import annotations

import pytest

from causalrl import certify_policy
from causalrl.data.dataset import ConfoundedTrajectoryDataset, Transition


def _uniform_bandit_dataset() -> ConfoundedTrajectoryDataset:
    # One state, two actions logged 50/50; action 1 pays 1.0, action 0 pays 0.0.
    trs: list[Transition] = []
    for _ in range(50):
        trs.append(Transition(0, 1, 1.0, 0, True))
        trs.append(Transition(0, 0, 0.0, 0, True))
    return ConfoundedTrajectoryDataset(trs, n_states=1, n_actions=2)


def test_certify_policy_prefers_improving_policy():
    ds = _uniform_bandit_dataset()
    target_actions = [1] * len(ds)  # always take the paying action
    cert = certify_policy(ds, target_actions, gamma_max=20.0)
    assert cert.decision == "prefer learned policy"
    assert cert.naive_contrast > 0
    assert cert.recommendation in {"act", "abstain"}


def test_certify_policy_indifferent_when_policy_mimics_behavior():
    ds = _uniform_bandit_dataset()
    target_actions = [tr.action for tr in ds.transitions]  # mimic the logged action
    cert = certify_policy(ds, target_actions)
    assert cert.naive_contrast == pytest.approx(0.0, abs=1e-9)


def test_length_mismatch_raises():
    ds = _uniform_bandit_dataset()
    with pytest.raises(ValueError, match="one action per logged transition"):
        certify_policy(ds, [1, 0])


def _risky_improvement_dataset() -> ConfoundedTrajectoryDataset:
    """One state, 50/50 logging. Action 0 always returns 0.5; action 1 returns 1.5 on 95% of its
    plays and -5.0 on the rest — a better mean (1.175 vs 0.5) with a far worse downside."""
    trs = [Transition(0, 0, 0.5, 0, True) for _ in range(1000)]
    trs += [Transition(0, 1, 1.5, 0, True) for _ in range(950)]
    trs += [Transition(0, 1, -5.0, 0, True) for _ in range(50)]
    return ConfoundedTrajectoryDataset(trs, n_states=1, n_actions=2)


def test_lcb_gate_refuses_a_mean_improving_policy_with_a_worse_downside():
    """The gate is not a restatement of the confounding verdict: the same policy that certifies
    against confounding is refused because one decision under it can return -5.0, where the
    logging policy's calibrated downside is 0.5."""
    ds = _risky_improvement_dataset()
    target_actions = [1] * len(ds)
    confounding_only = certify_policy(ds, target_actions, gamma_max=1.02)
    assert confounding_only.certified is True
    assert confounding_only.conformal_lcb is None  # the gate did not run

    gated = certify_policy(ds, target_actions, gamma_max=1.02, alpha=0.1)
    assert gated.naive_contrast == confounding_only.naive_contrast  # same contrast, same decision
    assert gated.conformal_lcb == pytest.approx(-5.0)
    assert gated.certified is False
    assert gated.recommendation == "abstain"
    assert "REFUSE" in gated.summary


def test_lcb_gate_passes_a_policy_that_dominates_the_logs():
    """And it does pass when the downside really does improve: the paying arm's calibrated lower
    bound is 1.0 against the logged mixture's 0.0."""
    ds = _uniform_bandit_dataset()
    gated = certify_policy(ds, [1] * len(ds), gamma_max=20.0, alpha=0.1)
    assert gated.conformal_lcb == pytest.approx(1.0)
    assert gated.certified is certify_policy(ds, [1] * len(ds), gamma_max=20.0).certified
    assert "PASS" in gated.summary


def test_lcb_gate_refuses_when_the_log_is_too_small_to_calibrate():
    """No evidence is not evidence of safety: an uninformative (-inf) bound must refuse."""
    trs = [Transition(0, 1, 1.0, 0, True) for _ in range(4)]
    trs += [Transition(0, 0, 0.0, 0, True) for _ in range(4)]
    ds = ConfoundedTrajectoryDataset(trs, n_states=1, n_actions=2)
    gated = certify_policy(ds, [1] * len(ds), gamma_max=20.0, alpha=0.1)
    assert gated.conformal_lcb == float("-inf")
    assert gated.certified is False
