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
