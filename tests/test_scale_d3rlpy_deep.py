"""Phase 4 §10: deepened d3rlpy bridge — both directions, FQE certificate, policy handle, unified.

d3rlpy is not installed in CI; a fake ``MDPDataset`` / policy drive the lazy paths deterministically
(the bridge reads plain arrays; only ``to_mdp_dataset`` imports d3rlpy).
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

from causalrl.certify.adapters import as_certificate
from causalrl.certify.certificate import Certificate, Kind
from causalrl.data.dataset import ConfoundedTrajectoryDataset, Transition
from causalrl.scale import certify_policy
from causalrl.scale.d3rlpy import (
    certify_fqe,
    mdp_dataset_to_confounded_dataset,
    mdp_dataset_to_trajectory_log,
    policy_actions,
    to_mdp_dataset,
    trajectory_log_to_mdp_dataset,
)


class _FakeMDPDataset:
    def __init__(self, observations, actions, rewards, terminals) -> None:
        self.observations = observations
        self.actions = actions
        self.rewards = rewards
        self.terminals = terminals


class _FakeD3rlpy:
    class dataset:
        MDPDataset = _FakeMDPDataset


class _FakePolicy:
    """A greedy 'always match the state index mod n_actions' policy with a d3rlpy-style predict."""

    def predict(self, observations: np.ndarray) -> np.ndarray:
        obs = np.asarray(observations)
        return obs.argmax(axis=1) % 2


def _dataset() -> ConfoundedTrajectoryDataset:
    trs = [
        Transition(0, 1, 1.0, 1, False),
        Transition(1, 0, 0.5, 0, True),
        Transition(0, 1, 0.2, 1, True),
    ]
    return ConfoundedTrajectoryDataset(trs, n_states=2, n_actions=2)


def test_mdp_dataset_round_trip_recovers_states_actions_rewards(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "d3rlpy", _FakeD3rlpy)
    ds = _dataset()
    mdp = to_mdp_dataset(ds)
    back = mdp_dataset_to_confounded_dataset(mdp)
    assert back.n_states == 2 and back.n_actions == 2
    assert [t.state for t in back.transitions] == [t.state for t in ds.transitions]
    assert [t.action for t in back.transitions] == [t.action for t in ds.transitions]
    # d3rlpy stores rewards as float32, so the round-trip is exact only to float32 precision.
    assert [t.reward for t in back.transitions] == pytest.approx(
        [t.reward for t in ds.transitions], abs=1e-6
    )
    assert [t.done for t in back.transitions] == [t.done for t in ds.transitions]


def test_trajectory_log_both_directions(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "d3rlpy", _FakeD3rlpy)
    from causalrl.data.trajectory import TrajectoryLog

    log = TrajectoryLog.from_confounded_dataset(_dataset())
    mdp = trajectory_log_to_mdp_dataset(log)
    assert mdp.observations.shape == (3, 2)
    log2 = mdp_dataset_to_trajectory_log(mdp)
    # Round-trips back to a log carrying the same per-transition states/actions/rewards.
    ds2 = log2.to_confounded_dataset()
    assert [t.state for t in ds2.transitions] == [0, 1, 0]
    assert [t.reward for t in ds2.transitions] == pytest.approx([1.0, 0.5, 0.2], abs=1e-6)


def test_policy_actions_is_a_do_handle() -> None:
    observations = np.eye(2)[[0, 1, 0]]  # one-hot states 0, 1, 0
    actions = policy_actions(_FakePolicy(), observations)
    assert actions == [0, 1, 0]
    assert all(isinstance(a, int) for a in actions)


def test_certify_fqe_is_empirical() -> None:
    cert = certify_fqe(3.14, policy="learned")
    assert isinstance(cert, Certificate)
    assert cert.kind is Kind.EMPIRICAL  # model-based OPE: no identification claim
    assert cert.value == 3.14
    assert cert.estimand.query == "policy_value" and cert.estimand.policy == "learned"
    assert cert.assumptions[0].name == "fqe-model"


def test_certify_policy_retargets_onto_unified_certificate() -> None:
    ds = _dataset()
    target = [1, 1, 1]
    decision = certify_policy(ds, target)
    unified = as_certificate(decision)  # adapter, no behaviour change (plan §10)
    assert isinstance(unified, Certificate)
    assert unified.kind is Kind.BOUNDED  # MSM sensitivity layer
    # The adapter is a view: the shipped DecisionCertificate is unchanged (no behaviour change).
    assert isinstance(decision.certified, bool)
    assert unified.value is not None or unified.hedge is not None
