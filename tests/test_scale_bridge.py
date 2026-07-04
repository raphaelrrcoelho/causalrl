"""Tests for the causalrl -> d3rlpy dataset bridge.

d3rlpy is not installed in CI, so a fake d3rlpy is injected into sys.modules to exercise the lazy
import and MDPDataset construction deterministically.
"""

from __future__ import annotations

import sys

from causalrl.data.dataset import ConfoundedTrajectoryDataset, Transition


class _FakeMDPDataset:
    def __init__(self, observations, actions, rewards, terminals):
        self.observations = observations
        self.actions = actions
        self.rewards = rewards
        self.terminals = terminals


class _FakeD3rlpy:
    class dataset:
        MDPDataset = _FakeMDPDataset


def test_to_mdp_dataset_one_hots_states(monkeypatch):
    monkeypatch.setitem(sys.modules, "d3rlpy", _FakeD3rlpy)
    from causalrl.scale.d3rlpy import to_mdp_dataset

    trs = [Transition(0, 1, 1.0, 1, False), Transition(1, 0, 0.5, 1, True)]
    ds = ConfoundedTrajectoryDataset(trs, n_states=2, n_actions=2)
    mdp = to_mdp_dataset(ds)
    assert mdp.observations.shape == (2, 2)
    assert mdp.observations[0].tolist() == [1.0, 0.0]  # state 0 one-hot
    assert mdp.observations[1].tolist() == [0.0, 1.0]  # state 1 one-hot
    assert mdp.actions.tolist() == [1, 0]
    assert mdp.rewards.tolist() == [1.0, 0.5]
    assert mdp.terminals.tolist() == [0.0, 1.0]
