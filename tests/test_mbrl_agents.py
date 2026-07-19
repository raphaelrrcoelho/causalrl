"""CertifiedPolicyAgent: certify-gated policy selection with abstention to behavior."""

from __future__ import annotations

from causalrl.agents.mbrl import CertifiedPolicyAgent
from causalrl.data.dataset import ConfoundedTrajectoryDataset, Transition


def _clean_improvement_dataset() -> ConfoundedTrajectoryDataset:
    # Unconfounded 50/50 logs, single context: action 1 pays 1.0, action 0 pays 0.0.
    trs: list[Transition] = []
    for _ in range(100):
        trs.append(Transition(0, 1, 1.0, 0, True))
        trs.append(Transition(0, 0, 0.0, 0, True))
    return ConfoundedTrajectoryDataset(trs, n_states=1, n_actions=2)


def test_certifies_and_ships_the_improving_action() -> None:
    # Low gamma_max: only modest confounding need be ruled out, so a clean strong improvement
    # certifies and is shipped.
    agent = CertifiedPolicyAgent(n_states=1, n_actions=2, gamma_max=1.2)
    agent.ingest_offline(_clean_improvement_dataset())
    assert agent.policy == [1]
    assert agent.act({"state": 0}) == 1


def test_abstains_to_behavior_when_nothing_certifies() -> None:
    # Behavior overwhelmingly plays action 0; the single action-1 sample cannot certify.
    trs = [Transition(0, 0, 0.0, 0, True) for _ in range(99)]
    trs.append(Transition(0, 1, 1.0, 0, True))
    ds = ConfoundedTrajectoryDataset(trs, n_states=1, n_actions=2)
    agent = CertifiedPolicyAgent(n_states=1, n_actions=2, gamma_max=1.5)
    agent.ingest_offline(ds)
    assert agent.policy == [0]  # abstains to the empirical behavior action


def test_policy_has_one_action_per_state() -> None:
    trs = [Transition(s % 2, 1, 1.0, 0, True) for s in range(40)]
    ds = ConfoundedTrajectoryDataset(trs, n_states=2, n_actions=2)
    agent = CertifiedPolicyAgent(n_states=2, n_actions=2)
    agent.ingest_offline(ds)
    assert len(agent.policy) == 2
    assert all(a in (0, 1) for a in agent.policy)
