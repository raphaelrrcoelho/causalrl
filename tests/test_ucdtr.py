import pytest

from causalrl.agents.offline_online import UCDTR
from causalrl.data.dataset import ConfoundedTrajectoryDataset, Transition
from causalrl.exceptions import NotIdentifiableError


def test_ucdtr_acts_in_range_without_offline_data():
    agent = UCDTR(n_states=3, n_actions=2, seed=0)
    a = agent.act({"state": 0, "t": 0})
    assert a in (0, 1)
    agent.update({"state": 0, "t": 0}, a, reward=1.0)


def test_ucdtr_ingest_keeps_all_actions_allowed_under_manski():
    # Manski natural bounds cannot strictly prune, so all actions remain allowed.
    transitions = [Transition(0, 0, 1.0, 1, True)] * 8 + [Transition(0, 1, 0.0, 1, True)] * 12
    d = ConfoundedTrajectoryDataset(transitions, n_states=2, n_actions=2)
    agent = UCDTR(n_states=2, n_actions=2, seed=0)
    agent.ingest_offline(d)
    actions = {agent.act({"state": 0, "t": 0}) for _ in range(50)}
    assert actions.issubset({0, 1})


def test_ucdtr_learns_better_arm_online_despite_biased_offline():
    # Offline logs make action 1 look great (confounded), but online action 0 truly pays.
    # UC-DTR must NOT lock into the biased offline policy — it learns action 0 online.
    biased = [Transition(0, 1, 1.0, 1, True)] * 10 + [Transition(0, 0, 0.0, 1, True)] * 2
    d = ConfoundedTrajectoryDataset(biased, n_states=2, n_actions=2)
    agent = UCDTR(n_states=2, n_actions=2, seed=0)
    agent.ingest_offline(d)
    for _ in range(300):
        s = {"state": 0, "t": 0}
        a = agent.act(s)
        agent.update(s, a, reward=1.0 if a == 0 else 0.0)
    votes = [agent.act({"state": 0, "t": 0}) for _ in range(200)]
    assert sum(v == 0 for v in votes) > 150


def test_ucdtr_strict_raises_on_unlogged_action():
    # state 0 only ever logs action 0; action 1 is never logged (vacuous bound).
    d = ConfoundedTrajectoryDataset([Transition(0, 0, 1.0, 1, True)], n_states=2, n_actions=2)
    agent = UCDTR(n_states=2, n_actions=2, seed=0, require_identified=True)
    with pytest.raises(NotIdentifiableError) as exc:
        agent.ingest_offline(d)
    assert exc.value.witness == (0, 1)
