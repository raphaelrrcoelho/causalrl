from causalrl.agents.baselines import NaiveOffline, OnlineOnlyUCB
from causalrl.data.dataset import ConfoundedTrajectoryDataset, Transition


def test_online_only_acts_in_range():
    agent = OnlineOnlyUCB(n_states=5, n_actions=2, seed=0)
    a = agent.act({"state": 0, "t": 0})
    assert a in (0, 1)
    agent.update({"state": 0, "t": 0}, a, reward=1.0)


def test_naive_offline_picks_empirical_best():
    transitions = [
        Transition(0, 0, 0.0, 1, True),
        Transition(0, 1, 1.0, 1, True),
        Transition(0, 1, 1.0, 1, True),
    ]
    d = ConfoundedTrajectoryDataset(transitions, n_states=2, n_actions=2)
    agent = NaiveOffline(n_states=2, n_actions=2)
    agent.ingest_offline(d)
    assert agent.act({"state": 0, "t": 0}) == 1
