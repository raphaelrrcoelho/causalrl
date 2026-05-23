from causalrl.agents.dovi import DOVI
from causalrl.data.dataset import ConfoundedTrajectoryDataset, Transition


def test_optimistic_q_respects_upper_bound_after_ingest():
    transitions = [
        Transition(0, 0, 1.0, 2, True),
        Transition(0, 1, 0.0, 2, True),
    ]
    d = ConfoundedTrajectoryDataset(transitions, n_states=3, n_actions=2)
    agent = DOVI(n_states=3, n_actions=2, horizon=1, seed=0)
    agent.ingest_offline(d)
    a = agent.act({"state": 0, "t": 0})
    assert a in (0, 1)
    assert agent.optimistic_q(0, 0) <= 1.0 + 1e-9


def test_dovi_acts_and_updates():
    agent = DOVI(n_states=3, n_actions=2, horizon=1, seed=0)
    a = agent.act({"state": 0, "t": 0})
    agent.update({"state": 0, "t": 0}, a, reward=1.0)
    assert a in (0, 1)


def test_ceiling_caps_optimism():
    # An action with a tight low upper bound must have its optimistic Q capped at that ceiling.
    transitions = [Transition(0, 1, 0.0, 2, True)] * 10 + [Transition(0, 0, 1.0, 2, True)]
    d = ConfoundedTrajectoryDataset(transitions, n_states=3, n_actions=2)
    agent = DOVI(n_states=3, n_actions=2, horizon=1, seed=0)
    agent.ingest_offline(d)
    # action 1: m=0, p~0.91 -> upper ~ 0.09; optimism for the untried action is capped there
    assert agent.optimistic_q(0, 1) <= 0.15
