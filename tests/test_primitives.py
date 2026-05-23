from causalrl.agents.primitives import bounds_table, non_dominated_actions
from causalrl.data.dataset import ConfoundedTrajectoryDataset, Transition


def make_dataset() -> ConfoundedTrajectoryDataset:
    transitions = [
        Transition(0, 0, 1.0, 1, True),
        Transition(0, 0, 1.0, 1, True),
        Transition(0, 0, 1.0, 1, True),  # action 0: m=1, p=0.75 -> [0.75, 1.0]
        Transition(0, 1, 0.0, 1, True),  # action 1: m=0, p=0.25 -> [0.0, 0.75]
    ]
    return ConfoundedTrajectoryDataset(transitions, n_states=2, n_actions=2)


def test_bounds_table_shape():
    d = make_dataset()
    table = bounds_table(d)
    assert set(table.keys()) == {(0, 0), (0, 1), (1, 0), (1, 1)}
    assert abs(table[(0, 0)][0] - 0.75) < 1e-9


def test_non_dominated_keeps_strong_action():
    d = make_dataset()
    survivors = non_dominated_actions(d, state=0)
    assert 0 in survivors  # the strong action always survives


def test_non_dominated_returns_subset_of_actions():
    d = make_dataset()
    survivors = non_dominated_actions(d, state=0)
    assert set(survivors).issubset({0, 1})
    assert len(survivors) >= 1  # never empties the action set
