import pytest

from causalrl.data.dataset import ConfoundedTrajectoryDataset, Transition
from causalrl.exceptions import NotIdentifiableError
from causalrl.identification.bounds import causal_q_bounds


def make_dataset() -> ConfoundedTrajectoryDataset:
    transitions = [
        Transition(0, 0, 1.0, 1, True),
        Transition(0, 0, 1.0, 1, True),
        Transition(0, 0, 0.0, 1, True),
        Transition(0, 1, 0.0, 1, True),
    ]
    return ConfoundedTrajectoryDataset(transitions, n_states=2, n_actions=2)


def test_bounds_formula():
    d = make_dataset()
    # action 0: m = 2/3, p = 3/4 -> lower = m*p = 0.5, upper = 0.5 + (1-0.75) = 0.75
    lo, hi = causal_q_bounds(d, state=0, action=0)
    assert abs(lo - 0.5) < 1e-9
    assert abs(hi - 0.75) < 1e-9


def test_bounds_widen_for_rare_action():
    d = make_dataset()
    # action 1: m = 0, p = 1/4 -> lower = 0, upper = 0 + 0.75 = 0.75
    lo, hi = causal_q_bounds(d, state=0, action=1)
    assert abs(lo - 0.0) < 1e-9
    assert abs(hi - 0.75) < 1e-9


def test_vacuous_bound_is_full_interval():
    d = make_dataset()
    lo, hi = causal_q_bounds(d, state=1, action=0)  # never logged
    assert (lo, hi) == (0.0, 1.0)


def test_require_identified_raises_on_vacuous():
    d = make_dataset()
    with pytest.raises(NotIdentifiableError) as exc:
        causal_q_bounds(d, state=1, action=0, require_identified=True)
    assert exc.value.witness == (1, 0)
