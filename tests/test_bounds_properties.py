import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from causalrl.data.dataset import ConfoundedTrajectoryDataset, Transition
from causalrl.identification.bounds import causal_q_bounds


@given(
    p_action0=st.floats(min_value=0.05, max_value=0.95),
    r0=st.floats(min_value=0.0, max_value=1.0),
    r1=st.floats(min_value=0.0, max_value=1.0),
    n=st.integers(min_value=200, max_value=600),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
@settings(max_examples=40, deadline=None)
def test_manski_bounds_bracket_true_do_effect(p_action0, r0, r1, n, seed):
    rng = np.random.default_rng(seed)

    def reward(action: int, u: int) -> float:
        return r0 if action == u else r1

    transitions = []
    for _ in range(n):
        u = int(rng.integers(0, 2))
        action = u if rng.random() < p_action0 else 1 - u
        y = float(rng.random() < reward(action, u))
        transitions.append(Transition(0, action, y, 1, True))
    d = ConfoundedTrajectoryDataset(transitions, n_states=2, n_actions=2)

    for a in (0, 1):
        true_do = 0.5 * reward(a, 0) + 0.5 * reward(a, 1)
        lo, hi = causal_q_bounds(d, state=0, action=a)
        assert lo - 0.1 <= true_do <= hi + 0.1
