import math

import numpy as np

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


def test_h1_optimistic_q_matches_v02_formula():
    # At horizon 1, optimistic_q must equal min(mean + ucb_bonus, ceiling) exactly (v0.2).
    a = DOVI(n_states=2, n_actions=2, horizon=1, seed=0)
    a._counts[0, 0] = 5.0
    a._sums[0, 0] = 3.0
    a._ceiling[0, 0] = 2.0  # high so the min() does not cap (tests the uncapped branch)
    a._t = 10
    a._q = None
    mean = 3.0 / 5.0
    bonus = math.sqrt(2.0 * math.log(10) / 5.0)
    assert abs(a.optimistic_q(0, 0) - (mean + bonus)) < 1e-9


def test_untried_optimism_scales_with_reward_max():
    a = DOVI(n_states=2, n_actions=2, horizon=1, seed=0, reward_max=3.0)
    a._ceiling[0, 0] = 10.0  # high so not capped
    a._q = None
    # Untried (count 0): r̃ = min(0 + reward_max, ceiling) = reward_max.
    assert abs(a.optimistic_q(0, 0) - 3.0) < 1e-9


def test_two_stage_backward_induction_matches_hand_computation():
    # States {0, 1} + terminal 2; horizon 2; deterministic transition (s, a) -> state a.
    a = DOVI(n_states=3, n_actions=2, horizon=2, seed=0)
    # Force r̃(s,a) == ceiling by saturating the online mean (mean=1.0 >= every ceiling).
    a._counts[:] = 1.0
    a._sums[:] = 1.0
    a._ceiling[:] = np.array([[0.5, 0.65], [0.3, 0.35], [0.0, 0.0]])
    a._trans[0, 0, 0] = 1.0
    a._trans_n[0, 0] = 1.0
    a._trans[0, 1, 1] = 1.0
    a._trans_n[0, 1] = 1.0
    a._trans[1, 0, 0] = 1.0
    a._trans_n[1, 0] = 1.0
    a._trans[1, 1, 1] = 1.0
    a._trans_n[1, 1] = 1.0
    a._q = None
    q = a._plan()
    # Last stage h=2: no bootstrap -> Q == r̃.
    assert abs(q[2, 0, 0] - 0.5) < 1e-9
    assert abs(q[2, 0, 1] - 0.65) < 1e-9
    # Stage h=1: Q = r̃ + V_2(next). V_2(0)=max(.5,.65)=.65 ; V_2(1)=max(.3,.35)=.35.
    assert abs(q[1, 0, 0] - 1.15) < 1e-9  # .5 + .65
    assert abs(q[1, 0, 1] - 1.00) < 1e-9  # .65 + .35
    assert abs(q[1, 1, 0] - 0.95) < 1e-9  # .3 + .65
    assert abs(q[1, 1, 1] - 0.70) < 1e-9  # .35 + .35


def test_done_transitions_do_not_bootstrap():
    # A (s,a) seen only as a terminal transition must bootstrap zero future value.
    a = DOVI(n_states=2, n_actions=1, horizon=3, seed=0)
    a._counts[:] = 1.0
    a._sums[:] = 1.0
    a._ceiling[:] = 0.4
    a.observe_transition(0, 0, 1, done=True)  # terminal: no entry in _trans, but _trans_n++
    a._q = None
    q = a._plan()
    # All stages: bootstrap is 0 (only a terminal transition observed) -> Q == r̃ == 0.4.
    assert abs(q[1, 0, 0] - 0.4) < 1e-9
    assert abs(q[3, 0, 0] - 0.4) < 1e-9
