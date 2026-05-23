from causalrl.data.dataset import ConfoundedTrajectoryDataset, Transition, generate_logs
from causalrl.envs.suite.dtr import DTREnv


def make_dataset() -> ConfoundedTrajectoryDataset:
    # state 0: action 0 taken 3x (rewards 1,1,0), action 1 taken 1x (reward 0)
    transitions = [
        Transition(state=0, action=0, reward=1.0, next_state=1, done=True),
        Transition(state=0, action=0, reward=1.0, next_state=1, done=True),
        Transition(state=0, action=0, reward=0.0, next_state=1, done=True),
        Transition(state=0, action=1, reward=0.0, next_state=1, done=True),
    ]
    return ConfoundedTrajectoryDataset(transitions, n_states=2, n_actions=2)


def test_len_and_shape():
    d = make_dataset()
    assert len(d) == 4
    assert d.n_states == 2
    assert d.n_actions == 2


def test_behavior_propensity():
    d = make_dataset()
    assert d.behavior_propensity(0, 0) == 0.75  # 3 of 4 at state 0
    assert d.behavior_propensity(0, 1) == 0.25
    assert d.behavior_propensity(1, 0) == 0.0  # state 1 never a decision point here


def test_mean_reward():
    d = make_dataset()
    assert abs(d.mean_reward(0, 0) - (2.0 / 3.0)) < 1e-9
    assert d.mean_reward(0, 1) == 0.0
    assert d.mean_reward(1, 0) == 0.0  # no data -> 0.0 by convention


def test_generate_logs_shapes():
    env = DTREnv(seed=0)
    d = generate_logs(env, n_episodes=100, seed=0)
    # 2 transitions per episode (stage 0 -> stage 1 -> terminal)
    assert len(d) == 200
    assert d.n_states == 5
    assert d.n_actions == 2
    # behavior policy ties action to U, so stage-0 propensity is strictly between 0 and 1
    assert 0.0 < d.behavior_propensity(0, 0) < 1.0
