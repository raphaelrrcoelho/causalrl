"""FittedQIteration — continuous-state backward induction, and what it does and does not license."""

import numpy as np
import pytest

from causalrl.agents.fitted import FittedQIteration
from causalrl.certify.certificate import Kind
from causalrl.state import FeatureTransition, IdentityEncoder, OneHotEncoder, RBFEncoder


def _one_hot(index: int, size: int) -> np.ndarray:
    vector = np.zeros(size)
    vector[index] = 1.0
    return vector


def _tabular_dataset(
    n_states: int = 4, n_actions: int = 3, per_cell: int = 60, seed: int = 0
) -> tuple[list[FeatureTransition], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """A random finite MDP with every (state, action) visited, as feature transitions."""
    rng = np.random.default_rng(seed)
    kernel = rng.dirichlet(np.ones(n_states), size=(n_states, n_actions))
    payoff = rng.uniform(0.0, 1.0, size=(n_states, n_actions))
    states, actions, rewards, nexts = [], [], [], []
    for state in range(n_states):
        for action in range(n_actions):
            for _ in range(per_cell):
                successor = int(rng.choice(n_states, p=kernel[state, action]))
                states.append(state)
                actions.append(action)
                rewards.append(float(payoff[state, action] + rng.normal(0.0, 0.01)))
                nexts.append(successor)
    transitions = [
        FeatureTransition(_one_hot(s, n_states), a, r, _one_hot(n, n_states), False)
        for s, a, r, n in zip(states, actions, rewards, nexts, strict=True)
    ]
    return (
        transitions,
        np.array(states),
        np.array(actions),
        np.array(rewards),
        np.array(nexts),
    )


def _tabular_reference(
    states: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    nexts: np.ndarray,
    *,
    n_states: int,
    n_actions: int,
    horizon: int,
    reward_max: float = 1.0,
) -> np.ndarray:
    """The same recursion done with table lookups, as FittedQIteration's docstring claims."""
    q = np.zeros((horizon + 2, n_states, n_actions))
    value = np.zeros((horizon + 2, n_states))
    for step in range(horizon, 0, -1):
        cap = reward_max * (horizon - step + 1)
        for state in range(n_states):
            for action in range(n_actions):
                rows = (states == state) & (actions == action)
                targets = np.minimum(rewards[rows] + value[step + 1][nexts[rows]], cap)
                q[step, state, action] = targets.mean()
        q[step] = np.minimum(q[step], cap)
        value[step] = q[step].max(axis=1)
    return q


def test_one_hot_features_reproduce_the_tabular_backup() -> None:
    """The faithfulness property: the tabular agent is a special case, not a separate algorithm.

    An indicator basis spans every function on a finite state set, so a least-squares fit over
    one-hot features recovers the per-cell empirical means the tabular backup uses. If this drifts,
    the continuous generalisation has stopped containing the case it generalises.
    """
    n_states, n_actions, horizon = 4, 3, 3
    transitions, states, actions, rewards, nexts = _tabular_dataset(n_states, n_actions)
    agent = FittedQIteration(n_actions, horizon, OneHotEncoder(n_states), seed=0).fit(transitions)
    reference = _tabular_reference(
        states,
        actions,
        rewards,
        nexts,
        n_states=n_states,
        n_actions=n_actions,
        horizon=horizon,
    )
    for step in range(1, horizon + 1):
        for state in range(n_states):
            got = agent.q_values({"state": state}, step)
            assert got == pytest.approx(reference[step, state], abs=1e-6)


def test_one_hot_features_reproduce_the_tabular_policy() -> None:
    n_states, n_actions, horizon = 4, 3, 3
    transitions, states, actions, rewards, nexts = _tabular_dataset(n_states, n_actions)
    agent = FittedQIteration(n_actions, horizon, OneHotEncoder(n_states), seed=0).fit(transitions)
    reference = _tabular_reference(
        states,
        actions,
        rewards,
        nexts,
        n_states=n_states,
        n_actions=n_actions,
        horizon=horizon,
    )
    for step in range(1, horizon + 1):
        for state in range(n_states):
            best = int(np.argmax(agent.q_values({"state": state}, step)))
            assert best == int(np.argmax(reference[step, state]))


def _threshold_dataset(n: int = 1500, seed: int = 1) -> tuple[list[FeatureTransition], RBFEncoder]:
    """Continuous ``x``; action 1 pays ``0.8x`` and action 0 pays ``0.8(1-x)``.

    The optimal action flips at ``x = 0.5`` — a policy no single stratum can express.
    """
    rng = np.random.default_rng(seed)
    encoder = RBFEncoder(
        IdentityEncoder(["x"]), np.linspace(0.0, 1.0, 12).reshape(-1, 1), bandwidth=0.15
    )
    transitions = []
    for _ in range(n):
        x = float(rng.uniform())
        action = int(rng.integers(0, 2))
        reward = (0.8 * x if action == 1 else 0.8 * (1.0 - x)) + float(rng.normal(0.0, 0.02))
        features = encoder.encode({"x": x})
        transitions.append(FeatureTransition(features, action, reward, features, True))
    return transitions, encoder


def test_a_continuous_state_recovers_a_state_dependent_policy() -> None:
    transitions, encoder = _threshold_dataset()
    agent = FittedQIteration(2, 1, encoder, seed=0).fit(transitions)
    grid = np.linspace(0.02, 0.98, 25)
    chosen = [int(np.argmax(agent.q_values({"x": float(x)}))) for x in grid]
    expected = [1 if x > 0.5 else 0 for x in grid]
    assert chosen == expected


def test_collapsing_the_state_destroys_the_policy() -> None:
    # The control: the same data through a single-bucket (constant) encoder cannot represent the
    # threshold at all, so it commits to one action everywhere.
    transitions, _ = _threshold_dataset()
    flat = [
        FeatureTransition(np.array([1.0]), t.action, t.reward, np.array([1.0]), True)
        for t in transitions
    ]
    agent = FittedQIteration(2, 1, IdentityEncoder(["const"]), seed=0).fit(flat)
    chosen = {int(np.argmax(agent.q_values({"const": 1.0}))) for _ in range(5)}
    assert len(chosen) == 1


def test_the_cap_is_the_remaining_horizon_times_the_reward_bound() -> None:
    agent = FittedQIteration(2, 3, OneHotEncoder(2), reward_max=0.5)
    assert agent.cap(1) == pytest.approx(1.5)
    assert agent.cap(3) == pytest.approx(0.5)


def test_the_cap_refuses_a_step_outside_the_horizon() -> None:
    with pytest.raises(ValueError, match="must lie in"):
        FittedQIteration(2, 3, OneHotEncoder(2)).cap(4)


def test_q_values_never_exceed_the_cap() -> None:
    transitions, _, _, _, _ = _tabular_dataset()
    agent = FittedQIteration(3, 3, OneHotEncoder(4), reward_max=1.0, seed=0).fit(transitions)
    for step in range(1, 4):
        for state in range(4):
            assert agent.q_values({"state": state}, step).max() <= agent.cap(step) + 1e-9


def test_an_action_with_no_data_falls_back_to_the_cap() -> None:
    # Optimistic in the unexplored direction, matching how the tabular agent treats an
    # unvisited cell. Action 1 is never taken here.
    transitions = [
        FeatureTransition(_one_hot(0, 2), 0, 0.1, _one_hot(1, 2), True) for _ in range(20)
    ]
    agent = FittedQIteration(2, 1, OneHotEncoder(2), reward_max=1.0, seed=0).fit(transitions)
    assert agent.q_values({"state": 0})[1] == pytest.approx(agent.cap(1))


def test_acting_before_fitting_is_refused() -> None:
    agent = FittedQIteration(2, 1, OneHotEncoder(2))
    with pytest.raises(RuntimeError, match="no plan yet"):
        agent.act({"state": 0})


def test_fitting_on_nothing_is_refused() -> None:
    with pytest.raises(ValueError, match="no transitions"):
        FittedQIteration(2, 1, OneHotEncoder(2)).fit([])


def test_observe_accumulates_and_invalidates_the_plan() -> None:
    agent = FittedQIteration(2, 1, OneHotEncoder(2), seed=0)
    for _ in range(20):
        agent.observe(FeatureTransition(_one_hot(0, 2), 0, 0.5, _one_hot(1, 2), True))
        agent.observe(FeatureTransition(_one_hot(0, 2), 1, 0.1, _one_hot(1, 2), True))
    agent.fit()
    assert agent.act({"state": 0}) == 0
    with pytest.raises(RuntimeError, match="no plan yet"):
        agent.observe(FeatureTransition(_one_hot(0, 2), 1, 9.0, _one_hot(1, 2), True))
        agent.act({"state": 0})


def test_the_tabular_transition_hook_is_refused_rather_than_silently_dropped() -> None:
    # The seam this module exists to close: accepting an int-typed transition would look like it
    # worked and discard the data.
    agent = FittedQIteration(2, 1, OneHotEncoder(2))
    with pytest.raises(NotImplementedError, match="feature-typed transitions"):
        agent.observe_transition(0, 1, 1, False)


def test_update_is_a_documented_no_op() -> None:
    transitions, _, _, _, _ = _tabular_dataset()
    agent = FittedQIteration(3, 3, OneHotEncoder(4), seed=0).fit(transitions)
    before = agent.q_values({"state": 0})
    agent.update({"state": 0}, 0, 99.0)
    assert agent.q_values({"state": 0}) == pytest.approx(before)


def test_features_from_a_different_encoder_are_refused() -> None:
    transitions = [
        FeatureTransition(np.zeros(7), 0, 0.0, np.zeros(7), True),
    ]
    with pytest.raises(ValueError, match="different encoder"):
        FittedQIteration(2, 1, OneHotEncoder(3)).fit(transitions)


def test_an_out_of_range_action_is_refused() -> None:
    transitions = [FeatureTransition(_one_hot(0, 2), 5, 0.0, _one_hot(1, 2), True)]
    with pytest.raises(ValueError, match="outside"):
        FittedQIteration(2, 1, OneHotEncoder(2)).fit(transitions)


def test_the_certificate_is_empirical_and_names_the_downgrade() -> None:
    # The honesty requirement: a fitted backup must not inherit DOVI's bounded status.
    agent = FittedQIteration(2, 2, OneHotEncoder(2), reward_max=1.0)
    certificate = agent.certificate()
    assert certificate.kind is Kind.EMPIRICAL
    assert certificate.hedge is not None
    assert certificate.hedge.downgraded_from == "bounded"
    assert "Manski" in certificate.hedge.reason
    assert certificate.method == "fitted_q_iteration"
    assert {a.name for a in certificate.assumptions} == {"reward-ceiling", "function-class"}


def test_is_certified_is_false_unlike_the_tabular_agent() -> None:
    assert FittedQIteration(2, 1, OneHotEncoder(2)).is_certified is False


def test_the_certificate_serialises() -> None:
    certificate = FittedQIteration(2, 1, OneHotEncoder(2)).certificate()
    assert certificate.to_dict()["kind"] == "empirical"


def test_a_custom_regressor_factory_is_used() -> None:
    calls: list[int] = []

    class _Counting:
        def __init__(self) -> None:
            calls.append(1)
            self._mean = 0.0

        def fit(self, x: np.ndarray, y: np.ndarray) -> "_Counting":
            self._mean = float(np.mean(y))
            return self

        def predict(self, x: np.ndarray) -> np.ndarray:
            return np.full(len(x), self._mean)

    transitions, _, _, _, _ = _tabular_dataset()
    FittedQIteration(3, 2, OneHotEncoder(4), regressor=_Counting, seed=0).fit(transitions)
    assert calls  # one per (step, action) with data


def test_constructor_validates_its_shape_arguments() -> None:
    with pytest.raises(ValueError, match="n_actions"):
        FittedQIteration(0, 1, OneHotEncoder(2))
    with pytest.raises(ValueError, match="horizon"):
        FittedQIteration(2, 0, OneHotEncoder(2))
