"""FunctionalManskiBounds — the per-cell Manski bound generalised to a function of features."""

import numpy as np
import pytest

from causalrl.bounds.functional import FunctionalManskiBounds
from causalrl.data.dataset import ConfoundedTrajectoryDataset, Transition
from causalrl.identification.bounds import causal_q_bounds


def _confounded_logs(
    n: int = 4000, n_states: int = 3, seed: int = 0
) -> tuple[ConfoundedTrajectoryDataset, np.ndarray, np.ndarray, np.ndarray]:
    """Logs whose propensity varies by state, as one-hot features alongside the tabular dataset."""
    rng = np.random.default_rng(seed)
    propensity = [0.2, 0.5, 0.8]
    transitions, features, actions, rewards = [], [], [], []
    for _ in range(n):
        state = int(rng.integers(0, n_states))
        action = int(rng.random() < propensity[state])
        reward = float(rng.random() < (0.3 + 0.2 * state + 0.1 * action))
        transitions.append(Transition(state, action, reward, state, True))
        one_hot = np.zeros(n_states)
        one_hot[state] = 1.0
        features.append(one_hot)
        actions.append(action)
        rewards.append(reward)
    dataset = ConfoundedTrajectoryDataset(transitions, n_states, 2)
    return dataset, np.array(features), np.array(actions), np.array(rewards)


def test_one_hot_features_reproduce_the_tabular_manski_bound() -> None:
    """The faithfulness property, again: the per-cell bound is the special case, not a rival.

    With indicator features the outcome regression is the cell mean and the propensity is the cell
    frequency, so the formula collapses to `causal_q_bounds`. If this drifts, the functional bound
    has stopped generalising the thing it claims to generalise.
    """
    n_states = 3
    dataset, features, actions, rewards = _confounded_logs(n_states=n_states)
    model = FunctionalManskiBounds(2, n_folds=5, seed=0).fit(features, actions, rewards)
    lower, upper = model.bounds(np.eye(n_states))
    for state in range(n_states):
        for action in range(2):
            reference = causal_q_bounds(dataset, state, action)
            assert lower[state, action] == pytest.approx(reference.lower, abs=2e-3)
            assert upper[state, action] == pytest.approx(reference.upper, abs=2e-3)


def test_bounds_are_ordered_and_inside_the_reward_range() -> None:
    _, features, actions, rewards = _confounded_logs()
    model = FunctionalManskiBounds(2, seed=0).fit(features, actions, rewards)
    lower, upper = model.bounds(np.eye(3))
    assert np.all(lower <= upper)
    assert np.all(lower >= -1e-9)
    assert np.all(upper <= 1.0 + 1e-9)


def test_a_never_logged_action_gets_the_vacuous_bound() -> None:
    # Nothing in the logs speaks to it, and the honest interval is the whole reward range.
    rng = np.random.default_rng(0)
    features = np.tile(np.array([[1.0, 0.0]]), (200, 1))
    actions = np.zeros(200, dtype=int)  # action 1 never taken
    rewards = rng.random(200)
    model = FunctionalManskiBounds(2, seed=0).fit(features, actions, rewards)
    lower, upper = model.bounds(np.array([[1.0, 0.0]]))
    assert lower[0, 1] == pytest.approx(0.0, abs=1e-9)
    assert upper[0, 1] == pytest.approx(1.0, abs=1e-9)


def test_a_deterministic_policy_leaves_its_own_action_tight() -> None:
    # The other side of the same coin: when an action is always taken its propensity is 1, so the
    # unlogged fraction vanishes and the interval collapses to the observed mean.
    features = np.tile(np.array([[1.0, 0.0]]), (200, 1))
    actions = np.zeros(200, dtype=int)
    rewards = np.full(200, 0.25)
    model = FunctionalManskiBounds(2, seed=0).fit(features, actions, rewards)
    lower, upper = model.bounds(np.array([[1.0, 0.0]]))
    assert upper[0, 0] - lower[0, 0] < 1e-6
    assert lower[0, 0] == pytest.approx(0.25, abs=1e-6)


def test_in_sample_bounds_are_out_of_fold() -> None:
    _, features, actions, rewards = _confounded_logs()
    model = FunctionalManskiBounds(2, n_folds=5, seed=0).fit(features, actions, rewards)
    lower, upper = model.in_sample
    assert lower.shape == (len(features), 2)
    assert np.all(lower <= upper)


def test_a_custom_reward_range_widens_the_unlogged_part() -> None:
    _, features, actions, rewards = _confounded_logs()
    narrow = FunctionalManskiBounds(2, reward_range=(0.0, 1.0), seed=0)
    wide = FunctionalManskiBounds(2, reward_range=(-1.0, 2.0), seed=0)
    narrow.fit(features, actions, rewards)
    wide.fit(features, actions, rewards)
    n_low, n_high = narrow.bounds(np.eye(3))
    w_low, w_high = wide.bounds(np.eye(3))
    assert np.all((w_high - w_low) >= (n_high - n_low) - 1e-9)


def test_the_overlap_diagnostic_reports_the_fitted_propensities() -> None:
    _, features, actions, rewards = _confounded_logs()
    model = FunctionalManskiBounds(2, seed=0).fit(features, actions, rewards)
    diagnostic = model.diagnostic()
    assert 0.0 <= diagnostic.min_propensity <= diagnostic.mean_propensity <= 1.0
    assert 0.0 <= diagnostic.vacuous_fraction <= 1.0
    assert "overlap" in diagnostic.summary()


def test_poor_overlap_shows_up_in_the_diagnostic() -> None:
    # A near-deterministic behaviour policy leaves the unchosen action almost unsupported.
    rng = np.random.default_rng(0)
    n = 1000
    features = np.tile(np.array([[1.0]]), (n, 1))
    actions = (rng.random(n) < 0.01).astype(int)
    rewards = rng.random(n)
    model = FunctionalManskiBounds(2, seed=0).fit(features, actions, rewards)
    assert model.diagnostic().vacuous_fraction > 0.0


def test_using_the_model_before_fitting_is_refused() -> None:
    model = FunctionalManskiBounds(2)
    with pytest.raises(RuntimeError, match="not been fitted"):
        model.bounds(np.eye(2))
    with pytest.raises(RuntimeError, match="not been fitted"):
        model.diagnostic()
    with pytest.raises(RuntimeError, match="not been fitted"):
        _ = model.in_sample


def test_a_single_fold_is_refused() -> None:
    with pytest.raises(ValueError, match="n_folds"):
        FunctionalManskiBounds(2, n_folds=1)


def test_an_inverted_reward_range_is_refused() -> None:
    with pytest.raises(ValueError, match="low < high"):
        FunctionalManskiBounds(2, reward_range=(1.0, 0.0))


def test_mismatched_column_lengths_are_refused() -> None:
    with pytest.raises(ValueError, match="agree in length"):
        FunctionalManskiBounds(2, seed=0).fit(np.zeros((5, 2)), np.zeros(4, dtype=int), np.zeros(5))


def test_a_one_dimensional_design_matrix_is_refused() -> None:
    with pytest.raises(ValueError, match="2-D"):
        FunctionalManskiBounds(2, seed=0).fit(np.zeros(5), np.zeros(5, dtype=int), np.zeros(5))


def test_too_few_rows_for_the_fold_count_is_refused() -> None:
    with pytest.raises(ValueError, match="cannot be split"):
        FunctionalManskiBounds(2, n_folds=5, seed=0).fit(
            np.zeros((3, 2)), np.zeros(3, dtype=int), np.zeros(3)
        )


def test_an_out_of_range_action_is_refused() -> None:
    with pytest.raises(ValueError, match="must lie in"):
        FunctionalManskiBounds(2, seed=0).fit(
            np.zeros((10, 2)), np.full(10, 5, dtype=int), np.zeros(10)
        )
