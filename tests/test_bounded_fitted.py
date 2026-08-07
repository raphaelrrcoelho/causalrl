"""BoundedFittedQIteration — does the propagated envelope actually contain the truth?"""

import numpy as np
import pytest

from causalrl.agents.bounded_fitted import BoundedFittedQIteration
from causalrl.certify.certificate import Kind
from causalrl.exceptions import UnverifiedAssumptionError
from causalrl.state import FeatureTransition, IdentityEncoder, OneHotEncoder

# A confounded-action, unconfounded-transition MDP. The hidden U drives both the logged action and
# the reward but NOT the dynamics, which is exactly the regime the transition assumption names.
_N_STATES, _N_ACTIONS, _HORIZON = 2, 2, 2
_REWARD = np.array([[[0.2, 0.9], [0.7, 0.3]], [[0.6, 0.1], [0.4, 0.8]]])  # P(R=1 | s, a, u)
_BEHAVIOUR = np.array([[0.15, 0.85], [0.8, 0.2]])  # P(A=1 | s, u)
_KERNEL = np.array([[[0.7, 0.3], [0.2, 0.8]], [[0.5, 0.5], [0.1, 0.9]]])  # P(s' | s, a)


def _one_hot(index: int) -> np.ndarray:
    vector = np.zeros(_N_STATES)
    vector[index] = 1.0
    return vector


def _true_q() -> np.ndarray:
    """Backward induction on the TRUE interventional reward ``E[R|do(a),s] = mean_u P(R|s,a,u)``."""
    interventional = _REWARD.mean(axis=2)
    q = np.zeros((_HORIZON + 2, _N_STATES, _N_ACTIONS))
    value = np.zeros((_HORIZON + 2, _N_STATES))
    for step in range(_HORIZON, 0, -1):
        for state in range(_N_STATES):
            for action in range(_N_ACTIONS):
                q[step, state, action] = (
                    interventional[state, action] + _KERNEL[state, action] @ (value[step + 1])
                )
        value[step] = q[step].max(axis=1)
    return q


def _logs(n: int = 12000, seed: int = 0) -> list[FeatureTransition]:
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        state = int(rng.integers(0, _N_STATES))
        hidden = int(rng.random() < 0.5)
        action = int(rng.random() < _BEHAVIOUR[state, hidden])
        reward = float(rng.random() < _REWARD[state, action, hidden])
        successor = int(rng.random() < _KERNEL[state, action, 1])
        out.append(FeatureTransition(_one_hot(state), action, reward, _one_hot(successor), False))
    return out


def _fitted() -> BoundedFittedQIteration:
    return BoundedFittedQIteration(
        _N_ACTIONS,
        _HORIZON,
        OneHotEncoder(_N_STATES),
        transition_assumption="unconfounded",
        seed=0,
    ).fit(_logs())


def test_the_envelope_contains_the_true_interventional_q() -> None:
    """The correctness claim: ``L <= Q* <= U`` at every cell and every step.

    The recursion propagates an interval because ``max`` and conditional expectation are monotone.
    If the envelope ever excludes the truth on an unconfounded-transition problem, the propagation
    is wrong — this is the test that would catch it.
    """
    agent = _fitted()
    truth = _true_q()
    for step in (1, 2):
        for state in range(_N_STATES):
            lower, upper = agent.envelope({"state": state}, step)
            for action in range(_N_ACTIONS):
                assert lower[action] - 1e-9 <= truth[step, state, action] <= upper[action] + 1e-9


def test_the_envelope_is_informative_not_vacuous() -> None:
    # Containment alone is cheap -- [-inf, inf] contains everything. The bound has to be narrower
    # than the range a no-data answer would give, which at step h is (H - h + 1) wide.
    agent = _fitted()
    for step in (1, 2):
        widest = float(_HORIZON - step + 1)
        for state in range(_N_STATES):
            lower, upper = agent.envelope({"state": state}, step)
            assert float(np.max(upper - lower)) < 0.85 * widest


def test_the_confounding_this_fixture_carries_is_material() -> None:
    """Guards the fixture itself: a bound that contains the truth is only interesting if a naive
    point estimate would have got it wrong."""
    interventional = _REWARD.mean(axis=2)
    naive = np.zeros((_N_STATES, _N_ACTIONS))
    for state in range(_N_STATES):
        for action in range(_N_ACTIONS):
            weights = np.array(
                [
                    (1 - _BEHAVIOUR[state, u]) if action == 0 else _BEHAVIOUR[state, u]
                    for u in (0, 1)
                ]
            )
            weights = weights / weights.sum()
            naive[state, action] = sum(weights[u] * _REWARD[state, action, u] for u in (0, 1))
    assert np.abs(naive - interventional).max() > 0.2
    # At state 0 the confounded mean prefers the wrong action outright.
    assert int(naive[0].argmax()) != int(interventional[0].argmax())


def test_a_horizon_one_envelope_is_the_reward_bound_alone() -> None:
    agent = BoundedFittedQIteration(_N_ACTIONS, 1, OneHotEncoder(_N_STATES), seed=0).fit(_logs())
    interventional = _REWARD.mean(axis=2)
    for state in range(_N_STATES):
        lower, upper = agent.envelope({"state": state})
        for action in range(_N_ACTIONS):
            assert lower[action] <= interventional[state, action] <= upper[action]


def test_horizon_one_needs_no_transition_assumption() -> None:
    # Nothing is propagated at horizon 1, so the transition gate does not apply.
    agent = BoundedFittedQIteration(_N_ACTIONS, 1, OneHotEncoder(_N_STATES), seed=0)
    assert agent.is_certified
    agent.fit(_logs(n=2000))
    assert agent.certificate().kind is Kind.BOUNDED


def test_multi_step_propagation_is_refused_without_the_transition_assumption() -> None:
    agent = BoundedFittedQIteration(_N_ACTIONS, 2, OneHotEncoder(_N_STATES), seed=0)
    assert not agent.is_certified
    with pytest.raises(UnverifiedAssumptionError, match="unconfounded"):
        agent.fit(_logs(n=500))


def test_allow_heuristic_runs_but_downgrades_the_certificate() -> None:
    agent = BoundedFittedQIteration(
        _N_ACTIONS, 2, OneHotEncoder(_N_STATES), allow_heuristic=True, seed=0
    ).fit(_logs(n=2000))
    certificate = agent.certificate()
    assert certificate.kind is Kind.EMPIRICAL
    assert certificate.hedge is not None
    assert certificate.hedge.downgraded_from == "bounded"
    assert "logged successor" in certificate.hedge.reason


def test_a_certified_run_reports_bounded_but_still_hedges_the_model_assumptions() -> None:
    certificate = _fitted().certificate()
    assert certificate.kind is Kind.BOUNDED
    assert certificate.hedge is not None  # specification is still an assumption, and it is stated
    assert "specification" in certificate.hedge.reason
    names = {a.name for a in certificate.assumptions}
    assert names == {
        "reward-range",
        "outcome-model-specification",
        "propensity-model-specification",
        "unconfounded-transitions",
    }


def test_the_overlap_diagnostic_rides_along_on_the_propensity_assumption() -> None:
    certificate = _fitted().certificate()
    propensity = next(
        a for a in certificate.assumptions if a.name == "propensity-model-specification"
    )
    assert propensity.checkable
    assert propensity.diagnostic is not None
    assert set(propensity.diagnostic) == {
        "min_propensity",
        "mean_propensity",
        "vacuous_fraction",
    }


def test_the_certificate_serialises() -> None:
    assert _fitted().certificate().to_dict()["kind"] == "bounded"


def test_interval_matches_the_envelope() -> None:
    agent = _fitted()
    lower, upper = agent.envelope({"state": 0}, 1)
    interval = agent.interval({"state": 0}, 0, 1)
    assert interval.lower == pytest.approx(float(lower[0]))
    assert interval.upper == pytest.approx(float(upper[0]))


def test_non_dominated_keeps_every_action_under_wide_natural_bounds() -> None:
    # Manski natural bounds are wide, so domination is typically a no-op -- correct, not a defect.
    agent = _fitted()
    assert agent.non_dominated({"state": 0}, 1) == [0, 1]


def test_non_dominated_drops_an_action_whose_interval_lies_below_another_floor() -> None:
    # A near-deterministic log makes both intervals tight, so domination can actually bite.
    rows = []
    for _ in range(400):
        rows.append(FeatureTransition(_one_hot(0), 0, 0.95, _one_hot(0), True))
    for _ in range(400):
        rows.append(FeatureTransition(_one_hot(1), 1, 0.05, _one_hot(1), True))
    agent = BoundedFittedQIteration(2, 1, OneHotEncoder(2), seed=0).fit(rows)
    survivors = agent.non_dominated({"state": 0})
    assert 0 in survivors


def test_acting_is_optimistic_in_the_upper_bound() -> None:
    agent = _fitted()
    _, upper = agent.envelope({"state": 0}, 1)
    chosen = {agent.act({"state": 0, "t": 0}) for _ in range(20)}
    assert chosen <= {int(a) for a in np.flatnonzero(upper >= upper.max() - 1e-12)}


def test_acting_before_fitting_is_refused() -> None:
    agent = BoundedFittedQIteration(2, 1, OneHotEncoder(2))
    with pytest.raises(RuntimeError, match="no envelope yet"):
        agent.act({"state": 0})


def test_the_tabular_transition_hook_is_refused() -> None:
    agent = BoundedFittedQIteration(2, 1, OneHotEncoder(2))
    with pytest.raises(NotImplementedError, match="feature-typed"):
        agent.observe_transition(0, 1, 1, False)


def test_a_continuous_encoder_gives_a_state_dependent_envelope() -> None:
    # The point of the exercise: the bound is a function of the features, not a per-cell constant.
    rng = np.random.default_rng(2)
    encoder = IdentityEncoder(["x"])
    rows = []
    for _ in range(3000):
        x = float(rng.uniform())
        action = int(rng.random() < 0.5)
        reward = float(rng.random() < (0.9 * x if action == 1 else 0.9 * (1.0 - x)))
        features = np.array([x])
        rows.append(FeatureTransition(features, action, reward, features, True))
    agent = BoundedFittedQIteration(2, 1, encoder, seed=0).fit(rows)
    low_x, _ = agent.envelope({"x": 0.05})
    high_x, _ = agent.envelope({"x": 0.95})
    # Action 0 is better at small x, action 1 at large x, and the lower bounds must track that.
    assert low_x[0] > low_x[1]
    assert high_x[1] > high_x[0]


def test_envelope_rejects_a_step_outside_the_horizon() -> None:
    agent = _fitted()
    with pytest.raises(ValueError, match="must lie in"):
        agent.envelope({"state": 0}, 5)


def test_features_from_a_different_encoder_are_refused() -> None:
    agent = BoundedFittedQIteration(2, 1, OneHotEncoder(3))
    with pytest.raises(ValueError, match="different encoder"):
        agent.fit([FeatureTransition(np.zeros(7), 0, 0.0, np.zeros(7), True)])


def test_constructor_validates_its_arguments() -> None:
    with pytest.raises(ValueError, match="n_actions"):
        BoundedFittedQIteration(0, 1, OneHotEncoder(2))
    with pytest.raises(ValueError, match="horizon"):
        BoundedFittedQIteration(2, 0, OneHotEncoder(2))
    with pytest.raises(ValueError, match="transition_assumption"):
        BoundedFittedQIteration(2, 1, OneHotEncoder(2), transition_assumption="maybe")
