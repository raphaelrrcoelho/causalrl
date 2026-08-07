"""Smoke test for examples/online_causal_mbrl.py.

Loads the example as a module and runs ``main()`` on a tiny budget so the whole
ingest -> refit -> act -> probe path executes at a fraction of the cost. The assertions pin the
example's *premise* and the *structural* outcome, both of which are stable:

* the world's exact ground truth, and the six-member ambiguity it was built to produce, including
  that the members genuinely disagree about the optimal action -- this is closed-form, not a
  sampling outcome;
* the belief trajectory: the observational-only arm keeps all six members however many rows it
  reads, and the arms that experiment end strictly below that.

Deliberately **not** asserted: which action any particular truncated run picks. ``act()`` draws one
I-MEC member per decision (Thompson sampling over structure), so on a short run the arm ordering by
regret is a draw, not a property. The one regret comparison made below is against the arm that
cannot improve at any sample size, over several seeds, and is stated as a weak inequality.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import pytest

BUDGET: dict[str, Any] = {
    "seeds": (0, 1, 2),
    "rounds": 3,
    "n_rollout": 64,
    "n_ambiguity_log": 1_500,
}


def _load_example_module() -> Any:
    """Load examples/online_causal_mbrl.py without executing its __main__ block."""
    example_path = Path(__file__).parent.parent / "examples" / "online_causal_mbrl.py"
    spec = importlib.util.spec_from_file_location("online_causal_mbrl", example_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def example() -> Any:
    return _load_example_module()


@pytest.fixture(scope="module")
def result(example: Any) -> Any:
    return example.main(**BUDGET)


def test_ground_truth_is_a_reversal(example: Any) -> None:
    """The trap, in closed form: the log ranks the actions the opposite way to ``do``.

    Everything downstream rests on this. If the observational contrast agreed with the causal one,
    an agent reading ``E[Y|A]`` would already be right and there would be nothing to demonstrate.
    """
    do_0, do_1 = example.TRUE_ACTION_VALUES
    obs_0, obs_1 = example.observational_contrast()
    assert do_1 > do_0, (do_0, do_1)
    assert obs_1 < obs_0, (obs_0, obs_1)
    assert example.regret(1.0) == 0.0
    assert example.regret(0.0) == pytest.approx(do_1 - do_0)


def test_true_edges_is_what_the_generator_actually_does(example: Any) -> None:
    """``TRUE_EDGES`` is recovered from the sampler by intervention, not restated from the constant.

    Everything the run reports about whether the belief still holds the *true* DAG is measured
    against this tuple, so a stale or mistyped entry would silently redefine "true". The recovery
    contrasts ``do(target=0)`` against ``do(target=1)`` rather than either against the observational
    marginal: a one-sided contrast measures how far the set value sits from the logging policy's
    average, which on this world is small for ``do(A=1)`` (0.074) precisely because the log takes
    ``A=1`` most of the time -- and small enough to look like no edge. The two-sided contrast is the
    effect itself and separates cleanly: every true edge moves its endpoint by at least 0.22, and a
    non-descendant is invariant *exactly*, so it moves only by sampling noise.

    This recovers descendants, not edges. On this world -- a complete DAG on three nodes -- every
    ancestor pair is also an edge, so the two coincide; on a sparser graph they would not.
    """
    n = 40_000
    recovered: set[tuple[str, str]] = set()
    for index, target in enumerate(example.VARIABLES):
        low = example.sample(n, seed=20 + index, do={target: np.zeros(n)})
        high = example.sample(n, seed=40 + index, do={target: np.ones(n)})
        for other in example.VARIABLES:
            if other == target:
                continue
            effect = abs(float(high[other].mean()) - float(low[other].mean()))
            if effect > 0.15:
                recovered.add((target, other))
            else:
                assert effect < 0.02, (target, other, effect)  # a non-descendant does not move
    assert recovered == set(example.TRUE_EDGES), (recovered, example.TRUE_EDGES)


def test_the_ambiguity_is_six_members_that_disagree_about_the_action(result: Any) -> None:
    """Observation alone leaves six DAGs, and they do not agree on what to do.

    ``>= 2 members disagreeing`` is the whole premise: a six-member class whose members all
    preferred the same action would be an equivalence class the decision does not care about, and
    the experiments below would buy nothing worth reporting.
    """
    verdicts = result.verdicts
    assert len(verdicts) == 6
    preferred = {verdict.preferred for verdict in verdicts}
    assert 1.0 in preferred, preferred  # the member matching the truth
    assert 0.0 in preferred, preferred  # the members that read do(A) off E[Y|A]
    assert None in preferred, preferred  # the members in which A cannot reach Y


def test_members_that_cannot_reach_the_outcome_score_the_actions_identically(result: Any) -> None:
    """``preferred is None`` is structural, not a Monte-Carlo near-tie.

    Where the treatment is not an ancestor of the outcome, ``do(A=0)`` and ``do(A=1)`` leave the
    outcome's mechanism *and* its exogenous draws untouched, so the two rollouts are the same
    sample and the two values are bit-identical. Anything looser here (a tolerance) would hide the
    difference between "this member says the action cannot matter" and "this member has a small
    effect the rollout budget cannot resolve".
    """
    indifferent = [v for v in result.verdicts if v.preferred is None]
    assert indifferent, "no member left the treatment unable to reach the outcome"
    for verdict in indifferent:
        assert verdict.values[0] == verdict.values[1], verdict
    for verdict in result.verdicts:
        if verdict.preferred is not None:
            assert verdict.values[0] != verdict.values[1], verdict


def test_arms_are_the_three_data_diets(result: Any, example: Any) -> None:
    names = [arm.name for arm in result.arms]
    assert names == ["observational only", "interventional only", "both"]
    for arm in result.arms:
        assert len(arm.rounds) == BUDGET["rounds"]
    observational, *experimenting = result.arms
    # Same per-round budget for every arm; only the kind of data differs.
    per_round = {
        spec.observational_per_round + spec.interventional_per_round for spec in example.ARMS
    }
    assert per_round == {example.ROWS_PER_ROUND}
    assert all(spec.interventional_per_round for spec in example.ARMS[1:])
    assert observational.name == example.ARMS[0].name
    assert [arm.name for arm in experimenting] == [spec.name for spec in example.ARMS[1:]]


def test_observation_alone_never_shrinks_the_belief(result: Any) -> None:
    """The arm with no experiments holds all six members at every round, however many rows it adds.

    This is the claim that makes the example worth running: the six DAGs induce the same
    observational law, so more observational rows cannot separate them. It is asserted on every
    round rather than the last, because a mid-run collapse would falsify it just as badly.
    """
    observational = result.arms[0]
    assert [r.belief_sizes for r in observational.rounds] == [(6, 6, 6)] * BUDGET["rounds"]
    assert all(r.truth_share == 1.0 for r in observational.rounds)
    assert observational.rounds[-1].mean_steps > observational.rounds[0].mean_steps


def test_experiments_collapse_the_belief_and_observation_does_not(result: Any) -> None:
    """Both experimenting arms end strictly below the observation-only arm's six members."""
    observational, *experimenting = result.arms
    assert min(observational.final.belief_sizes) == 6
    for arm in experimenting:
        assert max(arm.final.belief_sizes) < 6, (arm.name, arm.final.belief_sizes)
        # Every arm starts from the same undetermined belief; the experiments are what change it.
        assert max(arm.rounds[0].belief_sizes) == 6, (arm.name, arm.rounds[0].belief_sizes)
        # ...and what survives is the true DAG, not merely *a* DAG. A collapsed belief that had
        # excluded the truth would satisfy every assertion above and none of this one.
        assert arm.final.truth_share == 1.0, (arm.name, arm.final.truth_share)


def test_the_experimenting_arms_do_not_end_worse_than_observation_alone(result: Any) -> None:
    """A weak inequality on the pooled final regret -- the one comparison a short run supports.

    The observation-only arm cannot beat 5/6 of its belief pointing at the wrong action, at any
    sample size; the arms that experiment can. Asserted as ``<=`` rather than ``<`` because
    Thompson sampling over a six-member belief can hand the observational arm the correct member
    on a short run, and asserting a strict gap would be asserting a coin flip.
    """
    observational, *experimenting = result.arms
    for arm in experimenting:
        assert arm.final.mean_regret <= observational.final.mean_regret, (
            arm.name,
            arm.final.mean_regret,
            observational.final.mean_regret,
        )
        assert arm.final.optimal_share >= observational.final.optimal_share, arm.name


def test_regret_is_read_off_the_true_world(result: Any, example: Any) -> None:
    """Every reported regret is one of the world's exact values, never an in-model estimate."""
    exact = {example.regret(action) for action in example.ACTIONS}
    assert len(exact) == 2 and 0.0 in exact, exact
    for arm in result.arms:
        for summary in arm.rounds:
            # A mean over seeds of values drawn from `exact`: bounded by it, and on the grid when
            # the seeds agree. Bounds hold for any mixture, which is what makes this stable.
            assert min(exact) <= summary.mean_regret <= max(exact), (arm.name, summary)
            assert 0.0 <= summary.optimal_share <= 1.0
