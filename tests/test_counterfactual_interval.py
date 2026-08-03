# tests/test_counterfactual_interval.py
import numpy as np
import pytest

from causalrl.exceptions import NotIdentifiableError
from causalrl.identification.counterfactual_bounds import (
    CounterfactualBound,
    counterfactual_interval,
)
from causalrl.scm.fit import fit_scm
from causalrl.scm.graph import CausalGraph


def _binary_data(p1: float, p0: float, n: int = 60_000, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    a = (rng.random(n) < 0.5).astype(int)
    y = (rng.random(n) < np.where(a == 1, p1, p0)).astype(int)
    return {"A": a, "Y": y}


def test_binary_bound_matches_the_analytic_frechet_limits():
    # P(Y=1|A=1) = 0.8, P(Y=1|A=0) = 0.3. Query: E[Y_{do(A=1)} | A=0, Y=0].
    # Marginals of the coupling: P(Y_cf=1) = 0.8, P(Y_f=0 | A=0) = 0.7. Frechet gives
    # P(Y_cf=1, Y_f=0) in [max(0, 0.8 + 0.7 - 1), min(0.8, 0.7)] = [0.5, 0.7];
    # divide by 0.7  ->  [5/7, 1.0].
    scm = fit_scm(_binary_data(0.8, 0.3), graph=CausalGraph(directed_edges=[("A", "Y")]))
    bound = counterfactual_interval(
        scm, evidence={"A": 0.0, "Y": 0.0}, interventions={"A": 1.0}, target="Y"
    )
    assert bound.lower == pytest.approx(5.0 / 7.0, abs=0.03)
    assert bound.upper == pytest.approx(1.0, abs=0.03)
    assert bound.tight is True
    assert 0.0 <= bound.lower <= bound.upper <= 1.0  # Y is {0,1}-valued: a probability bound


def test_bound_is_vacuous_when_treatment_has_no_average_effect():
    # P(Y=1|A=1) = P(Y=1|A=0) = 0.65: A has NO effect on Y at the population level, so the
    # marginals of the coupling coincide (P(Y_cf=1) = P(Y_f=1|A=0) = 0.65). But the per-unit
    # coupling is still unidentified: a unit observed with A=0, Y=0 could still go either way
    # under do(A=1). Frechet: P(Y_cf=1, Y_f=0) in [max(0, 0.65+0.35-1), min(0.65,0.35)] = [0, 0.35];
    # divide by 0.35 -> the fully vacuous [0.0, 1.0].
    #
    # This is the sharp end of the Frechet LOWER bound specifically: it is exactly 0 here because
    # p_cf + p_f - 1 = 0. The naive (wrong) max(0, p_cf - p_f) = max(0, 0.65 - 0.35) = 0.3 would
    # instead report a lower bound near 0.3 -- a materially narrower, wrong interval. Unlike
    # test_binary_bound_matches_the_analytic_frechet_limits (whose p_cf=0.8, p_f=0.7 numbers turn
    # out, coincidentally, to make _extreme_under_sum's result insensitive to which of the two
    # formulas floors the cell -- both fill the same cell to the same total from the *other*
    # side's ceiling), the equal-marginals case pins the lower cell directly, so it is the sharper
    # regression check on the Frechet floor.
    scm = fit_scm(_binary_data(0.65, 0.65), graph=CausalGraph(directed_edges=[("A", "Y")]))
    bound = counterfactual_interval(
        scm, evidence={"A": 0.0, "Y": 0.0}, interventions={"A": 1.0}, target="Y"
    )
    assert bound.lower == pytest.approx(0.0, abs=0.05)
    assert bound.upper == pytest.approx(1.0, abs=0.03)
    assert bound.tight is True
    assert 0.0 <= bound.lower <= bound.upper <= 1.0  # Y is {0,1}-valued: a probability bound


def test_bound_always_contains_the_truth_under_a_known_coupling():
    scm = fit_scm(_binary_data(0.9, 0.2), graph=CausalGraph(directed_edges=[("A", "Y")]))
    bound = counterfactual_interval(
        scm, evidence={"A": 0.0, "Y": 0.0}, interventions={"A": 1.0}, target="Y"
    )
    # Monotone coupling truth: P(Y_cf=1, Y_f=0) = min(0.9, 0.8) = 0.8 -> 0.8/0.8 = 1.0
    assert bound.lower <= 1.0 <= bound.upper + 1e-9
    # A plain containment check alone would not catch an allocation bug that produces an
    # out-of-range "probability" (e.g. 1.5) provided the truth still happened to fall inside
    # [lower, upper] by luck of the numbers -- pin Y's {0,1} range explicitly too.
    assert 0.0 <= bound.lower <= bound.upper <= 1.0


def test_bound_is_degenerate_when_every_mechanism_is_invertible():
    rng = np.random.default_rng(0)
    a = rng.normal(size=20_000)
    data = {"A": a, "Y": 2.0 * a + rng.normal(scale=0.3, size=20_000)}
    from causalrl.scm.fitters import LinearGaussianFit

    scm = fit_scm(
        data,
        graph=CausalGraph(directed_edges=[("A", "Y")]),
        families={"A": LinearGaussianFit(), "Y": LinearGaussianFit()},
    )
    # Exact abduction: u_Y = 0 - g(A=0) ~ 0, so Y_{do(A=1)} = 2*1 + u_Y ~ 2.0.
    bound = counterfactual_interval(
        scm, evidence={"A": 0.0, "Y": 0.0}, interventions={"A": 1.0}, target="Y"
    )
    assert bound.upper - bound.lower < 1e-6
    assert bound.tight is True
    assert bound.lower == pytest.approx(2.0, abs=0.15)


def test_unaffected_discrete_confounder_does_not_block_the_query():
    # Z -> A, Z -> Y, A -> Y. Z is discrete/non-invertible but is NOT downstream of do(A),
    # so its counterfactual value is its factual value and the query stays answerable.
    rng = np.random.default_rng(3)
    n = 60_000
    z = (rng.random(n) < 0.5).astype(int)
    a = (rng.random(n) < np.where(z == 1, 0.8, 0.2)).astype(int)
    y = (rng.random(n) < np.where(a + z > 1, 0.9, 0.1)).astype(int)
    scm = fit_scm(
        {"Z": z, "A": a, "Y": y},
        graph=CausalGraph(directed_edges=[("Z", "A"), ("Z", "Y"), ("A", "Y")]),
    )
    bound = counterfactual_interval(
        scm, evidence={"Z": 1.0, "A": 0.0, "Y": 0.0}, interventions={"A": 1.0}, target="Y"
    )
    assert 0.0 <= bound.lower <= bound.upper <= 1.0
    assert bound.tight is True


def test_intervention_that_changes_nothing_returns_the_observed_point():
    # I1 regression: do(A=0) on a unit whose factual A is already 0 leaves Y's parent
    # configuration unchanged, so Y_cf and Y_f are the SAME random variable under every
    # admissible coupling -- the joint is confined to the diagonal and the identified set is the
    # single point the unit was observed at. The bug applied the Frechet/transportation
    # relaxation regardless and reported CounterfactualBound(0.0, 0.4255, tight=True), i.e. a
    # modelling choice dressed as a result. "What if we had done what we did" and placebo /
    # negative-control checks are exactly this query.
    scm = fit_scm(_binary_data(0.8, 0.3), graph=CausalGraph(directed_edges=[("A", "Y")]))
    assert counterfactual_interval(
        scm, evidence={"A": 0.0, "Y": 0.0}, interventions={"A": 0.0}, target="Y"
    ) == CounterfactualBound(0.0, 0.0, True)
    # A different observed value, so the answer is pinned to evidence[target] rather than to any
    # constant that happens to match the case above.
    assert counterfactual_interval(
        scm, evidence={"A": 1.0, "Y": 1.0}, interventions={"A": 1.0}, target="Y"
    ) == CounterfactualBound(1.0, 1.0, True)
    # The degenerate entry path: no intervention at all also leaves the parents unchanged.
    assert counterfactual_interval(
        scm, evidence={"A": 1.0, "Y": 0.0}, interventions={}, target="Y"
    ) == CounterfactualBound(0.0, 0.0, True)


def test_intervening_on_the_target_itself_returns_the_intervened_value():
    # I2 regression: do(Y=1) replaces Y's own mechanism with a constant, so Y_cf = 1 by
    # definition -- no coupling is involved. `interventions` used to be consumed only by
    # _ambiguous_upstream and _counterfactual_parents, neither of which looks at the target, so
    # _pmf_at then evaluated the UN-mutilated mechanism: the answer was byte-identical to the
    # no-intervention query and the reported interval [0.0, 0.4255] EXCLUDED the truth, while
    # scm.do() and counterfactual_expectation both answer this query correctly.
    scm = fit_scm(_binary_data(0.8, 0.3), graph=CausalGraph(directed_edges=[("A", "Y")]))
    assert counterfactual_interval(
        scm, evidence={"A": 0.0, "Y": 0.0}, interventions={"Y": 1.0}, target="Y"
    ) == CounterfactualBound(1.0, 1.0, True)
    # A per-sample vector value resolves to the unit's own entry, as _counterfactual_parents does.
    assert counterfactual_interval(
        scm, evidence={"A": 0.0, "Y": 0.0}, interventions={"Y": np.array([1.0])}, target="Y"
    ) == CounterfactualBound(1.0, 1.0, True)


def test_intervening_on_the_target_answers_even_when_an_ambiguous_node_lies_upstream():
    # Ordering guard for I2's fix: A -> M -> Y with M non-invertible is the configuration
    # test_upstream_non_invertible_node_refuses_rather_than_composing_a_loose_bound refuses. But
    # do(Y=1) pins Y directly, so nothing upstream can reach it and the query is answerable.
    # Placing the target-intervention check after _ambiguous_upstream would turn this defined
    # answer into a refusal.
    rng = np.random.default_rng(1)
    n = 20_000
    a = (rng.random(n) < 0.5).astype(int)
    m = (rng.random(n) < np.where(a == 1, 0.8, 0.2)).astype(int)
    y = (rng.random(n) < np.where(m == 1, 0.9, 0.1)).astype(int)
    scm = fit_scm(
        {"A": a, "M": m, "Y": y}, graph=CausalGraph(directed_edges=[("A", "M"), ("M", "Y")])
    )
    assert counterfactual_interval(
        scm,
        evidence={"A": 0.0, "M": 0.0, "Y": 0.0},
        interventions={"A": 1.0, "Y": 1.0},
        target="Y",
    ) == CounterfactualBound(1.0, 1.0, True)


def test_counterfactual_interval_requires_a_fitted_scm():
    from torch.distributions import Normal

    from causalrl.scm.mechanisms import FunctionalMechanism, LinearGaussianMechanism, Mechanism
    from causalrl.scm.scm import StructuralCausalModel

    mechanisms: dict[str, Mechanism] = {
        "A": FunctionalMechanism([], lambda pa, u: u),
        "Y": LinearGaussianMechanism(["A"], {"A": 2.0}),
    }
    scm = StructuralCausalModel(
        CausalGraph(directed_edges=[("A", "Y")]),
        mechanisms,
        {"A": Normal(0.0, 1.0), "Y": Normal(0.0, 0.1)},
    )
    with pytest.raises(ValueError, match="counterfactual_expectation"):
        counterfactual_interval(
            scm, evidence={"A": 0.0, "Y": 0.0}, interventions={"A": 1.0}, target="Y"
        )


def test_evidence_must_cover_every_node():
    scm = fit_scm(_binary_data(0.8, 0.3), graph=CausalGraph(directed_edges=[("A", "Y")]))
    with pytest.raises(KeyError, match="Y"):
        counterfactual_interval(scm, evidence={"A": 0.0}, interventions={"A": 1.0}, target="Y")


def test_interval_property_converts_for_the_bounds_surface():
    from causalrl.identification.bounds import Interval

    bound = CounterfactualBound(0.2, 0.7, True)
    assert bound.interval == Interval(0.2, 0.7)
    low, high, tight = bound
    assert (low, high, tight) == (0.2, 0.7, True)


def test_upstream_non_invertible_node_refuses_rather_than_composing_a_loose_bound():
    rng = np.random.default_rng(1)
    n = 20_000
    a = (rng.random(n) < 0.5).astype(int)
    m = (rng.random(n) < np.where(a == 1, 0.8, 0.2)).astype(int)
    y = (rng.random(n) < np.where(m == 1, 0.9, 0.1)).astype(int)
    scm = fit_scm(
        {"A": a, "M": m, "Y": y}, graph=CausalGraph(directed_edges=[("A", "M"), ("M", "Y")])
    )
    with pytest.raises(NotIdentifiableError, match="upstream"):
        counterfactual_interval(
            scm, evidence={"A": 0.0, "M": 0.0, "Y": 0.0}, interventions={"A": 1.0}, target="Y"
        )


def test_downstream_non_invertible_sibling_does_not_block_the_query():
    # A -> M, A -> Y. M is discrete/non-invertible and IS downstream of do(A) -- unlike the Z of
    # test_unaffected_discrete_confounder_does_not_block_the_query -- but it is NOT an ancestor of
    # Y (a sibling branch, not an intermediate on a path to the target), so _ambiguous_upstream
    # must not refuse and resolving Y's counterfactual parents must never touch M at all. Reuses
    # the headline case's P(Y=1|A=1)=0.8, P(Y=1|A=0)=0.3 so the closed-form Frechet answer
    # [5/7, 1.0] also checks that M's ambiguity leaks nothing into the bound.
    rng = np.random.default_rng(7)
    n = 60_000
    a = (rng.random(n) < 0.5).astype(int)
    m = (rng.random(n) < np.where(a == 1, 0.8, 0.2)).astype(int)
    y = (rng.random(n) < np.where(a == 1, 0.8, 0.3)).astype(int)
    scm = fit_scm(
        {"A": a, "M": m, "Y": y},
        graph=CausalGraph(directed_edges=[("A", "M"), ("A", "Y")]),
    )
    bound = counterfactual_interval(
        scm, evidence={"A": 0.0, "M": 0.0, "Y": 0.0}, interventions={"A": 1.0}, target="Y"
    )
    assert bound.lower == pytest.approx(5.0 / 7.0, abs=0.03)
    assert bound.upper == pytest.approx(1.0, abs=0.03)
    assert bound.tight is True
    assert 0.0 <= bound.lower <= bound.upper <= 1.0  # Y is {0,1}-valued: a probability bound


def test_naive_frechet_floor_produces_an_infeasible_allocation_above_one():
    # P(Y=1|A=1) = 0.75, P(Y=1|A=0) = 0.30. Query: E[Y_{do(A=1)} | A=0, Y=1] (evidence is the
    # Y=1 outcome, so p_factual = P(Y=1|A=0) = 0.30 directly -- no complement needed). Marginals
    # of the coupling: P(Y_cf=1) = 0.75, P(Y_f=1 | A=0) = 0.30. Frechet gives
    # P(Y_cf=1, Y_f=1) in [max(0, 0.75 + 0.30 - 1), min(0.75, 0.30)] = [0.05, 0.30];
    # divide by 0.30 -> [1/6, 1.0].
    #
    # This is the discriminator test_binary_bound_matches_the_analytic_frechet_limits happens not
    # to be (see that test's docstring/history): with the naive (wrong) floor
    # max(0, p_cf - p_f) = max(0, 0.75 - 0.30) = 0.45, the two cells' floors alone
    # (0 and 0.45) already sum to MORE than p_factual = 0.30 -- an infeasible starting
    # allocation. _extreme_under_sum's `remaining = total - floors.sum()` goes negative, the
    # loop's `if remaining <= 0: break` fires on its first iteration, and the cells are returned
    # unfilled-past-their-floor: low = high = 0.45 / 0.30 = 1.5, a "probability" above 1 in BOTH
    # directions at once -- an unmistakable, sharp failure mode of the exact formula in
    # DESIGN.md/the task brief, not an approximation artifact.
    scm = fit_scm(_binary_data(0.75, 0.30), graph=CausalGraph(directed_edges=[("A", "Y")]))
    bound = counterfactual_interval(
        scm, evidence={"A": 0.0, "Y": 1.0}, interventions={"A": 1.0}, target="Y"
    )
    assert bound.lower == pytest.approx(1.0 / 6.0, abs=0.03)
    assert bound.upper == pytest.approx(1.0, abs=0.03)
    assert bound.tight is True
    assert 0.0 <= bound.lower <= bound.upper <= 1.0  # Y is {0,1}-valued: a probability bound


def test_interval_bounds_are_ordered_and_within_the_outcome_range():
    scm = fit_scm(_binary_data(0.6, 0.4), graph=CausalGraph(directed_edges=[("A", "Y")]))
    bound = counterfactual_interval(
        scm, evidence={"A": 0.0, "Y": 1.0}, interventions={"A": 1.0}, target="Y"
    )
    assert 0.0 <= bound.lower <= bound.upper <= 1.0
