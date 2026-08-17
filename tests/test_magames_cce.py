"""CCE partial identification (T2 instrument): polytope, LP bounds, measured-regret certificates.

Pure Python/NumPy on the shipped ``CausalGame``, so fully locally verifiable. The named games are
the standard 2x2 anchors: a dominant-strategy game (unique CCE — width zero), an anti-coordination
game (CCE strictly wider than Nash), and a zero-sum game (every CCE achieves the value).
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pytest

from causalrl.certify.certificate import Kind
from causalrl.games import CausalGame
from causalrl.identification.bounds import Interval
from causalrl.magames import (
    CCEPolytope,
    PayoffError,
    cce_bounds,
    cce_polytope,
    cce_regret,
    certify_cce_do,
)
from causalrl.magames.population import AgentType, Population

_DOMINANT = {(0, 0): 3.0, (0, 1): 0.0, (1, 0): 5.0, (1, 1): 1.0}  # action 1 strictly dominant
_ANTI = {(0, 0): 0.0, (0, 1): 7.0, (1, 0): 2.0, (1, 1): 6.0}  # anti-coordination ("chicken")


def _symmetric_game(table: Mapping[tuple[int, int], float]) -> CausalGame:
    def payoff(own: int, others: tuple[int, ...], params: Mapping[str, float]) -> float:
        return table[(own, others[0])]

    t = AgentType(name="sym", actions=(0, 1), payoff=payoff)
    return Population(agents=("A1", "A2"), types={"A1": t, "A2": t}).to_game()


def _zero_sum_game() -> CausalGame:
    def matcher(own: int, others: tuple[int, ...], params: Mapping[str, float]) -> float:
        return 1.0 if own == others[0] else -1.0

    def mismatcher(own: int, others: tuple[int, ...], params: Mapping[str, float]) -> float:
        return -1.0 if own == others[0] else 1.0

    a = AgentType(name="match", actions=(0, 1), payoff=matcher)
    b = AgentType(name="mismatch", actions=(0, 1), payoff=mismatcher)
    return Population(agents=("A1", "A2"), types={"A1": a, "A2": b}).to_game()


def _welfare(game: CausalGame):
    def functional(profile: Mapping[str, int]) -> float:
        joint = tuple(profile[a] for a in game.agents)
        return sum(game.utilities[a][joint] for a in game.agents)

    return functional


def test_polytope_shape_and_labels() -> None:
    game = _symmetric_game(_DOMINANT)
    poly = cce_polytope(game)
    assert isinstance(poly, CCEPolytope)
    assert poly.agents == ("A1", "A2")
    assert len(poly.profiles) == 4
    assert poly.deviation_gains.shape == (4, 4)  # 2 agents x 2 deviation actions
    assert set(poly.constraint_labels) == {("A1", 0), ("A1", 1), ("A2", 0), ("A2", 1)}


def test_dominant_strategy_game_has_unique_cce() -> None:
    game = _symmetric_game(_DOMINANT)
    bounds = cce_bounds(game, _welfare(game))
    assert bounds == pytest.approx(Interval(2.0, 2.0))  # all mass on the dominant profile (1, 1)


def test_anti_coordination_cce_strictly_wider_than_nash() -> None:
    game = _symmetric_game(_ANTI)
    bounds = cce_bounds(game, _welfare(game))
    # Pure Nash welfare 9, mixed Nash welfare 28/3; the CCE polytope strictly extends beyond both.
    assert bounds.lower <= 9.0 <= bounds.upper
    assert bounds.lower <= 28.0 / 3.0 <= bounds.upper
    assert bounds.upper >= 10.0  # the public-signal distribution beats every Nash
    assert bounds.upper <= 12.0  # cannot exceed the best joint profile
    assert bounds.upper - bounds.lower > 1.0


def test_zero_sum_value_is_constant_over_cce() -> None:
    game = _zero_sum_game()

    def payoff_a1(profile: Mapping[str, int]) -> float:
        return game.utilities["A1"][(profile["A1"], profile["A2"])]

    bounds = cce_bounds(game, payoff_a1)
    assert bounds.lower == pytest.approx(0.0, abs=1e-8)
    assert bounds.upper == pytest.approx(0.0, abs=1e-8)


def test_do_restricts_the_polytope() -> None:
    game = _symmetric_game(_DOMINANT)
    poly = cce_polytope(game, do={"A2": 0})
    assert all(p[1] == 0 for p in poly.profiles)
    assert all(agent == "A1" for agent, _ in poly.constraint_labels)

    def payoff_a1(profile: Mapping[str, int]) -> float:
        return game.utilities["A1"][(profile["A1"], profile["A2"])]

    bounds = cce_bounds(game, payoff_a1, do={"A2": 0})
    # Forcing the partner to 0 leaves a unique best response: A1 plays 1 and collects 5.
    assert bounds == pytest.approx(Interval(5.0, 5.0))


def test_epsilon_inflates_the_interval() -> None:
    game = _symmetric_game(_DOMINANT)
    exact = cce_bounds(game, _welfare(game))
    inflated = cce_bounds(game, _welfare(game), epsilon=2.0)
    assert inflated.lower <= exact.lower and inflated.upper >= exact.upper
    assert inflated.upper == pytest.approx(6.0)  # epsilon=2 admits the cooperative profile


def test_epsilon_per_agent_mapping() -> None:
    game = _symmetric_game(_DOMINANT)
    bounds = cce_bounds(game, _welfare(game), epsilon={"A1": 2.0, "A2": 0.0})
    # A2's exact constraint still removes profiles where A2 plays 0; welfare peaks at (0, 1) -> 5.
    assert bounds.upper == pytest.approx(5.0)


def test_cce_regret_zero_at_dominant_profile_positive_at_uniform() -> None:
    game = _symmetric_game(_DOMINANT)
    dominant = {(1, 1): 1.0}
    uniform = {p: 0.25 for p in ((0, 0), (0, 1), (1, 0), (1, 1))}
    assert cce_regret(game, dominant) == pytest.approx(0.0)
    assert cce_regret(game, uniform) == pytest.approx(0.75)


def test_cce_regret_accepts_aligned_sequence_and_validates_mass() -> None:
    game = _symmetric_game(_DOMINANT)
    poly = cce_polytope(game)
    aligned = np.full(len(poly.profiles), 0.25)
    assert cce_regret(game, aligned) == pytest.approx(0.75)
    with pytest.raises(ValueError, match="sum to 1"):
        cce_regret(game, {(1, 1): 0.5})


def test_certify_identified_when_functional_constant() -> None:
    game = _symmetric_game(_DOMINANT)
    cert = certify_cce_do(game, _welfare(game))
    assert cert.kind is Kind.IDENTIFIED
    assert cert.value == pytest.approx(2.0)
    assert cert.hedge is None
    assert any(a.name == "no-regret" for a in cert.assumptions)


def test_certify_bounded_with_interval_value() -> None:
    game = _symmetric_game(_ANTI)
    cert = certify_cce_do(game, _welfare(game))
    assert cert.kind is Kind.BOUNDED
    assert isinstance(cert.value, Interval)
    assert cert.witness is not None and cert.witness.detail["width"] > 1.0


def test_certify_measured_regret_discharges_the_assumption() -> None:
    game = _symmetric_game(_DOMINANT)
    cert = certify_cce_do(game, _welfare(game), no_regret=False, epsilon=0.75)
    assert cert.kind is Kind.BOUNDED
    assert isinstance(cert.value, Interval)
    assert cert.witness is not None and cert.witness.detail["epsilon"] == 0.75


def test_certify_abstains_when_interval_is_vacuous() -> None:
    game = _symmetric_game(_DOMINANT)
    cert = certify_cce_do(game, _welfare(game), epsilon=100.0)
    assert cert.kind is Kind.EMPIRICAL
    assert cert.hedge is not None and "vacuous" in cert.hedge.reason


def test_certify_empirical_without_no_regret_or_measurement() -> None:
    game = _symmetric_game(_ANTI)
    cert = certify_cce_do(game, _welfare(game), no_regret=False)
    assert cert.kind is Kind.EMPIRICAL
    assert cert.hedge is not None and "no-regret" in cert.hedge.reason
    assert isinstance(cert.value, Interval)  # the interval is still reported as evidence


def test_certificate_serializes_roundtrip() -> None:
    from causalrl.certify.certificate import Certificate

    game = _symmetric_game(_ANTI)
    cert = certify_cce_do(game, _welfare(game))
    restored = Certificate.from_json(cert.to_json())
    assert restored.kind is cert.kind
    assert isinstance(restored.value, Interval)


def test_top_level_exports() -> None:
    import causalrl

    for name in ("cce_polytope", "cce_bounds", "cce_regret", "certify_cce_do", "CCEPolytope"):
        assert hasattr(causalrl, name)
        assert name in causalrl.__all__


def test_polytope_validates_do() -> None:
    game = _symmetric_game(_DOMINANT)
    with pytest.raises(KeyError, match="unknown agent"):
        cce_polytope(game, do={"nope": 0})
    with pytest.raises(ValueError, match="not available"):
        cce_polytope(game, do={"A2": 7})


def test_epsilon_validation() -> None:
    game = _symmetric_game(_DOMINANT)
    with pytest.raises(ValueError, match="nonnegative"):
        cce_bounds(game, _welfare(game), epsilon=-0.1)
    with pytest.raises(KeyError, match="unknown agents"):
        cce_bounds(game, _welfare(game), epsilon={"nope": 1.0})


def test_certificate_reports_epsilon_sensitivity() -> None:
    game = _symmetric_game(_DOMINANT)
    cert = certify_cce_do(game, _welfare(game), no_regret=False, epsilon=0.3)
    assert cert.witness is not None
    sens = cert.witness.detail["epsilon_sensitivity"]
    assert sens["width"] >= 0.0
    assert sens["upper"] >= 0.0 >= sens["lower"]
    # Finite-difference check on the actual bounds at the same epsilon.
    h = 1e-6
    lo0, hi0 = cce_bounds(game, _welfare(game), epsilon=0.3)
    lo1, hi1 = cce_bounds(game, _welfare(game), epsilon=0.3 + h)
    assert (hi1 - hi0) / h == pytest.approx(sens["upper"], abs=1e-5)
    assert (lo1 - lo0) / h == pytest.approx(sens["lower"], abs=1e-5)
    assert sens["width"] == pytest.approx(sens["upper"] - sens["lower"], abs=1e-9)


# --- the payoff table itself is an estimate ------------------------------------------------------


def _table_game(
    first: Mapping[tuple[int, int], float], second: Mapping[tuple[int, int], float], size: int
) -> CausalGame:
    """A 2-player game from two explicit tables, both keyed by ``(A1 action, A2 action)``."""

    def one(own: int, others: tuple[int, ...], params: Mapping[str, float]) -> float:
        return first[(own, others[0])]

    def two(own: int, others: tuple[int, ...], params: Mapping[str, float]) -> float:
        return second[(others[0], own)]

    actions = tuple(range(size))
    return Population(
        agents=("A1", "A2"),
        types={
            "A1": AgentType(name="a1", actions=actions, payoff=one),
            "A2": AgentType(name="a2", actions=actions, payoff=two),
        },
    ).to_game()


def _own_payoff(table: Mapping[tuple[int, int], float]):
    def functional(profile: Mapping[str, int]) -> float:
        return table[(profile["A1"], profile["A2"])]

    return functional


def test_payoff_error_widens_the_interval_to_cover_the_game_the_table_estimates() -> None:
    """The soundness property: measure a game to within delta, and the true bounds are inside.

    Twenty random 3x3 games, each measured with every cell moved by at most ``delta``. A deviation
    gain is a difference of two measured cells, so relaxing the polytope by ``2 * delta`` makes the
    estimated feasible set contain the true one; the functional's own error then widens the result.
    The `missed` counter is the null: without that widening the reported interval really does fall
    on the wrong side of the truth, so the test is not passing on slack alone.
    """
    rng = np.random.default_rng(0)
    delta, epsilon, size = 0.15, 0.05, 3
    missed = 0
    for _ in range(20):
        keys = [(i, j) for i in range(size) for j in range(size)]
        truth = ({k: float(rng.uniform(0.0, 1.0)) for k in keys} for _ in range(2))
        true_one, true_two = truth
        measured_one = {k: v + float(rng.uniform(-delta, delta)) for k, v in true_one.items()}
        measured_two = {k: v + float(rng.uniform(-delta, delta)) for k, v in true_two.items()}

        certificate = certify_cce_do(
            _table_game(measured_one, measured_two, size),
            _own_payoff(measured_one),
            no_regret=False,
            epsilon=epsilon,
            payoff_error=PayoffError(utility=delta, functional=delta),
        )
        true_interval = cce_bounds(
            _table_game(true_one, true_two, size), _own_payoff(true_one), epsilon=epsilon
        )

        assert certificate.ci is not None
        assert certificate.ci.lower <= true_interval.lower + 1e-9
        assert certificate.ci.upper >= true_interval.upper + -1e-9
        assert isinstance(certificate.value, Interval)
        if (
            certificate.value.lower > true_interval.lower + 1e-9
            or certificate.value.upper < true_interval.upper - 1e-9
        ):
            missed += 1
    assert missed > 0


def test_measured_payoffs_cannot_buy_an_identification_claim() -> None:
    """A functional constant over the *estimated* polytope is not constant over the game's."""
    game = _zero_sum_game()  # every CCE achieves the value: width is exactly zero
    exact = certify_cce_do(game, _welfare(game), epsilon=0.0)
    estimated = certify_cce_do(
        game, _welfare(game), epsilon=0.0, payoff_error=PayoffError(utility=0.05, functional=0.1)
    )

    assert exact.kind is Kind.IDENTIFIED
    assert estimated.kind is Kind.BOUNDED
    assert estimated.hedge is not None  # and it says why the constancy did not survive
    assert any(a.name == "payoff-estimate" for a in estimated.assumptions)


def test_an_exact_payoff_error_licenses_exactly_what_no_payoff_error_does() -> None:
    """Zero error is not the same object as no claim about error, but it must certify the same."""
    game = _zero_sum_game()
    stated = certify_cce_do(game, _welfare(game), epsilon=0.0, payoff_error=PayoffError(0.0))
    silent = certify_cce_do(game, _welfare(game), epsilon=0.0)

    assert stated.kind is silent.kind is Kind.IDENTIFIED
    for field in ("claim", "kind", "value", "alpha", "ci", "assumptions", "witness", "hedge"):
        assert getattr(stated, field) == getattr(silent, field)  # all but the run's timestamp


def test_omitting_payoff_error_leaves_todays_certificate_untouched() -> None:
    """The seam is opt-in: an exact game certifies byte-for-byte as it did before."""
    game = _symmetric_game(_ANTI)
    certificate = certify_cce_do(game, _welfare(game), epsilon=0.1)

    assert certificate.alpha is None
    assert certificate.ci is None
    assert "payoff_error" not in certificate.witness.detail
    assert [a.name for a in certificate.assumptions] == ["finite-game", "no-regret"]


def test_the_reported_level_travels_with_the_widened_interval() -> None:
    """``ci`` is the interval at ``alpha``; ``value`` stays the partial-identification region."""
    game = _symmetric_game(_ANTI)
    certificate = certify_cce_do(
        game,
        _welfare(game),
        epsilon=0.1,
        payoff_error=PayoffError(utility=0.05, functional=0.1, alpha=0.05),
    )

    assert certificate.alpha == 0.05
    assert isinstance(certificate.value, Interval)
    assert certificate.ci is not None
    assert certificate.ci.lower < certificate.value.lower
    assert certificate.ci.upper > certificate.value.upper


def test_enough_payoff_error_makes_the_certificate_abstain() -> None:
    """Measurement noise big enough to swamp the functional's range certifies nothing at all."""
    game = _symmetric_game(_ANTI)
    certificate = certify_cce_do(
        game, _welfare(game), epsilon=0.05, payoff_error=PayoffError(utility=5.0, functional=5.0)
    )

    assert certificate.kind is Kind.EMPIRICAL
    assert certificate.hedge is not None


def test_payoff_error_rejects_impossible_bounds() -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        PayoffError(utility=-0.1)
    with pytest.raises(ValueError, match="alpha"):
        PayoffError(utility=0.1, alpha=1.5)
