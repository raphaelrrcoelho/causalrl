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
