"""Plan §8.1: typed populations + equilibrium certificates.

Acceptance (c) certify_equilibrium flags a constructed non-robust equilibrium and passes a robust
one; (d) the type system refuses an IDENTIFIED claim under an EMPIRICAL-only topology. Pure-Python
(the shipped CausalGame), so fully locally verifiable.
"""

from __future__ import annotations

from collections.abc import Mapping

import pytest

from causalrl.certify.certificate import Certificate, Kind
from causalrl.games import pure_nash_equilibria
from causalrl.magames.equilibrium import KindNotLicensedError, certify_equilibrium
from causalrl.magames.population import AgentType, LearnerTopology, Population

_COORD = {(0, 0): 1.0, (0, 1): 0.0, (1, 0): 0.0, (1, 1): 1.0}  # (own, other) -> payoff (match = 1)
_PD = {(0, 0): 3.0, (0, 1): 0.0, (1, 0): 5.0, (1, 1): 1.0}  # 1 = defect is dominant


def _coord_payoff(own: int, others: tuple[int, ...], params: Mapping[str, float]) -> float:
    return _COORD[(own, others[0])]


def _pd_payoff(own: int, others: tuple[int, ...], params: Mapping[str, float]) -> float:
    return _PD[(own, others[0])]


def _population(payoff, topology=LearnerTopology.SINGLE_LEARNER) -> Population:
    t = AgentType(name="sym", actions=(0, 1), payoff=payoff)
    return Population(agents=("A1", "A2"), types={"A1": t, "A2": t}, topology=topology)


def test_population_materialises_symmetric_game() -> None:
    game = _population(_coord_payoff).to_game()
    assert game.agents == ("A1", "A2")
    assert game.utilities["A1"][(0, 0)] == 1.0 and game.utilities["A1"][(0, 1)] == 0.0
    # coordination game has exactly the two matching pure equilibria
    eq = {tuple(sorted(p.items())) for p in pure_nash_equilibria(game)}
    assert eq == {(("A1", 0), ("A2", 0)), (("A1", 1), ("A2", 1))}


def test_certify_equilibrium_base_case() -> None:
    game = _population(_coord_payoff).to_game()
    cert = certify_equilibrium(game, {"A1": 0, "A2": 0})
    assert cert.hedge is None and cert.kind is Kind.IDENTIFIED
    assert cert.witness is not None and cert.witness.kind == "nash-equilibrium"


def test_certify_equilibrium_flags_non_robust() -> None:
    """Acceptance (c): (0,0) is Nash but NOT robust to forcing the partner to 1."""
    game = _population(_coord_payoff).to_game()
    cert = certify_equilibrium(game, {"A1": 0, "A2": 0}, do={"A2": 1})
    assert cert.value is None and cert.hedge is not None
    assert cert.hedge.reason == "not-an-equilibrium"
    assert cert.hedge.detail is not None and cert.hedge.detail["deviating_agent"] == "A1"


def test_certify_equilibrium_passes_robust() -> None:
    """Acceptance (c): the dominant-strategy PD equilibrium survives forcing the partner."""
    game = _population(_pd_payoff).to_game()
    cert = certify_equilibrium(game, {"A1": 1, "A2": 1}, do={"A2": 0})
    assert cert.hedge is None
    assert cert.witness is not None and cert.witness.detail["max_regret"] == 0.0


def test_kind_capped_by_topology() -> None:
    game = _population(_pd_payoff, LearnerTopology.INDEPENDENT_LEARNERS).to_game()
    cert = certify_equilibrium(
        game, {"A1": 1, "A2": 1}, topology=LearnerTopology.INDEPENDENT_LEARNERS
    )
    assert cert.kind is Kind.EMPIRICAL  # simultaneous learners: EMPIRICAL only (I2)


def test_require_identified_on_empirical_topology_raises() -> None:
    """Acceptance (d): the type system refuses IDENTIFIED where only EMPIRICAL is licensed."""
    game = _population(_pd_payoff).to_game()
    with pytest.raises(KindNotLicensedError, match="IDENTIFIED"):
        certify_equilibrium(
            game,
            {"A1": 1, "A2": 1},
            topology=LearnerTopology.INDEPENDENT_LEARNERS,
            require_kind=Kind.IDENTIFIED,
        )


def test_equilibrium_certificate_roundtrips() -> None:
    game = _population(_pd_payoff).to_game()
    cert = certify_equilibrium(game, {"A1": 1, "A2": 1})
    assert Certificate.from_json(cert.to_json()).kind is Kind.IDENTIFIED


def test_population_rejects_untyped_agent() -> None:
    t = AgentType(name="sym", actions=(0, 1), payoff=_pd_payoff)
    with pytest.raises(ValueError, match="no AgentType"):
        Population(agents=("A1", "A2"), types={"A1": t})
