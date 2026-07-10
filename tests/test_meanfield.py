"""Plan §8.2: mean-field limit of a symmetric population (EXPERIMENTAL, evaluation-only).

Reachable only via ``causalrl.meanfield`` (not the top-level public API); certificates are
EMPIRICAL. Pure-Python + numpy; fully local.
"""

from __future__ import annotations

import causalrl
from causalrl.certify.certificate import Kind
from causalrl.meanfield import (
    UNSTABLE,
    MeanFieldGame,
    best_response_fraction,
    certify_mean_field_equilibrium,
    mean_field_equilibria,
)


def _coordination() -> MeanFieldGame:
    # matching the population pays: action 1 pays p, action 0 pays 1 - p
    return MeanFieldGame(payoff=lambda a, p: p if a == 1 else 1.0 - p)


def _anti_coordination() -> MeanFieldGame:
    # mismatching pays: action 1 pays 1 - p, action 0 pays p -> a single interior equilibrium
    return MeanFieldGame(payoff=lambda a, p: (1.0 - p) if a == 1 else p)


def test_unstable_flag_and_not_public() -> None:
    assert UNSTABLE is True
    # experimental package is deliberately absent from the frozen top-level API
    assert "MeanFieldGame" not in causalrl.__all__


def test_coordination_has_two_pure_and_one_interior_equilibrium() -> None:
    eqs = mean_field_equilibria(_coordination())
    assert eqs == [0.0, 0.5, 1.0]  # both pure + the unstable interior at p=0.5


def test_anti_coordination_has_single_interior_equilibrium() -> None:
    eqs = mean_field_equilibria(_anti_coordination())
    assert eqs == [0.5]  # neither pure profile is stable; the mix at 1/2 is the equilibrium


def test_best_response_fraction() -> None:
    game = _coordination()
    assert best_response_fraction(game, 0.9) == 1.0  # most play 1 -> best-respond 1
    assert best_response_fraction(game, 0.1) == 0.0
    assert best_response_fraction(game, 0.5) == 0.5  # indifferent


def test_certificate_is_empirical() -> None:
    cert = certify_mean_field_equilibrium(_coordination())
    assert cert.kind is Kind.EMPIRICAL
    assert cert.assumptions[0].name == "mean-field-limit"
    assert cert.assumptions[0].params["unstable"] is True
