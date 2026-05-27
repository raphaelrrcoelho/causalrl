"""Headline: the MACID equilibrium solver recovers the textbook pure equilibria."""

from __future__ import annotations

from causalrl.envs.suite.games import coordination_game, matching_pennies, prisoners_dilemma
from causalrl.games import pure_nash_equilibria


def test_canonical_games_recover_textbook_equilibria() -> None:
    assert pure_nash_equilibria(prisoners_dilemma()) == [{"row": 1, "col": 1}]  # mutual defection
    assert len(pure_nash_equilibria(coordination_game())) == 2  # both matching profiles
    assert pure_nash_equilibria(matching_pennies()) == []  # only a mixed equilibrium exists
