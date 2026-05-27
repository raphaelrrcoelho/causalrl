"""Causal games: best responses, pure Nash equilibria, and the influence-diagram structure."""

from __future__ import annotations

import pytest

from causalrl.envs.suite.games import coordination_game, matching_pennies, prisoners_dilemma
from causalrl.exceptions import CausalGraphError
from causalrl.games import CausalGame, best_response, is_nash_equilibrium, pure_nash_equilibria
from causalrl.scm.graph import CausalGraph


def test_prisoners_dilemma_dominant_defection_and_unique_ne() -> None:
    game = prisoners_dilemma()
    assert best_response(game, "row", {"row": 0, "col": 0}) == frozenset({1})  # defect dominates
    assert best_response(game, "row", {"row": 0, "col": 1}) == frozenset({1})
    assert pure_nash_equilibria(game) == [{"row": 1, "col": 1}]


def test_coordination_game_has_two_equilibria() -> None:
    equilibria = pure_nash_equilibria(coordination_game())
    assert {"row": 0, "col": 0} in equilibria
    assert {"row": 1, "col": 1} in equilibria
    assert len(equilibria) == 2


def test_matching_pennies_has_no_pure_equilibrium() -> None:
    assert pure_nash_equilibria(matching_pennies()) == []


def test_is_nash_equilibrium_matches_enumeration() -> None:
    game = prisoners_dilemma()
    assert is_nash_equilibrium(game, {"row": 1, "col": 1})
    assert not is_nash_equilibrium(game, {"row": 0, "col": 0})


def test_influence_diagram_structure() -> None:
    game = prisoners_dilemma()
    assert set(game.graph.nodes) >= {"D_row", "D_col", "U_row", "U_col"}
    assert ("D_row", "U_col") in game.graph.directed_edges  # row's decision affects col's utility
    assert ("D_col", "U_row") in game.graph.directed_edges


def test_malformed_utilities_raise() -> None:
    incomplete = {"row": {(0, 0): 1.0}, "col": {(0, 0): 1.0}}  # missing 3 of 4 profiles
    graph = CausalGraph(
        directed_edges=[
            ("D_row", "U_row"),
            ("D_row", "U_col"),
            ("D_col", "U_row"),
            ("D_col", "U_col"),
        ]
    )
    with pytest.raises(CausalGraphError):
        CausalGame(("row", "col"), {"row": (0, 1), "col": (0, 1)}, incomplete, graph)
