"""Mixed-strategy Nash equilibria (Task 9): exact support enumeration for two-player games."""

from __future__ import annotations

from itertools import product

import pytest

from causalrl.envs.suite.games import coordination_game, matching_pennies, prisoners_dilemma
from causalrl.games import (
    CausalGame,
    decision_node,
    mixed_nash_equilibria,
    utility_node,
)
from causalrl.scm.graph import CausalGraph


def test_matching_pennies_unique_uniform_mix() -> None:
    equilibria = mixed_nash_equilibria(matching_pennies())
    assert len(equilibria) == 1
    assert equilibria[0]["row"] == pytest.approx({0: 0.5, 1: 0.5})
    assert equilibria[0]["col"] == pytest.approx({0: 0.5, 1: 0.5})


def test_prisoners_dilemma_only_the_pure_equilibrium() -> None:
    # The dominant-strategy equilibrium is the only one, recovered as a degenerate (pure) mix.
    equilibria = mixed_nash_equilibria(prisoners_dilemma())
    assert equilibria == [{"row": {0: 0.0, 1: 1.0}, "col": {0: 0.0, 1: 1.0}}]


def test_coordination_game_two_pure_plus_one_mixed() -> None:
    equilibria = mixed_nash_equilibria(coordination_game())
    assert len(equilibria) == 3
    # The two pure equilibria appear as point masses ...
    assert {"row": {0: 1.0, 1: 0.0}, "col": {0: 1.0, 1: 0.0}} in equilibria
    assert {"row": {0: 0.0, 1: 1.0}, "col": {0: 0.0, 1: 1.0}} in equilibria
    # ... alongside the symmetric 50/50 mixed equilibrium.
    mixed = [e for e in equilibria if e["row"][0] not in (0.0, 1.0)]
    assert len(mixed) == 1
    assert mixed[0]["row"] == pytest.approx({0: 0.5, 1: 0.5})
    assert mixed[0]["col"] == pytest.approx({0: 0.5, 1: 0.5})


def test_every_pure_equilibrium_is_a_mixed_equilibrium() -> None:
    # Consistency: each pure NE must show up (as a point mass) among the mixed equilibria.
    from causalrl.games import pure_nash_equilibria

    game = coordination_game()
    mixed = mixed_nash_equilibria(game)
    for pure in pure_nash_equilibria(game):
        point_mass = {
            agent: {a: 1.0 if a == pure[agent] else 0.0 for a in game.actions[agent]}
            for agent in game.agents
        }
        assert point_mass in mixed


def _three_player_zero_game() -> CausalGame:
    agents = ("a", "b", "c")
    actions = {x: (0, 1) for x in agents}
    profiles = list(product((0, 1), (0, 1), (0, 1)))
    utilities = {x: {p: 0.0 for p in profiles} for x in agents}
    decisions = [decision_node(x) for x in agents]
    utility_nodes = [utility_node(x) for x in agents]
    graph = CausalGraph(directed_edges=[(d, u) for d in decisions for u in utility_nodes])
    return CausalGame(agents, actions, utilities, graph)


def test_more_than_two_players_is_out_of_scope() -> None:
    with pytest.raises(NotImplementedError):
        mixed_nash_equilibria(_three_player_zero_game())
