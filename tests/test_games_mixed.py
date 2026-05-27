"""Mixed-strategy Nash equilibria (Task 9): exact two-player, verified-numerical for n>=3."""

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


def _three_player_cyclic_matching() -> CausalGame:
    # Cyclic "matching pennies" for three: a wants to match b, b wants to match c, c wants to
    # mismatch a. No pure equilibrium; the unique totally-mixed one is (1/2, 1/2) for everyone.
    agents = ("a", "b", "c")
    actions = {x: (0, 1) for x in agents}
    profiles = list(product((0, 1), (0, 1), (0, 1)))
    utilities = {
        "a": {(x, y, z): 1.0 if x == y else 0.0 for (x, y, z) in profiles},
        "b": {(x, y, z): 1.0 if y == z else 0.0 for (x, y, z) in profiles},
        "c": {(x, y, z): 1.0 if z != x else 0.0 for (x, y, z) in profiles},
    }
    decisions = [decision_node(x) for x in agents]
    utility_nodes = [utility_node(x) for x in agents]
    graph = CausalGraph(directed_edges=[(d, u) for d in decisions for u in utility_nodes])
    return CausalGame(agents, actions, utilities, graph)


def test_three_player_cyclic_has_the_uniform_mixed_equilibrium() -> None:
    equilibria = mixed_nash_equilibria(_three_player_cyclic_matching())
    uniform = {0: 0.5, 1: 0.5}
    assert any(
        e["a"] == pytest.approx(uniform)
        and e["b"] == pytest.approx(uniform)
        and e["c"] == pytest.approx(uniform)
        for e in equilibria
    )


def test_three_player_results_are_all_epsilon_nash() -> None:
    from causalrl.games import _is_epsilon_nash

    game = _three_player_cyclic_matching()
    equilibria = mixed_nash_equilibria(game)
    assert equilibria  # at least the uniform mix
    assert all(_is_epsilon_nash(game, e, epsilon=1e-5) for e in equilibria)


def test_fewer_than_two_agents_raises() -> None:
    from causalrl.exceptions import CausalGraphError

    agents = ("solo",)
    actions = {"solo": (0, 1)}
    utilities = {"solo": {(0,): 1.0, (1,): 0.0}}
    graph = CausalGraph(directed_edges=[(decision_node("solo"), utility_node("solo"))])
    with pytest.raises(CausalGraphError):
        mixed_nash_equilibria(CausalGame(agents, actions, utilities, graph))
