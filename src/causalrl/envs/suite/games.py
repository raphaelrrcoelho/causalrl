"""Canonical two-player games as causal influence diagrams (taxonomy Task 9)."""

from __future__ import annotations

from causalrl.games import CausalGame, decision_node, utility_node
from causalrl.scm.graph import CausalGraph

_AGENTS = ("row", "col")
_ACTIONS = {"row": (0, 1), "col": (0, 1)}


def _influence_diagram() -> CausalGraph:
    """Each agent's decision affects every agent's utility (the normal-form structure)."""
    decisions = [decision_node(a) for a in _AGENTS]
    utilities = [utility_node(a) for a in _AGENTS]
    return CausalGraph(directed_edges=[(d, u) for d in decisions for u in utilities])


def prisoners_dilemma() -> CausalGame:
    """Cooperate (0) or defect (1); ``T=5, R=3, P=1, S=0``. The unique NE is (defect, defect)."""
    row = {(0, 0): 3.0, (0, 1): 0.0, (1, 0): 5.0, (1, 1): 1.0}
    col = {(0, 0): 3.0, (0, 1): 5.0, (1, 0): 0.0, (1, 1): 1.0}
    return CausalGame(_AGENTS, _ACTIONS, {"row": row, "col": col}, _influence_diagram())


def coordination_game() -> CausalGame:
    """Both agents earn ``1`` iff their actions match. Two NE: (0, 0) and (1, 1)."""
    match = {(0, 0): 1.0, (1, 1): 1.0, (0, 1): 0.0, (1, 0): 0.0}
    return CausalGame(
        _AGENTS, _ACTIONS, {"row": dict(match), "col": dict(match)}, _influence_diagram()
    )


def matching_pennies() -> CausalGame:
    """Row wins (``1``) on a match, col wins on a mismatch. No pure-strategy NE."""
    row = {(0, 0): 1.0, (1, 1): 1.0, (0, 1): 0.0, (1, 0): 0.0}
    col = {(0, 0): 0.0, (1, 1): 0.0, (0, 1): 1.0, (1, 0): 1.0}
    return CausalGame(_AGENTS, _ACTIONS, {"row": row, "col": col}, _influence_diagram())
