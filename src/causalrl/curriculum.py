"""Causal curriculum learning (taxonomy Task 7).

Sequence skill acquisition by the causal structure: a skill is learnable only once the skills it
causally depends on (its parents / prerequisites) are mastered. A curriculum that follows a
topological order of the causal DAG lets a learner reach the goal; an order that violates
prerequisites strands the blocked skills (and everything downstream of them).

Faithful to Y. Bengio, J. Louradour, R. Collobert, J. Weston, *Curriculum Learning*, ICML 2009 — the
causal contribution is the topological ordering rule. No external code is ported.
"""

from __future__ import annotations

from collections.abc import Sequence

from causalrl.exceptions import CausalGraphError
from causalrl.scm.graph import CausalGraph

__all__ = ["PrerequisiteLearner", "causal_curriculum", "is_valid_curriculum"]


def causal_curriculum(graph: CausalGraph, goal: str | None = None) -> list[str]:
    """A curriculum (skill order) respecting the causal structure: a topological order in which
    every parent (prerequisite) precedes its children. With ``goal``, restrict to the goal and its
    ancestors — the skills the goal depends on — still in topological order."""
    order = graph.topological_order()
    if goal is None:
        return order
    if goal not in graph.nodes:
        raise CausalGraphError(f"unknown goal skill: {goal!r}")
    relevant = graph.ancestors(goal)  # inclusive of goal
    return [node for node in order if node in relevant]


def is_valid_curriculum(graph: CausalGraph, order: Sequence[str]) -> bool:
    """Whether ``order`` respects prerequisites: each skill appears after every parent of it that is
    also in ``order``."""
    present = set(order)
    seen: set[str] = set()
    for skill in order:
        prerequisites = set(graph.parents(skill)) & present
        if not prerequisites <= seen:
            return False
        seen.add(skill)
    return True


class PrerequisiteLearner:
    """Causally-gated skill acquisition: walking the curriculum left to right, a skill is mastered
    iff all of its parents (prerequisites) are already mastered. Deterministic and order-faithful,
    so the effect of the ordering is exactly readable from the mastered set."""

    def __init__(self, graph: CausalGraph) -> None:
        self._graph = graph
        self._mastered: set[str] = set()

    def train(self, curriculum: Sequence[str]) -> frozenset[str]:
        """Process the curriculum once and return the set of mastered skills."""
        self._mastered = set()
        for skill in curriculum:
            if set(self._graph.parents(skill)) <= self._mastered:
                self._mastered.add(skill)
        return frozenset(self._mastered)

    def masters(self, skill: str) -> bool:
        """Whether ``skill`` was mastered by the most recent :meth:`train`."""
        return skill in self._mastered
