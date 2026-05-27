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

import numpy as np

from causalrl.exceptions import CausalGraphError
from causalrl.scm.graph import CausalGraph
from causalrl.shaping import TabularMDP

__all__ = [
    "PrerequisiteLearner",
    "causal_curriculum",
    "curriculum_q_learning",
    "is_valid_curriculum",
]


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


def curriculum_q_learning(
    tasks: Sequence[TabularMDP],
    *,
    episodes_per_task: int,
    alpha: float = 0.5,
    epsilon: float = 0.1,
    seed: int | None = None,
) -> dict[int, int]:
    """Learn the target policy by Q-learning through a curriculum of subtasks, easiest first.

    ``tasks`` is ordered from the simplest subtask to the target (the last element); all tasks share
    the same state and action spaces. The Q-table is carried forward between stages (warm-start
    transfer), so value learned on the easy subtasks bootstraps the harder ones. This is the causal
    curriculum applied to RL: order subgoals by prerequisite structure (see
    :func:`causal_curriculum`), then train in that order to reach a target policy that flat learning
    on the sparse target alone struggles to find. Returns the greedy policy on the target task.

    Faithful to Y. Bengio, J. Louradour, R. Collobert, J. Weston, *Curriculum Learning*, ICML 2009;
    the Q-learning update matches :func:`causalrl.shaping.q_learning`. No external code is ported.
    """
    if not tasks:
        raise CausalGraphError("curriculum must contain at least one task")
    target = tasks[-1]
    rng = np.random.default_rng(seed)
    q = np.zeros((target.n_states, target.n_actions))
    for task in tasks:
        for _ in range(episodes_per_task):
            s = 0
            for _ in range(4 * task.n_states):
                if s in task.terminals:
                    break
                if rng.random() < epsilon:
                    a = int(rng.integers(0, task.n_actions))
                else:
                    a = int(np.argmax(q[s]))
                s_next = task.transitions[(s, a)]
                bootstrap = 0.0 if s_next in task.terminals else float(np.max(q[s_next]))
                q[s, a] += alpha * (task.rewards[(s, a)] + task.gamma * bootstrap - q[s, a])
                s = s_next
    return {s: int(np.argmax(q[s])) for s in range(target.n_states) if s not in target.terminals}
