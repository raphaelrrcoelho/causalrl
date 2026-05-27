"""Skill-prerequisite graphs for causal curriculum learning (taxonomy Task 7).

Each builder returns a ``CausalGraph`` whose edges are prerequisite relations (``parent -> child``
means the parent must be mastered before the child) plus the goal skill.
"""

from __future__ import annotations

from causalrl.scm.graph import CausalGraph


def make_skill_chain(length: int = 4) -> tuple[CausalGraph, str]:
    """A prerequisite chain ``S0 -> S1 -> ... -> S{length-1}``; the goal is the last skill."""
    if length < 2:
        raise ValueError("chain length must be at least 2")
    skills = [f"S{i}" for i in range(length)]
    edges = [(skills[i], skills[i + 1]) for i in range(length - 1)]
    return CausalGraph(directed_edges=edges), skills[-1]


def make_skill_diamond() -> tuple[CausalGraph, str]:
    """Branching prerequisites ``S0 -> {S1, S2} -> S3``; the goal is ``S3``."""
    edges = [("S0", "S1"), ("S0", "S2"), ("S1", "S3"), ("S2", "S3")]
    return CausalGraph(directed_edges=edges), "S3"
