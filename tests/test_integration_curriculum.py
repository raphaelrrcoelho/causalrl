"""Headline: the causal curriculum masters the goal; a prerequisite-violating order does not."""

from __future__ import annotations

from collections.abc import Callable

from causalrl.curriculum import PrerequisiteLearner, causal_curriculum
from causalrl.envs.suite.curriculum import make_skill_chain, make_skill_diamond
from causalrl.scm.graph import CausalGraph


def test_causal_order_masters_goal_violating_order_does_not() -> None:
    builders: list[Callable[[], tuple[CausalGraph, str]]] = [
        lambda: make_skill_chain(5),
        make_skill_diamond,
    ]
    for build in builders:
        graph, goal = build()
        curriculum = causal_curriculum(graph, goal)

        on_curriculum = PrerequisiteLearner(graph)
        on_curriculum.train(curriculum)
        assert on_curriculum.masters(goal)

        violating = PrerequisiteLearner(graph)
        violating.train(list(reversed(curriculum)))
        assert not violating.masters(goal)
