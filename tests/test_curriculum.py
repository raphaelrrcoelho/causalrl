"""Causal curriculum: topological ordering, validity, and prerequisite-gated mastery."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from causalrl.curriculum import PrerequisiteLearner, causal_curriculum, is_valid_curriculum
from causalrl.envs.suite.curriculum import make_skill_chain, make_skill_diamond
from causalrl.exceptions import CausalGraphError


def test_chain_curriculum_order_and_validity() -> None:
    graph, goal = make_skill_chain(4)
    assert causal_curriculum(graph, goal) == ["S0", "S1", "S2", "S3"]
    assert is_valid_curriculum(graph, ["S0", "S1", "S2", "S3"])
    assert not is_valid_curriculum(graph, ["S3", "S2", "S1", "S0"])


def test_diamond_curriculum_is_valid_topological() -> None:
    graph, goal = make_skill_diamond()
    order = causal_curriculum(graph, goal)
    assert order[0] == "S0"
    assert order[-1] == "S3"
    assert is_valid_curriculum(graph, order)


def test_learner_masters_goal_on_causal_curriculum() -> None:
    graph, goal = make_skill_chain(5)
    learner = PrerequisiteLearner(graph)
    mastered = learner.train(causal_curriculum(graph, goal))
    assert learner.masters(goal)
    assert mastered == frozenset({"S0", "S1", "S2", "S3", "S4"})


def test_learner_fails_goal_on_reversed_curriculum() -> None:
    graph, goal = make_skill_chain(5)
    learner = PrerequisiteLearner(graph)
    mastered = learner.train(list(reversed(causal_curriculum(graph, goal))))
    assert not learner.masters(goal)
    assert mastered == frozenset({"S0"})  # only the prerequisite-free root is learned


@given(st.permutations(["S0", "S1", "S2", "S3"]))
def test_mastery_iff_valid_curriculum(order: list[str]) -> None:
    graph, _ = make_skill_diamond()
    learner = PrerequisiteLearner(graph)
    mastered = learner.train(order)
    mastered_all = mastered == frozenset({"S0", "S1", "S2", "S3"})
    assert mastered_all == is_valid_curriculum(graph, order)


def test_unknown_goal_raises() -> None:
    graph, _ = make_skill_chain(3)
    with pytest.raises(CausalGraphError):
        causal_curriculum(graph, "S99")
