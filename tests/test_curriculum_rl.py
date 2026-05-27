"""Curriculum-driven RL (Task 7): warm-start transfer through subtasks learns a sparse target.

The chain is laid out so action 0 ("left") is the all-zeros greedy default, which heads *away* from
the goal. Flat Q-learning on the sparse far goal therefore rarely reaches it within a fixed budget,
while a prerequisite-ordered curriculum of nearer subgoals does.
"""

from __future__ import annotations

import pytest

from causalrl.curriculum import causal_curriculum, curriculum_q_learning
from causalrl.exceptions import CausalGraphError
from causalrl.scm.graph import CausalGraph
from causalrl.shaping import TabularMDP, q_learning

_N = 12  # chain length; long enough that the far goal is hard to reach by chance


def _chain_task(goal: int) -> TabularMDP:
    """A length-_N chain. Action 0 moves left, 1 moves right; reward 1 on reaching ``goal``."""
    transitions: dict[tuple[int, int], int] = {}
    rewards: dict[tuple[int, int], float] = {}
    for s in range(_N):
        left, right = max(s - 1, 0), min(s + 1, _N - 1)
        transitions[(s, 0)], transitions[(s, 1)] = left, right
        rewards[(s, 0)] = 1.0 if left == goal else 0.0
        rewards[(s, 1)] = 1.0 if right == goal else 0.0
    return TabularMDP(_N, 2, transitions, rewards, frozenset({goal}), gamma=0.95)


def _reaches_goal(policy: dict[int, int], goal: int) -> bool:
    """Roll the greedy policy out from state 0; does it reach ``goal``?"""
    s = 0
    for _ in range(4 * _N):
        if s == goal:
            return True
        s = max(s - 1, 0) if policy[s] == 0 else min(s + 1, _N - 1)
    return s == goal


def test_curriculum_reaches_a_sparse_far_goal() -> None:
    goal = _N - 1
    # Subgoals ordered by the causal chain 0 -> 1 -> ... -> goal (the prerequisite structure).
    chain = CausalGraph(directed_edges=[(str(i), str(i + 1)) for i in range(goal)])
    order = causal_curriculum(chain, goal=str(goal))
    tasks = [_chain_task(int(node)) for node in order if int(node) > 0]

    policy = curriculum_q_learning(tasks, episodes_per_task=25, seed=0)
    assert _reaches_goal(policy, goal)


def test_curriculum_beats_flat_on_the_same_episode_budget() -> None:
    goal = _N - 1
    tasks = [_chain_task(g) for g in range(1, _N)]
    total_episodes = 25 * len(tasks)

    curriculum_policy = curriculum_q_learning(tasks, episodes_per_task=25, seed=0)
    flat_policy = q_learning(_chain_task(goal), episodes=total_episodes, seed=0)

    assert _reaches_goal(curriculum_policy, goal)
    assert not _reaches_goal(flat_policy, goal)  # the sparse far goal is out of flat's reach


def test_empty_curriculum_raises() -> None:
    with pytest.raises(CausalGraphError):
        curriculum_q_learning([], episodes_per_task=10)
