"""InterventionalAgent + ScalarAgentAdapter — lifting an arm-indexed agent to set-valued actions."""

from typing import Any

import pytest

from causalrl.agents.base import Agent
from causalrl.agents.interventional import InterventionalAgent, ScalarAgentAdapter
from causalrl.deadline import Deadline
from causalrl.intervention import InterventionSpace


class _FixedAgent(Agent):
    """An arm-indexed agent that always names the same arm and records what it is told."""

    def __init__(self, action: int) -> None:
        self.action = action
        self.updates: list[tuple[int, float]] = []

    def act(self, observation: dict[str, Any]) -> int:
        return self.action

    def update(self, observation: dict[str, Any], action: int, reward: float) -> None:
        self.updates.append((action, reward))


ARMS = [{"A": 0}, {"A": 1}, {"A": 1, "B": 1}]
SPACE = InterventionSpace.create({"A": [0, 1], "B": [0, 1]})


def test_adapter_maps_the_chosen_index_to_its_intervention() -> None:
    adapter = ScalarAgentAdapter(_FixedAgent(1), ARMS)
    assert adapter.act({}, space=SPACE) == {"A": 1}


def test_adapter_is_an_interventional_agent() -> None:
    assert isinstance(ScalarAgentAdapter(_FixedAgent(0), ARMS), InterventionalAgent)


def test_adapter_round_trips_the_reward_to_the_right_arm_index() -> None:
    inner = _FixedAgent(2)
    adapter = ScalarAgentAdapter(inner, ARMS)
    intervention = adapter.act({}, space=SPACE)
    adapter.update({}, intervention, 1.5)
    assert inner.updates == [(2, 1.5)]


def test_update_is_indifferent_to_key_order() -> None:
    inner = _FixedAgent(2)
    adapter = ScalarAgentAdapter(inner, ARMS)
    adapter.update({}, {"B": 1, "A": 1}, 2.0)  # same intervention, reversed insertion order
    assert inner.updates == [(2, 2.0)]


def test_deadline_is_accepted_and_ignored() -> None:
    # The wrapped scalar Agent.act takes no budget, so there is nothing to forward it to; the
    # adapter must still accept one rather than making callers special-case it.
    adapter = ScalarAgentAdapter(_FixedAgent(0), ARMS)
    assert adapter.act({}, space=SPACE, deadline=Deadline.after(0.0)) == {"A": 0}


def test_an_inadmissible_arm_raises_rather_than_being_silently_substituted() -> None:
    narrowed = InterventionSpace.create({"A": [0]})  # arm 1 sets A=1, which this space forbids
    adapter = ScalarAgentAdapter(_FixedAgent(1), ARMS)
    with pytest.raises(ValueError, match="not admissible"):
        adapter.act({}, space=narrowed)


def test_out_of_range_index_names_the_disagreement() -> None:
    adapter = ScalarAgentAdapter(_FixedAgent(7), ARMS)
    with pytest.raises(IndexError, match="outside the codebook"):
        adapter.act({}, space=SPACE)


def test_update_with_an_unknown_intervention_raises() -> None:
    adapter = ScalarAgentAdapter(_FixedAgent(0), ARMS)
    with pytest.raises(KeyError, match="not in this adapter's codebook"):
        adapter.update({}, {"A": 5}, 1.0)


def test_duplicate_arms_are_refused() -> None:
    with pytest.raises(ValueError, match="duplicate intervention"):
        ScalarAgentAdapter(_FixedAgent(0), [{"A": 0}, {"A": 0}])


def test_empty_arm_list_is_refused() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        ScalarAgentAdapter(_FixedAgent(0), [])


def test_arms_property_does_not_alias_internal_state() -> None:
    adapter = ScalarAgentAdapter(_FixedAgent(0), ARMS)
    exposed = adapter.arms
    dict(exposed[0]).clear()
    assert adapter.arms[0] == {"A": 0}


def test_check_permitted_reports_the_offending_entry() -> None:
    with pytest.raises(ValueError, match="'A': 9"):
        InterventionalAgent.check_permitted({"A": 9}, SPACE)
