"""Deadline made load-bearing, and continuous intervention domains made actionable.

Deadline shipped in 3.0 with no consumer: it sat in InterventionalAgent's signature and every
implementation ignored it. These tests pin the two things that fixes -- an agent that returns its
incumbent when the clock runs out, and a certificate that says the sweep was partial.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from causalrl import Deadline, Interval, InterventionSpace
from causalrl.agents.anytime import Anytime, AnytimeInterventionSearch, SearchReport
from causalrl.intervention import Continuous, Discrete


def _dose_value(observation: Mapping[str, Any], intervention: Mapping[str, Any]) -> float:
    """Peaks at dose = 0.7; a search must find it rather than enumerate for it."""
    return -((float(intervention["dose"]) - 0.7) ** 2)


def _continuous_space() -> InterventionSpace:
    return InterventionSpace.create({"dose": Continuous(0.0, 2.0)})


def test_a_continuous_domain_cannot_be_enumerated_but_can_be_searched() -> None:
    """The gap: an interval has no arm list, so assignments() must refuse and search must work."""
    space = _continuous_space()
    with pytest.raises(TypeError, match="no enumerable value list"):
        list(space.assignments(["dose"]))

    agent = AnytimeInterventionSearch(_dose_value, rounds=8, candidates_per_round=32, seed=0)
    chosen = agent.act({}, space=space)

    assert space.permits(chosen)
    assert abs(float(chosen["dose"]) - 0.7) < 0.05


def test_an_expired_deadline_returns_the_incumbent_rather_than_overrunning() -> None:
    """The contract Deadline documented and nothing implemented."""
    agent = AnytimeInterventionSearch(_dose_value, rounds=64, candidates_per_round=64, seed=0)
    chosen = agent.act({}, space=_continuous_space(), deadline=Deadline.after(0.0))

    assert _continuous_space().permits(chosen)  # a usable answer, not an exception
    assert agent.last_search.exhausted is False
    assert agent.last_search.rounds < agent.last_search.rounds_planned


def test_a_truncated_search_is_hedged_and_an_exhausted_one_is_not() -> None:
    """Search completeness is reported like identification status: never silently assumed."""
    truncated = AnytimeInterventionSearch(_dose_value, rounds=64, candidates_per_round=64, seed=1)
    truncated.act({}, space=_continuous_space(), deadline=Deadline.after(0.0))
    hedge = truncated.last_search.certificate().hedge
    assert hedge is not None
    assert "budget-truncated" in hedge.reason
    assert hedge.downgraded_from == "exhaustive-search"

    complete = AnytimeInterventionSearch(_dose_value, rounds=3, candidates_per_round=4, seed=1)
    complete.act({}, space=_continuous_space())
    assert complete.last_search.exhausted is True
    assert complete.last_search.certificate().hedge is None


def test_it_satisfies_the_anytime_protocol() -> None:
    agent = AnytimeInterventionSearch(_dose_value, rounds=2, candidates_per_round=4, seed=2)
    assert isinstance(agent, Anytime)
    agent.act({}, space=_continuous_space())
    assert agent.incumbent() == agent.incumbent()
    assert agent.refine() is False  # converged: a generic driver must stop, not spin


def test_a_generous_deadline_does_not_truncate() -> None:
    agent = AnytimeInterventionSearch(_dose_value, rounds=4, candidates_per_round=8, seed=3)
    agent.act({}, space=_continuous_space(), deadline=Deadline.after(30.0))
    assert agent.last_search.exhausted is True


def test_mixed_discrete_and_continuous_domains_are_searched_together() -> None:
    def value(_obs: Mapping[str, Any], iv: Mapping[str, Any]) -> float:
        return float(iv["arm"]) - (float(iv["dose"]) - 1.0) ** 2

    space = InterventionSpace.create({"arm": Discrete((0, 1)), "dose": Continuous(0.0, 2.0)})
    agent = AnytimeInterventionSearch(value, rounds=10, candidates_per_round=32, seed=4)
    chosen = agent.act({}, space=space)

    assert chosen["arm"] == 1
    assert abs(float(chosen["dose"]) - 1.0) < 0.1
    assert space.permits(chosen)


def test_the_search_report_is_the_record_of_what_ran() -> None:
    agent = AnytimeInterventionSearch(_dose_value, rounds=3, candidates_per_round=5, seed=5)
    agent.act({}, space=_continuous_space())
    report = agent.last_search

    assert isinstance(report, SearchReport)
    assert report.rounds == 3
    assert report.candidates == 3 * 5 + 1  # the initial draw, plus every proposal
    assert report.best_value <= 0.0


def test_an_empty_space_yields_the_observational_regime() -> None:
    agent = AnytimeInterventionSearch(_dose_value, seed=6)
    assert agent.act({}, space=InterventionSpace()) == {}
    assert agent.last_search.exhausted is True


def test_projection_and_sampling_stay_inside_the_domain() -> None:
    space = InterventionSpace.create({"dose": Continuous(0.5, 1.5), "arm": Discrete((0, 2))})
    assert space.project({"dose": 9.0, "arm": 3})["dose"] == 1.5
    assert space.project({"dose": 9.0, "arm": 3})["arm"] == 2
    assert space.permits(space.project({"dose": -4.0, "arm": 1}))


def test_intersecting_a_continuous_domain_with_another() -> None:
    a = InterventionSpace.create({"dose": Continuous(0.0, 1.0)})
    b = InterventionSpace.create({"dose": Continuous(0.5, 2.0)})
    assert (a & b).domain("dose") == Continuous(0.5, 1.0)

    discrete = InterventionSpace.create({"dose": Discrete((0.25, 0.75, 5.0))})
    assert (a & discrete).domain("dose") == Discrete((0.25, 0.75))
    # No overlap at all: the variable drops out rather than surviving with an empty domain.
    assert (
        InterventionSpace.create({"dose": Continuous(9.0, 10.0)}) & discrete
    ).variables == frozenset()


def test_continuous_is_not_the_estimand_interval_type() -> None:
    """Deliberately distinct names: Interval bounds an estimand, Continuous is an action domain."""
    assert Continuous is not Interval
    with pytest.raises(ValueError, match="low <= high"):
        Continuous(1.0, 0.0)
