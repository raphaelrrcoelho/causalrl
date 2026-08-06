"""Intervention / InterventionSpace — the set-valued action vocabulary."""

import pytest

from causalrl.intervention import InterventionSpace, canonical


def test_canonical_is_order_independent_and_hashable() -> None:
    a = canonical({"A": 1, "B": 0})
    b = canonical({"B": 0, "A": 1})
    assert a == b
    assert len({a, b}) == 1


def test_create_normalizes_order_so_equal_spaces_compare_equal() -> None:
    x = InterventionSpace.create({"B": [0, 1], "A": [0, 1]})
    y = InterventionSpace.create({"A": [0, 1], "B": [0, 1]})
    assert x == y
    assert hash(x) == hash(y)
    assert x.variables == frozenset({"A", "B"})


def test_empty_domain_is_refused() -> None:
    # A variable that is manipulable but has no value it can take would make every intervention
    # set containing it unsatisfiable while still counting as admissible.
    with pytest.raises(ValueError, match="empty domain"):
        InterventionSpace.create({"A": []})


def test_values_names_the_variable_it_cannot_find() -> None:
    space = InterventionSpace.create({"A": [0, 1]})
    assert space.values("A") == (0, 1)
    with pytest.raises(KeyError, match="not manipulable"):
        space.values("B")


def test_permits_checks_both_variable_and_value() -> None:
    space = InterventionSpace.create({"A": [0, 1], "B": ["on", "off"]})
    assert space.permits({"A": 1, "B": "on"})
    assert space.permits({})  # the observational regime is always admissible
    assert not space.permits({"A": 2})  # value outside the domain
    assert not space.permits({"C": 0})  # variable not manipulable here


def test_restrict_keeps_only_the_named_variables() -> None:
    space = InterventionSpace.create({"A": [0, 1], "B": [0, 1], "C": [0]})
    assert space.restrict(["A", "C", "Z"]).variables == frozenset({"A", "C"})


def test_intersection_keeps_shared_variables_and_shared_values() -> None:
    x = InterventionSpace.create({"A": [0, 1, 2], "B": [0, 1], "C": [0]})
    y = InterventionSpace.create({"A": [1, 2, 3], "B": [7], "D": [0]})
    both = x & y
    assert both.variables == frozenset({"A"})  # B's domains do not overlap, so B drops out
    assert both.values("A") == (1, 2)


def test_assignments_enumerates_the_arms_of_an_intervention_set() -> None:
    space = InterventionSpace.create({"A": [0, 1], "B": ["x", "y"]})
    arms = list(space.assignments(["A", "B"]))
    assert len(arms) == 4
    assert {canonical(arm) for arm in arms} == {
        canonical({"A": a, "B": b}) for a in (0, 1) for b in ("x", "y")
    }


def test_empty_set_yields_exactly_the_observational_regime() -> None:
    # frozenset() is what pomis emits when not intervening is possibly optimal; it must map to
    # one arm (the empty intervention), not to zero arms.
    space = InterventionSpace.create({"A": [0, 1]})
    assert list(space.assignments([])) == [{}]


def test_assignments_names_a_variable_that_is_not_manipulable() -> None:
    space = InterventionSpace.create({"A": [0, 1]})
    with pytest.raises(KeyError, match="not manipulable"):
        list(space.assignments(["A", "B"]))


def test_assignments_refuses_an_explosive_product() -> None:
    space = InterventionSpace.create({name: list(range(10)) for name in "ABCDEF"})
    with pytest.raises(ValueError, match="_MAX_ASSIGNMENTS"):
        list(space.assignments("ABCDEF"))
