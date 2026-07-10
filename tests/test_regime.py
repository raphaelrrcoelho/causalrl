"""Phase 0: Regime — a labeled data-generating configuration on Domain (§5.3)."""

import pytest

from causalrl.identification.id_algorithm import Domain
from causalrl.regime import Regime


def test_create_and_params() -> None:
    r = Regime.create("source", selection={"X"}, parameters={"gamma": 1.5, "alpha": 0.1})
    assert r.name == "source"
    assert r.selection == frozenset({"X"})
    assert r.params == {"gamma": 1.5, "alpha": 0.1}


def test_hashable_and_order_normalized() -> None:
    a = Regime.create("a", selection={"X"}, parameters={"x": 1, "y": 2})
    b = Regime.create("a", selection={"X"}, parameters={"y": 2, "x": 1})
    assert a == b
    assert hash(a) == hash(b)
    assert len({a, b}) == 1


def test_compose_union() -> None:
    a = Regime.create("a", selection={"X"}, parameters={"g": 1})
    b = Regime.create("b", selection={"Y"}, parameters={"h": 2})
    c = a | b
    assert c.selection == frozenset({"X", "Y"})
    assert c.params == {"g": 1, "h": 2}


def test_compose_conflict_detection() -> None:
    a = Regime.create("a", parameters={"g": 1})
    b = Regime.create("b", parameters={"g": 2})
    with pytest.raises(ValueError, match="conflict"):
        _ = a | b


def test_compose_same_value_ok() -> None:
    a = Regime.create("a", parameters={"g": 1})
    b = Regime.create("b", parameters={"g": 1})
    assert (a | b).params == {"g": 1}


def test_to_domain() -> None:
    d = Regime.create("source", selection={"X"}).to_domain()
    assert isinstance(d, Domain)
    assert d.name == "source"
    assert d.selection == frozenset({"X"})


def test_json_roundtrip() -> None:
    r = Regime.create("source", selection={"X", "Z"}, parameters={"gamma": 1.5, "flag": True})
    assert Regime.from_json(r.to_json()) == r
