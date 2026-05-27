"""Causal discovery: the CMI independence test and PC structure recovery."""

from __future__ import annotations

import numpy as np
import pytest

from causalrl.discovery import CPDAG, conditional_mutual_information, discover
from causalrl.envs.suite.discovery import sample_discovery_data
from causalrl.exceptions import CausalGraphError


def test_cmi_near_zero_for_independent_columns() -> None:
    rng = np.random.default_rng(0)
    data = {"A": rng.integers(0, 2, size=20_000), "B": rng.integers(0, 2, size=20_000)}
    assert conditional_mutual_information(data, "A", "B", ()) < 0.005


def test_cmi_positive_for_dependent_columns() -> None:
    rng = np.random.default_rng(1)
    a = rng.integers(0, 2, size=20_000)
    data = {"A": a, "B": a.copy()}  # B == A
    assert conditional_mutual_information(data, "A", "B", ()) > 0.5


def test_cmi_reveals_collider_dependence() -> None:
    data = sample_discovery_data(n=20_000, seed=2)
    # X and Y are marginally independent but become dependent given the collider Z.
    assert conditional_mutual_information(data, "X", "Y", ()) < 0.01
    assert conditional_mutual_information(data, "X", "Y", ("Z",)) > 0.02


def test_discover_recovers_the_cpdag() -> None:
    data = sample_discovery_data(n=10_000, seed=0)
    cpdag = discover(data, ["X", "Y", "Z", "W"])
    assert cpdag.undirected_edges == frozenset()  # fully oriented
    assert cpdag.directed_edges == frozenset({("X", "Z"), ("Y", "Z"), ("Z", "W")})


def test_to_causal_graph_bridges_to_planning() -> None:
    data = sample_discovery_data(n=10_000, seed=0)
    graph = discover(data, ["X", "Y", "Z", "W"]).to_causal_graph()
    assert set(graph.parents("Z")) == {"X", "Y"}
    assert graph.parents("W") == ["Z"]


def test_to_causal_graph_raises_when_unoriented() -> None:
    cpdag = CPDAG(("A", "B"), frozenset(), frozenset({frozenset({"A", "B"})}))
    with pytest.raises(CausalGraphError):
        cpdag.to_causal_graph()


def test_unknown_variable_raises() -> None:
    data = sample_discovery_data(n=1_000, seed=0)
    with pytest.raises(CausalGraphError):
        discover(data, ["X", "Q"])
