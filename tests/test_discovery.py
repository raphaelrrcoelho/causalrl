"""Causal discovery: the CMI independence test and PC structure recovery."""

from __future__ import annotations

import numpy as np
import pytest
from torch.distributions import Bernoulli, Distribution, Uniform

from causalrl.discovery import (
    CPDAG,
    _apply_meek_rules,
    conditional_mutual_information,
    discover,
)
from causalrl.envs.suite.discovery import sample_discovery_data
from causalrl.exceptions import CausalGraphError
from causalrl.scm.graph import CausalGraph
from causalrl.scm.mechanisms import FunctionalMechanism, Mechanism
from causalrl.scm.scm import StructuralCausalModel


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


def _chain_data(n: int = 12_000, seed: int = 0) -> dict[str, np.ndarray]:
    """Samples from a noisy chain ``A -> B -> C`` (no collider, no determinism)."""
    graph = CausalGraph(directed_edges=[("A", "B"), ("B", "C")])
    mechanisms: dict[str, Mechanism] = {
        "A": FunctionalMechanism([], lambda pa, u: u),
        "B": FunctionalMechanism(["A"], lambda pa, u: (pa["A"] + (u < 0.1).float()) % 2),
        "C": FunctionalMechanism(["B"], lambda pa, u: (pa["B"] + (u < 0.1).float()) % 2),
    }
    exogenous: dict[str, Distribution] = {
        "A": Bernoulli(0.5),
        "B": Uniform(0.0, 1.0),
        "C": Uniform(0.0, 1.0),
    }
    samples = StructuralCausalModel(graph, mechanisms, exogenous).see(n, seed=seed)
    return {name: column.long().numpy() for name, column in samples.items()}


def test_discover_returns_equivalence_class_for_a_chain() -> None:
    # A -> B -> C has no v-structure, so PC cannot orient it: the CPDAG keeps both edges undirected
    # (the Markov equivalence class), and the bridge refuses to fabricate a DAG.
    cpdag = discover(_chain_data(), ["A", "B", "C"])
    assert cpdag.directed_edges == frozenset()
    assert cpdag.undirected_edges == {frozenset({"A", "B"}), frozenset({"B", "C"})}
    with pytest.raises(CausalGraphError):
        cpdag.to_causal_graph()


def test_meek_rule_r2_orients_a_chain() -> None:
    # R2: a directed path A -> B -> C with A - C undirected forces A -> C (avoiding a cycle).
    directed = {("A", "B"), ("B", "C")}
    undirected = {frozenset({"A", "C"})}
    _apply_meek_rules(["A", "B", "C"], directed, undirected)
    assert ("A", "C") in directed
    assert undirected == set()


def test_meek_rule_r3_orients_into_a_common_child() -> None:
    # R3: non-adjacent C, D both point into B with A - B, A - C, A - D undirected forces A -> B.
    directed = {("C", "B"), ("D", "B")}
    undirected = {frozenset({"A", "B"}), frozenset({"A", "C"}), frozenset({"A", "D"})}
    _apply_meek_rules(["A", "B", "C", "D"], directed, undirected)
    assert ("A", "B") in directed
