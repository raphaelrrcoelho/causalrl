"""Interventional causal discovery (Task 5, L1+L2): orienting via the invariance principle."""

from __future__ import annotations

import numpy as np
import pytest
from torch import Tensor
from torch.distributions import Bernoulli, Distribution, Uniform

from causalrl.discovery import discover, discover_interventional
from causalrl.exceptions import CausalGraphError
from causalrl.scm.graph import CausalGraph
from causalrl.scm.mechanisms import FunctionalMechanism, Mechanism
from causalrl.scm.scm import StructuralCausalModel


def _noisy_copy(parent: str):
    return FunctionalMechanism([parent], lambda pa, u: (pa[parent] + (u < 0.1).float()) % 2)


def _chain_scm() -> StructuralCausalModel:
    """A noisy chain ``A -> B -> C`` (each child copies its parent, flipped with prob 0.1)."""
    graph = CausalGraph(directed_edges=[("A", "B"), ("B", "C")])
    mechanisms: dict[str, Mechanism] = {
        "A": FunctionalMechanism([], lambda pa, u: u),
        "B": _noisy_copy("A"),
        "C": _noisy_copy("B"),
    }
    exogenous: dict[str, Distribution] = {
        "A": Bernoulli(0.5),
        "B": Uniform(0.0, 1.0),
        "C": Uniform(0.0, 1.0),
    }
    return StructuralCausalModel(graph, mechanisms, exogenous)


def _branch_scm() -> StructuralCausalModel:
    """A noisy tree ``A -> B`` with ``B -> C`` and ``B -> D`` (a fork below ``B``)."""
    graph = CausalGraph(directed_edges=[("A", "B"), ("B", "C"), ("B", "D")])
    mechanisms: dict[str, Mechanism] = {
        "A": FunctionalMechanism([], lambda pa, u: u),
        "B": _noisy_copy("A"),
        "C": _noisy_copy("B"),
        "D": _noisy_copy("B"),
    }
    exogenous: dict[str, Distribution] = {
        "A": Bernoulli(0.5),
        "B": Uniform(0.0, 1.0),
        "C": Uniform(0.0, 1.0),
        "D": Uniform(0.0, 1.0),
    }
    return StructuralCausalModel(graph, mechanisms, exogenous)


def _columns(samples: dict[str, Tensor]) -> dict[str, np.ndarray]:
    return {name: column.long().numpy() for name, column in samples.items()}


def _obs() -> dict[str, np.ndarray]:
    return _columns(_chain_scm().see(12_000, seed=0))


def _do(target: str, value: float) -> dict[str, np.ndarray]:
    return _columns(_chain_scm().do({target: value}).see(12_000, seed=1))


def test_observational_alone_leaves_the_chain_unoriented() -> None:
    # Baseline: PC cannot orient a chain with no v-structure (it returns the equivalence class).
    cpdag = discover(_obs(), ["A", "B", "C"])
    assert cpdag.directed_edges == frozenset()
    assert cpdag.undirected_edges == {frozenset({"A", "B"}), frozenset({"B", "C"})}


def test_intervening_on_the_middle_orients_the_whole_chain() -> None:
    # do(B): A is invariant (a parent) so A->B; C shifts (a child) so B->C. The chain is recovered.
    cpdag = discover_interventional(_obs(), {"B": _do("B", 1.0)}, ["A", "B", "C"])
    assert cpdag.undirected_edges == frozenset()
    assert cpdag.directed_edges == {("A", "B"), ("B", "C")}


def test_intervening_on_an_endpoint_then_meek_orients_the_chain() -> None:
    # do(A) orients A->B (B shifts); Meek R1 then propagates B->C (A is not adjacent to C).
    cpdag = discover_interventional(_obs(), {"A": _do("A", 1.0)}, ["A", "B", "C"])
    assert cpdag.undirected_edges == frozenset()
    assert cpdag.directed_edges == {("A", "B"), ("B", "C")}


def test_intervening_on_a_leaf_does_not_over_orient() -> None:
    # Tree A->B->{C,D}. do(C) only invalidates the B-C edge (B is invariant -> B->C). With no
    # v-structure created and nothing for Meek to propagate, A-B and B-D must stay undirected.
    obs = _columns(_branch_scm().see(12_000, seed=0))
    do_c = _columns(_branch_scm().do({"C": 1.0}).see(12_000, seed=1))
    cpdag = discover_interventional(obs, {"C": do_c}, ["A", "B", "C", "D"])
    assert ("B", "C") in cpdag.directed_edges
    assert cpdag.undirected_edges == {frozenset({"A", "B"}), frozenset({"B", "D"})}


def test_unknown_target_raises() -> None:
    with pytest.raises(CausalGraphError):
        discover_interventional(_obs(), {"Q": {}}, ["A", "B", "C"])


def test_an_empty_do_sample_refuses_rather_than_orienting_everything() -> None:
    # The shift test is an empirical total variation, and `_empirical_pmf` on an empty column is
    # {} -- so the distance to ANY observational marginal is 0.5, ten times the default
    # shift_threshold. Before the guard this oriented every edge incident to the target from zero
    # data and let Meek propagate outward from the fabrication. Discriminating: the assertion is
    # that it RAISES, so removing the guard fails the test (the call would return a fully oriented
    # CPDAG instead).
    empty = {name: np.array([], dtype=np.int64) for name in ("A", "B", "C")}
    with pytest.raises(CausalGraphError, match="0 sample"):
        discover_interventional(_obs(), {"B": empty}, ["A", "B", "C"])


def test_a_tiny_do_sample_refuses_rather_than_orienting_on_noise() -> None:
    # A single sample's pmf is a point mass, so its distance to the observational marginal is
    # 1 - p_obs(v) -- which clears the 0.05 threshold for almost any value. Small samples are the
    # same failure as the empty one, just less obvious.
    tiny = {name: column[:3] for name, column in _do("B", 1.0).items()}
    with pytest.raises(CausalGraphError, match="3 sample"):
        discover_interventional(_obs(), {"B": tiny}, ["A", "B", "C"])


def test_min_interventional_samples_is_lowerable_for_callers_who_accept_the_risk() -> None:
    # The guard refuses by default but does not forbid: a caller who knowingly accepts a small
    # sample can lower the floor. Pinning this keeps the guard a default rather than a wall.
    small = {name: column[:8] for name, column in _do("B", 1.0).items()}
    cpdag = discover_interventional(
        _obs(), {"B": small}, ["A", "B", "C"], min_interventional_samples=8
    )
    assert cpdag.undirected_edges == frozenset()
