import pytest

from causalrl.discovery import CPDAG, orient
from causalrl.exceptions import CausalGraphError


def test_orient_resolves_undirected_edge_by_tier():
    cpdag = CPDAG(("Z", "A"), frozenset(), frozenset({frozenset(("Z", "A"))}))
    dag = orient(cpdag, tiers=[["Z"], ["A"]])
    assert dag.directed_edges == [("Z", "A")]


def test_orient_passes_through_fully_oriented_cpdag():
    cpdag = CPDAG(("X", "Z"), frozenset({("X", "Z")}), frozenset())
    dag = orient(cpdag)
    assert dag.directed_edges == [("X", "Z")]
    assert sorted(dag.nodes) == ["X", "Z"]


def test_orient_uses_acyclicity_when_tiers_are_silent():
    # A -> B already directed; B - C undirected; C - A undirected.
    # Orienting C -> A then A -> B forces B ... C acyclic: only C -> B closes no cycle.
    cpdag = CPDAG(
        ("A", "B", "C"),
        frozenset({("A", "B"), ("C", "A")}),
        frozenset({frozenset(("B", "C"))}),
    )
    dag = orient(cpdag)
    assert ("C", "B") in dag.directed_edges
    assert ("B", "C") not in dag.directed_edges


def test_orient_raises_on_unresolvable_edge_and_names_the_escape_hatch():
    cpdag = CPDAG(("A", "B"), frozenset(), frozenset({frozenset(("A", "B"))}))
    with pytest.raises(CausalGraphError, match="fit_scm_mec"):
        orient(cpdag)


def test_orient_raises_when_tiers_omit_a_variable():
    cpdag = CPDAG(("Z", "A"), frozenset(), frozenset({frozenset(("Z", "A"))}))
    with pytest.raises(CausalGraphError, match="not covered by tiers"):
        orient(cpdag, tiers=[["Z"]])


def test_orient_raises_when_tiers_contradict_discovered_structure_causing_cycle():
    # CPDAG with A->B existing. Undirected edges: A-C, B-C.
    # Tiers: [["B"], ["C"], ["A"]] means B(0) < C(1) < A(2).
    # Sorted pending order: (A,C), (B,C).
    # Process (A,C): rank[A]=2 > rank[C]=1, so force C->A (lower tier first).
    #   directed = {A->B, C->A}. No cycle.
    # Process (B,C): rank[B]=0 < rank[C]=1, so force B->C.
    #   Check: path from C to B? C->A->B (via existing edges). Cycle!
    #   Error: tier-implied edge B -> C would create a cycle: B -> C -> A -> B
    cpdag = CPDAG(
        ("A", "B", "C"),
        frozenset({("A", "B")}),
        frozenset({frozenset(("A", "C")), frozenset(("B", "C"))}),
    )
    # Assert on both the edge and the cycle path to catch regressions
    # (the cycle path suffix is critical to identify which edges conflict)
    with pytest.raises(
        CausalGraphError, match=r"tier-implied edge B -> C would create a cycle: B -> C -> A -> B"
    ):
        orient(cpdag, tiers=[["B"], ["C"], ["A"]])


def test_orient_multi_pass_loop_resolves_cascading_edges():
    # Multi-pass test: verifies the deferred-list second-iteration path is exercised.
    # Undirected edges: A-B (same tier), B-C (different tier).
    # Existing edge: C->A.
    # Tiers: [["A", "B"], ["C"]]
    # Sorted pending order: (A,B), (B,C).
    # Pass 1: (A,B) can't resolve by tier or acyclicity (both directions OK when C->A alone),
    #         so defer. (B,C) tier-forced B->C; now directed = {C->A, B->C}.
    #         progressed=True.
    # Pass 2: (A,B) now resolvable: A->B creates path B->C->A->B (cycle!);
    #         B->A has no cycle. Acyclicity forces B->A.
    # Assert all edges resolved: {C->A, B->C, B->A}.
    cpdag = CPDAG(
        ("A", "B", "C"),
        frozenset({("C", "A")}),
        frozenset({frozenset(("A", "B")), frozenset(("B", "C"))}),
    )
    dag = orient(cpdag, tiers=[["A", "B"], ["C"]])
    expected_edges = {("C", "A"), ("B", "C"), ("B", "A")}
    assert set(dag.directed_edges) == expected_edges
