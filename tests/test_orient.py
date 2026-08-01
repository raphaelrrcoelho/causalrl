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
    # CPDAG has A -> B (discovered), and undirected B-C, C-A.
    # Tiers say B < C < A, which would force B->C, C->A, A->B — a cycle.
    cpdag = CPDAG(
        ("A", "B", "C"),
        frozenset({("A", "B")}),
        frozenset({frozenset(("B", "C")), frozenset(("C", "A"))}),
    )
    with pytest.raises(CausalGraphError, match="tier-implied edge.*would create a cycle"):
        orient(cpdag, tiers=[["B"], ["C"], ["A"]])


def test_orient_multi_pass_loop_resolves_cascading_edges():
    # Two undirected edges P-X and X-Z; one directed Z->P.
    # Tiers say P < X,Z (same tier). First pass orients P->X by tier (progresses).
    # Second pass: X-Z remains undirected; now acyclicity decides because
    # resolving P->X created path Z->P->X, so X->Z would close a cycle.
    cpdag = CPDAG(
        ("P", "X", "Z"),
        frozenset({("Z", "P")}),
        frozenset({frozenset(("P", "X")), frozenset(("X", "Z"))}),
    )
    dag = orient(cpdag, tiers=[["P"], ["X", "Z"]])
    # All edges should be oriented: Z->P (existing), P->X (tier), Z->X (acyclicity)
    expected_edges = {("Z", "P"), ("P", "X"), ("Z", "X")}
    assert set(dag.directed_edges) == expected_edges
