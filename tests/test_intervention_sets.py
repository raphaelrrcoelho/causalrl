import pytest

from causalrl.exceptions import CausalGraphError
from causalrl.identification.intervention_sets import minimal_intervention_sets, pomis
from causalrl.scm.graph import CausalGraph


def fs(*names: str) -> frozenset[str]:
    return frozenset(names)


def test_pomis_single_edge_no_confounding():
    g = CausalGraph(directed_edges=[("X", "Y")])
    assert set(pomis(g, "Y")) == {fs("X")}


def test_pomis_bow_arc_is_mabuc():
    g = CausalGraph(directed_edges=[("X", "Y")], bidirected_edges=[("X", "Y")])
    assert set(pomis(g, "Y")) == {fs(), fs("X")}


def test_pomis_unconfounded_chain_is_parents_only():
    # Only Pa(Y) = {X}; the deeper ancestor Z is NOT a POMIS.
    g = CausalGraph(directed_edges=[("Z", "X"), ("X", "Y")])
    assert set(pomis(g, "Y")) == {fs("X")}


def test_pomis_demo_chain_root_confounder():
    # X1->X2->X3->Y with X1<->Y : observe (empty) stays possibly-optimal.
    g = CausalGraph(
        directed_edges=[("X1", "X2"), ("X2", "X3"), ("X3", "Y")],
        bidirected_edges=[("X1", "Y")],
    )
    assert set(pomis(g, "Y")) == {fs(), fs("X3")}


def test_pomis_mid_chain_confounder():
    # X1->X2->X3->Y with X2<->Y : two disjoint non-trivial POMISs, no empty set.
    g = CausalGraph(
        directed_edges=[("X1", "X2"), ("X2", "X3"), ("X3", "Y")],
        bidirected_edges=[("X2", "Y")],
    )
    assert set(pomis(g, "Y")) == {fs("X1"), fs("X3")}


def test_pomis_returns_sorted_list_of_frozensets():
    g = CausalGraph(directed_edges=[("X", "Y")], bidirected_edges=[("X", "Y")])
    result = pomis(g, "Y")
    assert isinstance(result, list)
    assert all(isinstance(s, frozenset) for s in result)
    assert result == sorted(result, key=lambda s: (len(s), sorted(s)))


def test_mis_chain_excludes_redundant_superset():
    # Z->X->Y: MIS = {empty, {X}, {Z}}; {Z,X} is NOT minimal (Z redundant under do(X)).
    g = CausalGraph(directed_edges=[("Z", "X"), ("X", "Y")])
    assert set(minimal_intervention_sets(g, "Y")) == {fs(), fs("X"), fs("Z")}


def test_pomis_subset_of_mis():
    g = CausalGraph(
        directed_edges=[("X1", "X2"), ("X2", "X3"), ("X3", "Y")],
        bidirected_edges=[("X1", "Y")],
    )
    assert set(pomis(g, "Y")) <= set(minimal_intervention_sets(g, "Y"))


def test_pomis_unknown_reward_raises():
    g = CausalGraph(directed_edges=[("X", "Y")])
    with pytest.raises(CausalGraphError):
        pomis(g, "Z")
