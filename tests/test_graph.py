import pytest

from causalrl.exceptions import CausalGraphError
from causalrl.scm.graph import CausalGraph


def make_chain() -> CausalGraph:
    # Z -> X -> Y
    return CausalGraph(directed_edges=[("Z", "X"), ("X", "Y")])


def test_nodes_and_parents():
    g = make_chain()
    assert set(g.nodes) == {"Z", "X", "Y"}
    assert g.parents("Y") == ["X"]
    assert g.parents("Z") == []


def test_children():
    g = make_chain()
    assert g.children("Z") == ["X"]


def test_topological_order():
    g = make_chain()
    order = g.topological_order()
    assert order.index("Z") < order.index("X") < order.index("Y")


def test_cycle_rejected():
    with pytest.raises(CausalGraphError):
        CausalGraph(directed_edges=[("A", "B"), ("B", "A")])


def test_unknown_node_raises():
    g = make_chain()
    with pytest.raises(CausalGraphError):
        g.parents("Q")


def test_confounding_and_c_components():
    # X <-> Y bidirected (unobserved confounder), plus isolated Z
    g = CausalGraph(
        directed_edges=[("X", "Y")], bidirected_edges=[("X", "Y")], nodes=["X", "Y", "Z"]
    )
    assert g.is_confounded("X", "Y") is True
    assert g.is_confounded("X", "Z") is False
    comps = {frozenset(c) for c in g.c_components()}
    assert frozenset({"X", "Y"}) in comps
    assert frozenset({"Z"}) in comps
