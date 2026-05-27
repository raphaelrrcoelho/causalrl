from causalrl.scm.graph import CausalGraph


def _chain() -> CausalGraph:
    # Z -> X -> Y, with a bidirected X <-> Y
    return CausalGraph(
        directed_edges=[("Z", "X"), ("X", "Y")],
        bidirected_edges=[("X", "Y")],
    )


def test_ancestors_includes_inputs():
    g = _chain()
    assert g.ancestors("Y") == {"X", "Y", "Z"}
    assert g.ancestors("X") == {"X", "Z"}
    assert g.ancestors({"X", "Y"}) == {"X", "Y", "Z"}


def test_descendants_are_strict():
    g = _chain()
    assert g.descendants("Z") == {"X", "Y"}
    assert g.descendants("Y") == set()
    assert g.descendants({"X"}) == {"Y"}


def test_induced_subgraph_keeps_only_internal_edges():
    g = _chain()
    h = g.induced_subgraph({"X", "Y"})
    assert set(h.nodes) == {"X", "Y"}
    assert h.parents("Y") == ["X"]  # X -> Y kept
    assert h.parents("X") == []  # Z -> X dropped (Z excluded)
    assert h.is_confounded("X", "Y") is True  # bidirected kept (both endpoints in)


def test_do_mutilate_removes_incoming_and_bidirected_at_node():
    g = _chain()
    h = g.do_mutilate({"X"})
    assert h.parents("X") == []  # Z -> X removed (incoming to X)
    assert h.parents("Y") == ["X"]  # X -> Y kept (not incoming to X)
    assert h.is_confounded("X", "Y") is False  # bidirected at X removed
    assert set(h.nodes) == {"X", "Y", "Z"}  # node set preserved


def test_do_mutilate_empty_is_identity_like():
    g = _chain()
    h = g.do_mutilate(set())
    assert h.parents("X") == ["Z"]
    assert h.is_confounded("X", "Y") is True
