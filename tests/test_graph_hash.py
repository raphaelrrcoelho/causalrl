"""Phase 0: graphs/ shim + graph_hash() (canonical ADMG fingerprint for provenance, I8)."""

from causalrl.graphs import graph_hash
from causalrl.scm.graph import CausalGraph


def _g(
    directed: list[tuple[str, str]],
    bidirected: list[tuple[str, str]] | None = None,
    nodes: list[str] | None = None,
) -> CausalGraph:
    return CausalGraph(directed, bidirected, nodes=nodes)


def test_graph_hash_is_hex_sha256() -> None:
    h = graph_hash(_g([("X", "Y")]))
    assert isinstance(h, str)
    assert len(h) == 64
    int(h, 16)  # parses as hexadecimal


def test_graph_hash_stable_across_constructions() -> None:
    assert graph_hash(_g([("X", "Y"), ("Y", "Z")])) == graph_hash(_g([("X", "Y"), ("Y", "Z")]))


def test_graph_hash_edge_order_independent() -> None:
    assert graph_hash(_g([("X", "Y"), ("Y", "Z")])) == graph_hash(_g([("Y", "Z"), ("X", "Y")]))


def test_graph_hash_direction_sensitive() -> None:
    assert graph_hash(_g([("X", "Y")])) != graph_hash(_g([("Y", "X")]))


def test_graph_hash_detects_added_edge() -> None:
    assert graph_hash(_g([("X", "Y")])) != graph_hash(_g([("X", "Y"), ("Y", "Z")]))


def test_graph_hash_isolated_node_matters() -> None:
    assert graph_hash(_g([("X", "Y")])) != graph_hash(_g([("X", "Y")], nodes=["X", "Y", "Z"]))


def test_graph_hash_bidirected_symmetric() -> None:
    a = graph_hash(_g([("X", "Y")], bidirected=[("X", "Y")]))
    b = graph_hash(_g([("X", "Y")], bidirected=[("Y", "X")]))
    assert a == b


def test_graph_hash_bidirected_distinct_from_directed() -> None:
    assert graph_hash(_g([("X", "Y")])) != graph_hash(_g([("X", "Y")], bidirected=[("X", "Y")]))


def test_causalgraph_reexported_from_graphs() -> None:
    from causalrl.graphs import CausalGraph as CG

    assert CG is CausalGraph
