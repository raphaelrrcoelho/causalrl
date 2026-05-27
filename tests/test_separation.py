"""S-node separation helpers shared by the transportability code (lifted from ``transport``)."""

from __future__ import annotations

from causalrl.identification._separation import canonical_dag, d_separated, selection_nodes
from causalrl.scm.graph import CausalGraph


def test_selection_node_breaks_invariance() -> None:
    g = CausalGraph(directed_edges=[("Z", "X"), ("Z", "Y"), ("X", "Y")])
    # S->Z makes Z's mechanism domain-specific: S is NOT d-separated from Y given X.
    assert not d_separated(g, selection_nodes(["Z"]), {"Y"}, {"X"}, selection=["Z"])
    # but S is separated from Y once we also condition on Z.
    assert d_separated(g, selection_nodes(["Z"]), {"Y"}, {"X", "Z"}, selection=["Z"])


def test_canonical_dag_projects_bidirected_to_latent() -> None:
    g = CausalGraph(directed_edges=[("X", "Y")], bidirected_edges=[("X", "Y")])
    dag = canonical_dag(g, [])
    assert any(n.startswith("__L__::") for n in dag.nodes)
