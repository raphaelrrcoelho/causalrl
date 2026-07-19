"""Directed graphs that may contain cycles (experimental; plan §11).

The shipped :class:`eqcert.scm.graph.CausalGraph` is acyclic by construction. Feedback systems --
control loops, coupled steady-state models -- are *cyclic* SCMs, so their graphs need a type that
permits directed cycles. :class:`CyclicCausalGraph` is that type: a directed graph (optionally with
bidirected latent-confounding edges) carrying the strongly-connected-component structure the cyclic
Markov theory is defined over, plus the Forré-Mooij *acyclification* that reduces sigma-separation
to ordinary m-separation.
"""

from __future__ import annotations

from collections.abc import Iterable

import networkx as nx

from eqcert.scm.graph import CausalGraph

Edge = tuple[str, str]


class CyclicCausalGraph:
    """A directed (possibly cyclic) graph with optional bidirected latent-confounding edges.

    Unlike :class:`~eqcert.scm.graph.CausalGraph` this imposes **no acyclicity constraint**. Use
    :meth:`is_acyclic` to test, :meth:`strongly_connected_components` for the cycle structure, and
    :meth:`acyclification` to obtain the ADMG whose m-separations are this graph's
    sigma-separations.
    """

    def __init__(
        self,
        directed_edges: Iterable[Edge],
        bidirected_edges: Iterable[Edge] | None = None,
        nodes: Iterable[str] | None = None,
    ) -> None:
        self._dg: nx.DiGraph[str] = nx.DiGraph()
        self._bi: nx.Graph[str] = nx.Graph()
        if nodes is not None:
            self._dg.add_nodes_from(nodes)
        self._dg.add_edges_from(directed_edges)
        for a, b in bidirected_edges or []:
            self._bi.add_edge(a, b)
        # keep the node sets aligned across the directed and bidirected views
        self._dg.add_nodes_from(self._bi.nodes)
        self._bi.add_nodes_from(self._dg.nodes)
        self._scc_of: dict[str, frozenset[str]] | None = None

    @property
    def nodes(self) -> list[str]:
        return list(self._dg.nodes)

    @property
    def directed_edges(self) -> list[Edge]:
        return [(u, v) for u, v in self._dg.edges]

    @property
    def bidirected_edges(self) -> list[Edge]:
        return [(u, v) for u, v in self._bi.edges]

    def parents(self, node: str) -> list[str]:
        self._check(node)
        return list(self._dg.predecessors(node))

    def children(self, node: str) -> list[str]:
        self._check(node)
        return list(self._dg.successors(node))

    def _check(self, node: str) -> None:
        if node not in self._dg:
            raise KeyError(f"unknown node: {node!r}")

    def is_acyclic(self) -> bool:
        """Whether the directed part is a DAG (no directed cycle)."""
        return bool(nx.is_directed_acyclic_graph(self._dg))

    def strongly_connected_components(self) -> list[set[str]]:
        """Strongly connected components of the directed part (isolated nodes are singletons)."""
        return [set(component) for component in nx.strongly_connected_components(self._dg)]

    def scc_of(self, node: str) -> frozenset[str]:
        """The strongly connected component containing ``node``."""
        self._check(node)
        if self._scc_of is None:
            mapping: dict[str, frozenset[str]] = {}
            for component in nx.strongly_connected_components(self._dg):
                frozen = frozenset(component)
                for member in component:
                    mapping[member] = frozen
            self._scc_of = mapping
        return self._scc_of[node]

    def acyclification(self) -> CausalGraph:
        """The Forré-Mooij acyclification: an acyclic ADMG whose m-separations are exactly this
        graph's sigma-separations.

        Construction (Forré & Mooij 2018): each strongly connected component ``S`` becomes a
        bidirected clique; every directed edge into ``S`` from outside is lifted to point at *every*
        member of ``S``; the within-``S`` directed edges are dropped. On an acyclic input every
        component is a singleton, so this is the identity ADMG and sigma-separation coincides with
        the shipped d-separation.
        """
        component_of: dict[str, frozenset[str]] = {}
        components = list(nx.strongly_connected_components(self._dg))
        for component in components:
            frozen = frozenset(component)
            for member in component:
                component_of[member] = frozen

        directed: list[Edge] = []
        seen: set[Edge] = set()
        for u, v in self._dg.edges:
            if component_of[u] == component_of[v]:
                continue  # within-SCC directed edge -> replaced by the bidirected clique below
            for w in component_of[v]:  # lift the external parent onto every SCC member
                edge = (u, w)
                if edge not in seen:
                    seen.add(edge)
                    directed.append(edge)

        bidirected: list[Edge] = [(u, v) for u, v in self._bi.edges]
        for component in components:
            members = sorted(component)
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    bidirected.append((members[i], members[j]))

        return CausalGraph(directed_edges=directed, bidirected_edges=bidirected, nodes=self.nodes)
