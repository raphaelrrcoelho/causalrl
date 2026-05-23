import networkx as nx

from causalrl.exceptions import CausalGraphError


class CausalGraph:
    """A causal graph: a DAG over observed variables plus bidirected edges that
    denote unobserved confounders (an ADMG)."""

    def __init__(
        self,
        directed_edges: list[tuple[str, str]],
        bidirected_edges: list[tuple[str, str]] | None = None,
        nodes: list[str] | None = None,
    ) -> None:
        self._dag: nx.DiGraph[str] = nx.DiGraph()
        self._bi: nx.Graph[str] = nx.Graph()
        if nodes:
            self._dag.add_nodes_from(nodes)
            self._bi.add_nodes_from(nodes)
        self._dag.add_edges_from(directed_edges)
        for a, b in bidirected_edges or []:
            self._bi.add_node(a)
            self._bi.add_node(b)
            self._bi.add_edge(a, b)
        # keep node sets aligned
        self._bi.add_nodes_from(self._dag.nodes)
        self._dag.add_nodes_from(self._bi.nodes)
        if not nx.is_directed_acyclic_graph(self._dag):
            raise CausalGraphError("directed edges must form a DAG (cycle detected)")

    @property
    def nodes(self) -> list[str]:
        return list(self._dag.nodes)

    def _check(self, node: str) -> None:
        if node not in self._dag:
            raise CausalGraphError(f"unknown node: {node!r}")

    def parents(self, node: str) -> list[str]:
        self._check(node)
        return list(self._dag.predecessors(node))

    def children(self, node: str) -> list[str]:
        self._check(node)
        return list(self._dag.successors(node))

    def topological_order(self) -> list[str]:
        return list(nx.topological_sort(self._dag))

    def is_confounded(self, a: str, b: str) -> bool:
        self._check(a)
        self._check(b)
        return self._bi.has_edge(a, b)

    def c_components(self) -> list[set[str]]:
        """Connected components of the bidirected graph (isolated nodes are singletons)."""
        return [set(c) for c in nx.connected_components(self._bi)]

    def remove_incoming_edges(self, node: str) -> CausalGraph:
        """Return a copy with all directed edges into `node` removed (graph mutilation)."""
        self._check(node)
        directed = [(u, v) for u, v in self._dag.edges if v != node]
        bidirected = list(self._bi.edges)
        return CausalGraph(directed, bidirected, nodes=self.nodes)
