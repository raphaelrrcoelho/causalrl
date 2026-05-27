from collections.abc import Iterable

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

    def _as_node_set(self, nodes: str | Iterable[str]) -> set[str]:
        ns = {nodes} if isinstance(nodes, str) else set(nodes)
        for n in ns:
            self._check(n)
        return ns

    def ancestors(self, nodes: str | Iterable[str]) -> set[str]:
        """Ancestors of `nodes` via directed edges, INCLUDING the inputs (the inclusive An(·)
        convention used by the identification/POMIS literature)."""
        ns = self._as_node_set(nodes)
        result = set(ns)
        for n in ns:
            result |= set(nx.ancestors(self._dag, n))  # type: ignore[reportUnknownMemberType]
        return result

    def descendants(self, nodes: str | Iterable[str]) -> set[str]:
        """Strict descendants of `nodes` via directed edges (excludes the inputs)."""
        ns = self._as_node_set(nodes)
        result: set[str] = set()
        for n in ns:
            result |= set(nx.descendants(self._dag, n))  # type: ignore[reportUnknownMemberType]
        return result

    def induced_subgraph(self, nodes: str | Iterable[str]) -> CausalGraph:
        """Subgraph on `nodes`: keep directed/bidirected edges with both endpoints in `nodes`."""
        keep = self._as_node_set(nodes)
        directed = [(u, v) for u, v in self._dag.edges if u in keep and v in keep]
        bidirected = [(u, v) for u, v in self._bi.edges if u in keep and v in keep]
        return CausalGraph(directed, bidirected, nodes=list(keep))

    def do_mutilate(self, intervened: str | Iterable[str]) -> CausalGraph:
        """ADMG mutilation for do(intervened): drop incoming directed edges to each
        intervened node AND every bidirected edge incident to an intervened node
        (intervention severs latent confounding into the set). Distinct from
        ``remove_incoming_edges``, which keeps bidirected edges."""
        x = self._as_node_set(intervened)
        directed = [(u, v) for u, v in self._dag.edges if v not in x]
        bidirected = [(u, v) for u, v in self._bi.edges if u not in x and v not in x]
        return CausalGraph(directed, bidirected, nodes=self.nodes)

    def _kept_reach(self, children: Iterable[str], keep: set[str]) -> set[str]:
        """Kept nodes reachable from `children` via directed edges whose interior nodes are
        all outside `keep` (a kept node is a boundary: it is reached but not expanded)."""
        reached: set[str] = set()
        seen: set[str] = set()
        frontier = list(children)
        while frontier:
            w = frontier.pop()
            if w in keep:
                reached.add(w)
            elif w not in seen:
                seen.add(w)
                frontier.extend(list(self._dag.successors(w)))
        return reached

    def latent_projection(self, keep: str | Iterable[str]) -> CausalGraph:
        """Latent projection onto `keep`: marginalize out every node not in `keep`, adding a
        directed edge for each directed path through removed nodes and a bidirected edge for
        each confounding path through removed nodes (the Tian-Pearl / Verma projection).

        Directed ``Vi -> Vj`` when a directed path from ``Vi`` to ``Vj`` has all interior
        nodes removed; bidirected ``Vi <-> Vj`` when a removed common cause (a marginalized
        node, or the latent behind a bidirected edge) reaches both through removed interiors.
        Removing a collider induces no confounding (its parents are never in its reached set)."""
        keep_set = self._as_node_set(keep)
        directed: list[tuple[str, str]] = []
        for vi in keep_set:
            for w in self._kept_reach(list(self._dag.successors(vi)), keep_set):
                directed.append((vi, w))

        bidirected: set[tuple[str, str]] = set()
        sources = [list(self._dag.successors(w)) for w in self._dag.nodes if w not in keep_set]
        sources += [[a, b] for a, b in self._bi.edges]
        for children in sources:
            reached = sorted(self._kept_reach(children, keep_set))
            for i in range(len(reached)):
                for j in range(i + 1, len(reached)):
                    bidirected.add((reached[i], reached[j]))

        return CausalGraph(directed, list(bidirected), nodes=list(keep_set))
