from __future__ import annotations

from collections.abc import Iterable, Mapping

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
        complete_parents: Mapping[str, str] | None = None,
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
        self._complete_parents: dict[str, str] = dict(complete_parents or {})
        for node in self._complete_parents:
            if node not in self._dag:
                raise CausalGraphError(f"unknown node in complete_parents: {node!r}")
            if self._bi.degree(node) > 0:
                partners = sorted(self._bi.neighbors(node))
                raise CausalGraphError(
                    f"{node!r} is declared to have complete parents but shares a bidirected edge "
                    f"with {partners}: a bidirected edge asserts an unobserved common cause, which "
                    f"IS an unobserved parent of {node!r}. The two statements cannot both hold. "
                    "Use CausalGraph.assert_complete_parents to drop those edges deliberately, "
                    "which records why."
                )

    @property
    def nodes(self) -> list[str]:
        return list(self._dag.nodes)

    @property
    def directed_edges(self) -> list[tuple[str, str]]:
        """The directed edges as ``(parent, child)`` pairs."""
        return [(u, v) for u, v in self._dag.edges]

    @property
    def bidirected_edges(self) -> list[tuple[str, str]]:
        """The bidirected (latent-confounding) edges."""
        return [(u, v) for u, v in self._bi.edges]

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

    def has_bidirected_edges(self) -> bool:
        """Whether this graph contains latent-confounding (bidirected) edges."""
        return self._bi.number_of_edges() > 0

    @property
    def complete_parents(self) -> dict[str, str]:
        """Nodes asserted to have NO unobserved parent, mapped to why.

        See :meth:`has_complete_parents`.
        """
        return dict(self._complete_parents)

    def has_complete_parents(self, node: str) -> bool:
        """Whether ``node``'s observed parents are asserted to be all the parents there are.

        This is the assertion that converts a bound into a point estimate, and it is the one thing
        a graph cannot learn from data: no observational test distinguishes "no unobserved parent"
        from "an unobserved parent I did not measure". It is licensed by *design* -- a randomised
        assignment, a feature flag, an experimenter's schedule, a rule-based pricing policy -- where
        the mechanism that sets the node is written down and contains nothing else.

        The library already had :class:`~causalrl.scm.fitters.PinnedMechanism`, which asserts the
        node's *equation*. That is a different and weaker claim: an equation can be supplied while
        the world still feeds the node something you did not model. This asserts the absence.
        """
        self._check(node)
        return node in self._complete_parents

    def assert_complete_parents(self, *nodes: str, reason: str) -> CausalGraph:
        """A copy of this graph in which ``nodes`` have no unobserved parents, and why.

        Bidirected edges incident to ``nodes`` are DROPPED, because they contradict the assertion:
        a latent common cause of ``A`` and ``Y`` is an unobserved parent of ``A``. This is what
        upgrades an effect from bounded to identified, so it is deliberately a distinct call with a
        mandatory ``reason`` rather than a constructor flag -- the assertion is a trust primitive,
        and an incorrect one silently converts an honest interval into a false point estimate.

        ``reason`` travels with the graph into every certificate built from it, so a reader can see
        which claims rest on it.
        """
        if not nodes:
            raise ValueError("assert_complete_parents needs at least one node")
        if not reason.strip():
            raise ValueError(
                "reason must be non-empty: this assertion is not checkable from data, so the "
                "record of why it is licensed is the only thing standing behind every claim it "
                "strengthens."
            )
        for node in nodes:
            self._check(node)
        targets = set(nodes)
        kept = [(a, b) for a, b in self.bidirected_edges if a not in targets and b not in targets]
        merged = dict(self._complete_parents)
        merged.update({node: reason for node in nodes})
        return CausalGraph(
            directed_edges=self.directed_edges,
            bidirected_edges=kept,
            nodes=self.nodes,
            complete_parents=merged,
        )

    def has_incident_bidirected_edges(self, node: str) -> bool:
        """Whether `node` is incident to any latent-confounding edge."""
        self._check(node)
        return self._bi.degree(node) > 0

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
