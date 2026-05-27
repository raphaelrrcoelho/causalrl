"""S-node canonical-DAG construction and m-separation, shared by the transportability code.

Bidirected ``A<->B`` projects to ``A<-L->B`` and each selection variable ``v`` gains a node
``S->v``, reducing ADMG m-separation (with selection) to DAG d-separation over the observed and
selection nodes. Lifted out of :mod:`causalrl.identification.transport` so
:mod:`causalrl.identification.id_algorithm` can use it without a circular import.
"""

from __future__ import annotations

from collections.abc import Iterable

import networkx as nx

from causalrl.scm.graph import CausalGraph

SEL = "__S__::"  # selection-node prefix
LAT = "__L__::"  # latent (bidirected) prefix


def canonical_dag(graph: CausalGraph, selection: Iterable[str]) -> nx.DiGraph[str]:
    """The latent-projection canonical DAG: ``A<->B`` becomes ``A<-L->B`` and each selection
    variable ``v`` gains a node ``S->v``."""
    dag: nx.DiGraph[str] = nx.DiGraph()
    dag.add_nodes_from(graph.nodes)
    dag.add_edges_from(graph.directed_edges)
    for i, (a, b) in enumerate(graph.bidirected_edges):
        latent = f"{LAT}{i}"
        dag.add_edge(latent, a)
        dag.add_edge(latent, b)
    for v in selection:
        dag.add_edge(f"{SEL}{v}", v)
    return dag


def selection_nodes(selection: Iterable[str]) -> set[str]:
    return {f"{SEL}{v}" for v in selection}


def d_separated(
    graph: CausalGraph,
    x: set[str],
    y: set[str],
    z: set[str],
    selection: Iterable[str] = (),
) -> bool:
    """m-separation in ``graph`` (ADMG) with optional selection nodes: is ``x`` independent of
    ``y`` given ``z``?"""
    dag = canonical_dag(graph, selection)
    return bool(nx.is_d_separator(dag, x, y, z))  # type: ignore[reportUnknownMemberType]
