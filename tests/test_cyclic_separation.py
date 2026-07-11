"""sigma-separation (experimental cyclic support; plan §11: sigma/d coincidence on DAGs)."""

from __future__ import annotations

import numpy as np
import pytest

from causalrl.experimental.cyclic import CyclicCausalGraph, sigma_separated
from causalrl.identification._separation import d_separated
from causalrl.scm.graph import CausalGraph


def _random_dag_edges(
    rng: np.random.Generator, n: int, p: float
) -> tuple[list[str], list[tuple[str, str]]]:
    nodes = [f"v{i}" for i in range(n)]
    edges = [
        (nodes[i], nodes[j])
        for i in range(n)
        for j in range(i + 1, n)  # only forward edges -> guaranteed acyclic
        if rng.random() < p
    ]
    return nodes, edges


def _disjoint_query(
    rng: np.random.Generator, nodes: list[str]
) -> tuple[set[str], set[str], set[str]]:
    shuffled = list(nodes)
    rng.shuffle(shuffled)
    # x, y non-empty; z the (possibly empty) middle slice; all pairwise disjoint.
    a = 1 + int(rng.integers(0, max(1, len(shuffled) // 3)))
    b = a + 1 + int(rng.integers(0, max(1, len(shuffled) // 3)))
    return set(shuffled[:a]), set(shuffled[a:b]), set(shuffled[b:])


def test_coincides_with_d_separation_on_random_dags() -> None:
    """On acyclic graphs sigma-separation must equal the shipped d-separation (the acyclification
    is the identity), including with latent (bidirected) edges."""
    rng = np.random.default_rng(20260711)
    for _ in range(60):
        nodes, directed = _random_dag_edges(rng, n=6, p=0.35)
        # a few latent confounders among the observed nodes
        bidirected = [
            (nodes[i], nodes[j])
            for i in range(len(nodes))
            for j in range(i + 1, len(nodes))
            if rng.random() < 0.1
        ]
        cg = CausalGraph(directed_edges=directed, bidirected_edges=bidirected, nodes=nodes)
        ccg = CyclicCausalGraph(directed, bidirected, nodes=nodes)
        assert ccg.is_acyclic()
        for _ in range(6):
            x, y, z = _disjoint_query(rng, nodes)
            assert sigma_separated(ccg, x, y, z) == d_separated(cg, x, y, z)


def test_two_cycle_structure() -> None:
    ccg = CyclicCausalGraph([("A", "B"), ("B", "A"), ("B", "C")])
    assert not ccg.is_acyclic()
    sccs = {frozenset(s) for s in ccg.strongly_connected_components()}
    assert frozenset({"A", "B"}) in sccs
    assert frozenset({"C"}) in sccs
    assert ccg.scc_of("A") == frozenset({"A", "B"})


def test_two_cycle_sigma_separation_oracle() -> None:
    # A <-> B (2-cycle) -> C. Conditioning on the mediator B separates A from C; nothing else does.
    ccg = CyclicCausalGraph([("A", "B"), ("B", "A"), ("B", "C")])
    assert sigma_separated(ccg, {"A"}, {"C"}, {"B"}) is True
    assert sigma_separated(ccg, {"A"}, {"C"}, set()) is False
    assert sigma_separated(ccg, {"A"}, {"B"}, set()) is False  # same SCC: mutually dependent


def test_acyclification_rewrites_the_scc() -> None:
    ccg = CyclicCausalGraph([("A", "B"), ("B", "A"), ("B", "C")])
    acy = ccg.acyclification()
    directed = set(acy.directed_edges)
    bidirected = {frozenset(e) for e in acy.bidirected_edges}
    assert ("B", "C") in directed
    assert ("A", "B") not in directed and ("B", "A") not in directed  # within-SCC edges dropped
    assert frozenset({"A", "B"}) in bidirected  # SCC became a bidirected clique
    assert acy.nodes  # acyclification is itself a valid (acyclic) ADMG


def test_acyclification_of_a_dag_is_the_identity() -> None:
    directed = [("a", "b"), ("b", "c"), ("a", "c")]
    ccg = CyclicCausalGraph(directed)
    acy = ccg.acyclification()
    assert set(acy.directed_edges) == set(directed)
    assert acy.bidirected_edges == []


@pytest.mark.parametrize(
    ("x", "y", "z", "expected"),
    [
        ({"a"}, {"c"}, {"b"}, True),  # chain a->b->c blocked by b
        ({"a"}, {"c"}, set(), False),  # chain open
        ({"a"}, {"c"}, {"b"}, True),
    ],
)
def test_chain_matches_d_separation(x: set[str], y: set[str], z: set[str], expected: bool) -> None:
    ccg = CyclicCausalGraph([("a", "b"), ("b", "c")])
    assert sigma_separated(ccg, x, y, z) is expected
