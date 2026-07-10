"""Phase 0: Hypothesis property tests for the new primitives (§12.1 property-based tier).

Covers ``graph_hash`` canonicality over random small DAGs and ``Certificate`` JSON round-tripping
over randomized certificates. Both targets are torch-free.
"""

from hypothesis import given
from hypothesis import strategies as st

from causalrl.certify import Assumption, Certificate, EstimandSpec, Hedge, Kind, Provenance, Witness
from causalrl.graphs import graph_hash
from causalrl.identification.bounds import Interval
from causalrl.scm.graph import CausalGraph

_NODES = ["A", "B", "C", "D"]
_DagSpec = tuple[list[str], list[tuple[str, str]], list[tuple[str, str]]]


@st.composite
def _dag_with_permuted_edges(draw: st.DrawFn) -> _DagSpec:
    """A random small DAG (forward edges only ⇒ acyclic) plus a permutation of its edge list."""
    k = draw(st.integers(min_value=1, max_value=len(_NODES)))
    nodes = _NODES[:k]
    possible = [(nodes[i], nodes[j]) for i in range(k) for j in range(i + 1, k)]
    edges = (
        draw(st.lists(st.sampled_from(possible), unique=True, max_size=len(possible)))
        if possible
        else []
    )
    permuted = draw(st.permutations(edges))
    return nodes, edges, list(permuted)


@given(_dag_with_permuted_edges())
def test_graph_hash_invariant_under_edge_order(
    spec: _DagSpec,
) -> None:
    nodes, edges, permuted = spec
    assert graph_hash(CausalGraph(edges, nodes=nodes)) == graph_hash(
        CausalGraph(permuted, nodes=nodes)
    )


@given(_dag_with_permuted_edges())
def test_graph_hash_sensitive_to_isolated_node(
    spec: _DagSpec,
) -> None:
    nodes, edges, _ = spec
    base = graph_hash(CausalGraph(edges, nodes=nodes))
    with_extra = graph_hash(CausalGraph(edges, nodes=[*nodes, "ZZZ"]))
    assert base != with_extra


_SAFE_TEXT = st.text(alphabet="abcdef ABCDEF 0123", max_size=20)
_FINITE = st.floats(allow_nan=False, allow_infinity=False, width=32)


@given(
    claim=_SAFE_TEXT,
    kind=st.sampled_from(list(Kind)),
    value=st.one_of(st.none(), _FINITE, st.builds(Interval, _FINITE, _FINITE)),
    gamma=st.floats(min_value=1.0, max_value=10.0),
    hedged=st.booleans(),
)
def test_certificate_json_roundtrip(
    claim: str, kind: Kind, value: float | Interval | None, gamma: float, hedged: bool
) -> None:
    cert = Certificate(
        claim=claim,
        estimand=EstimandSpec(query="do", target="mean", domains=("A", "B")),
        kind=kind,
        value=value,
        alpha=0.05,
        assumptions=(Assumption("MSM", {"gamma": gamma}, checkable=True),),
        method="prop-test",
        witness=Witness("adjustment-set", {"set": ["Z"]}),
        hedge=Hedge("downgrade", {"why": "test"}, downgraded_from="mean") if hedged else None,
        provenance=Provenance(
            library_version="1.3.0",
            seeds=(0, 1),
            graph_hash="deadbeef",
            timestamp="2026-07-10T00:00:00+00:00",
        ),
    )
    assert Certificate.from_json(cert.to_json()) == cert
