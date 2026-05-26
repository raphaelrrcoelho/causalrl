import hypothesis.strategies as st
from hypothesis import given, settings

from causalrl.identification.intervention_sets import minimal_intervention_sets, pomis
from causalrl.scm.graph import CausalGraph

_NODES = ["A", "B", "C", "D"]
# Only "forward" pairs in this fixed order, so directed edges can never form a cycle.
_PAIRS = [(_NODES[i], _NODES[j]) for i in range(len(_NODES)) for j in range(i + 1, len(_NODES))]


@st.composite
def admgs(draw: st.DrawFn) -> tuple[CausalGraph, str]:
    directed = draw(st.lists(st.sampled_from(_PAIRS), unique=True))
    bidirected = draw(st.lists(st.sampled_from(_PAIRS), unique=True))
    graph = CausalGraph(directed_edges=directed, bidirected_edges=bidirected, nodes=_NODES)
    reward = draw(st.sampled_from(_NODES))
    return graph, reward


@settings(max_examples=200, deadline=None)
@given(admgs())
def test_pomis_invariants(problem: tuple[CausalGraph, str]):
    graph, reward = problem
    p = pomis(graph, reward)
    m = minimal_intervention_sets(graph, reward)

    assert len(p) == len(set(p)), "POMIS list must be deduplicated"
    assert len(p) >= 1, "there is always at least one POMIS"
    assert set(p) <= set(m), "every POMIS is a MIS"

    allowed = graph.ancestors(reward) - {reward}
    for s in p:
        assert s <= allowed, "a POMIS can only target strict ancestors of the reward"


@settings(max_examples=200, deadline=None)
@given(admgs())
def test_no_confounding_means_unique_pomis_is_parents(problem: tuple[CausalGraph, str]):
    graph, reward = problem
    anc = graph.induced_subgraph(graph.ancestors(reward))
    confounded = any(len(c) > 1 for c in anc.c_components())
    if not confounded:
        assert pomis(graph, reward) == [frozenset(graph.parents(reward))]
