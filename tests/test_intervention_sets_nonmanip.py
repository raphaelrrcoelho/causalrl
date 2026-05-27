import hypothesis.strategies as st
from hypothesis import given, settings

from causalrl.identification.intervention_sets import minimal_intervention_sets, pomis
from causalrl.scm.graph import CausalGraph


def fs(*names: str) -> frozenset[str]:
    return frozenset(names)


def _fig2a() -> CausalGraph:
    # Lee & Bareinboim 2019 (R-40), Fig. 2a.
    return CausalGraph(
        directed_edges=[("A", "C"), ("B", "C"), ("A", "Y"), ("C", "Y")],
        bidirected_edges=[("A", "B"), ("B", "Y")],
    )


def test_fig2a_unconstrained_matches_paper():
    assert set(pomis(_fig2a(), "Y")) == {fs(), fs("A"), fs("A", "C")}


def test_fig2a_non_manipulable_A():
    # N={A}, manipulable={B,C}.  P^A = {empty,{B},{C}}; filtering unconstrained gives only {empty}.
    assert set(pomis(_fig2a(), "Y", manipulable={"B", "C"})) == {fs(), fs("B"), fs("C")}


def test_fig2a_non_manipulable_B():
    # N={B}, manipulable={A,C}.  P^B = {empty,{A},{A,C}}.
    assert set(pomis(_fig2a(), "Y", manipulable={"A", "C"})) == {fs(), fs("A"), fs("A", "C")}


def test_fig2a_non_manipulable_C():
    # N={C}, manipulable={A,B}.  P^C = {empty,{A},{A,B}}; {A,B} is NOT an unconstrained POMIS.
    assert set(pomis(_fig2a(), "Y", manipulable={"A", "B"})) == {fs(), fs("A"), fs("A", "B")}


def test_frontdoor_non_manipulable_Z():
    # X->Z->Y, X<->Y ; Z non-manipulable.  P^Z = {empty,{X}} (front-door projects to a bow arc).
    g = CausalGraph(directed_edges=[("X", "Z"), ("Z", "Y")], bidirected_edges=[("X", "Y")])
    assert set(pomis(g, "Y", manipulable={"X"})) == {fs(), fs("X")}


def test_mis_filters_out_non_manipulable():
    g = _fig2a()
    constrained = minimal_intervention_sets(g, "Y", manipulable={"B", "C"})
    assert all("A" not in s for s in constrained)
    assert set(constrained) <= set(minimal_intervention_sets(g, "Y"))


def test_manipulable_none_equals_all_nonreward():
    g = _fig2a()
    assert pomis(g, "Y") == pomis(g, "Y", manipulable={"A", "B", "C"})
    assert minimal_intervention_sets(g, "Y") == minimal_intervention_sets(
        g, "Y", manipulable={"A", "B", "C"}
    )


_NODES = ["A", "B", "C", "D"]
_PAIRS = [(_NODES[i], _NODES[j]) for i in range(len(_NODES)) for j in range(i + 1, len(_NODES))]


@st.composite
def admg_with_manipulable(draw: st.DrawFn) -> tuple[CausalGraph, str, set[str]]:
    directed = draw(st.lists(st.sampled_from(_PAIRS), unique=True))
    bidirected = draw(st.lists(st.sampled_from(_PAIRS), unique=True))
    graph = CausalGraph(directed_edges=directed, bidirected_edges=bidirected, nodes=_NODES)
    reward = draw(st.sampled_from(_NODES))
    non_reward = [n for n in _NODES if n != reward]
    manipulable = set(draw(st.lists(st.sampled_from(non_reward), unique=True)))
    return graph, reward, manipulable


@settings(max_examples=200, deadline=None)
@given(admg_with_manipulable())
def test_constrained_pomis_subset_of_manipulable(problem: tuple[CausalGraph, str, set[str]]):
    graph, reward, manipulable = problem
    for s in pomis(graph, reward, manipulable=manipulable):
        assert s <= manipulable


@settings(max_examples=200, deadline=None)
@given(admg_with_manipulable())
def test_prop1_unconstrained_pomis_disjoint_from_n_are_constrained(
    problem: tuple[CausalGraph, str, set[str]],
):
    # r40 Proposition 1: an unconstrained POMIS disjoint from N stays a (constrained) POMIS.
    graph, reward, manipulable = problem
    constrained = set(pomis(graph, reward, manipulable=manipulable))
    for s in pomis(graph, reward):
        if s <= manipulable:
            assert s in constrained
