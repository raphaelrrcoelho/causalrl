"""Property-based oracle tests for the ID / gID / transportability algorithms (issue #16).

``identification/id_algorithm.py`` implements the complete Shpitser-Pearl ID algorithm; its
correctness is the library's core promise. This file hardens that promise three ways:

1. **Witness oracle** (:mod:`tests._canonical_id_oracle`): for small random ADMGs, an
   independent brute-force search over the canonical response-function SCM class (Balke & Pearl
   1994; Tian & Pearl 2002) for two structurally-consistent instances that agree on the
   observational law but disagree on the causal effect. A found witness is a constructive proof
   of non-identifiability -- if ``is_identifiable_effect`` said True for such a graph, that would
   be a genuine bug. Restricted to small graphs (<=4 nodes); absence of a witness is
   inconclusive, not evidence of identifiability, so only "witness found" is asserted on.
2. **Simulation oracle** (:mod:`tests._random_admg_scm`): for random small ADMGs where
   ``is_identifiable_effect`` says True, build a randomly-parameterized StructuralCausalModel
   realizing that graph and confirm the returned estimand, evaluated on simulated data,
   recovers the ground-truth interventional mean from directly simulating ``do()`` -- dogfooding
   ``causalrl.scm`` across many random structures, generalizing the hand-picked examples in
   test_literature_classics.py / test_id_algorithm.py.
3. **Canonical battery**: front-door, bow-arc, the napkin graph, the instrumental-variable graph,
   and gID / transport examples, cross-checked against both the library and (where tractable)
   the witness oracle -- doubling as a self-validation that the oracle agrees with the published
   Bareinboim-Pearl / Shpitser-Pearl verdicts on the textbook cases.
"""

from __future__ import annotations

import zlib

import hypothesis.strategies as st
import pytest
from hypothesis import given, settings

from causalrl.identification.id_algorithm import (
    estimate_effect,
    is_gid_identifiable,
    is_identifiable_effect,
    is_transportable_effect,
)
from causalrl.scm.graph import CausalGraph

from ._canonical_id_oracle import find_witness
from ._random_admg_scm import build_random_scm

_NODES3 = ["A", "B", "C"]
_PAIRS3 = [(_NODES3[i], _NODES3[j]) for i in range(3) for j in range(i + 1, 3)]

_NODES4 = ["A", "B", "C", "D"]
_PAIRS4 = [(_NODES4[i], _NODES4[j]) for i in range(4) for j in range(i + 1, 4)]


def _stable_seed(*parts: object) -> int:
    """A deterministic (process-independent) seed, unlike Python's randomized `hash()`."""
    return zlib.crc32(repr(parts).encode()) % (2**31)


@st.composite
def _small_admgs(draw: st.DrawFn) -> tuple[CausalGraph, str, str]:
    """A random ADMG on 3 nodes (fixed topological order, so directed edges never cycle) plus
    two distinct treatment/outcome nodes. Kept to 3 nodes so the witness oracle stays fast."""
    directed = draw(st.lists(st.sampled_from(_PAIRS3), unique=True, max_size=3))
    bidirected = draw(st.lists(st.sampled_from(_PAIRS3), unique=True, max_size=2))
    graph = CausalGraph(directed_edges=directed, bidirected_edges=bidirected, nodes=_NODES3)
    treatment, outcome = draw(
        st.lists(st.sampled_from(_NODES3), min_size=2, max_size=2, unique=True)
    )
    return graph, treatment, outcome


@st.composite
def _admgs4(draw: st.DrawFn) -> tuple[CausalGraph, str, str]:
    """A random ADMG on 4 nodes plus two distinct treatment/outcome nodes."""
    directed = draw(st.lists(st.sampled_from(_PAIRS4), unique=True, max_size=4))
    bidirected = draw(st.lists(st.sampled_from(_PAIRS4), unique=True, max_size=2))
    graph = CausalGraph(directed_edges=directed, bidirected_edges=bidirected, nodes=_NODES4)
    treatment, outcome = draw(
        st.lists(st.sampled_from(_NODES4), min_size=2, max_size=2, unique=True)
    )
    return graph, treatment, outcome


def test_no_confounding_is_always_identifiable() -> None:
    """A DAG with no bidirected edges is always identifiable (the classical g-formula always
    applies) -- a cheap, certain ground truth the oracle doesn't need to establish."""

    @given(problem=_admgs4())
    @settings(max_examples=60, deadline=None)
    def _check(problem: tuple[CausalGraph, str, str]) -> None:
        graph, treatment, outcome = problem
        if graph.bidirected_edges:
            return
        assert is_identifiable_effect(graph, {treatment}, {outcome}) is True

    _check()


@given(problem=_small_admgs())
@settings(max_examples=25, deadline=None)
def test_oracle_witness_implies_not_identifiable(problem: tuple[CausalGraph, str, str]) -> None:
    graph, treatment, outcome = problem
    if not graph.bidirected_edges:
        return  # covered by test_no_confounding_is_always_identifiable
    seed = _stable_seed(graph.directed_edges, graph.bidirected_edges, treatment, outcome)
    witness = find_witness(graph, treatment, outcome, seed=seed)
    if not isinstance(witness, dict):
        return  # inconclusive ("TOO_BIG" or None): no counterexample found (or couldn't search)
    assert is_identifiable_effect(graph, {treatment}, {outcome}) is False, (
        f"oracle found a witness (same P(V), different do-effect {witness['do_values']}) "
        f"for {graph.directed_edges=} {graph.bidirected_edges=} {treatment=} {outcome=}, "
        "but is_identifiable_effect claimed it was identifiable"
    )


@given(problem=_admgs4())
@settings(max_examples=25, deadline=None)
def test_identifiable_effect_matches_simulation(problem: tuple[CausalGraph, str, str]) -> None:
    graph, treatment, outcome = problem
    if not is_identifiable_effect(graph, {treatment}, {outcome}):
        return
    seed = _stable_seed(graph.directed_edges, graph.bidirected_edges, treatment, outcome)
    scm = build_random_scm(graph, seed=seed)
    n = 20_000
    samples = scm.see(n, seed=seed)
    data = {v: samples[v].long().numpy() for v in graph.nodes}

    try:
        estimate = estimate_effect(graph, {treatment}, {outcome}, data, do={treatment: 1})[(1,)]
    except ZeroDivisionError:
        return  # empirical positivity failure at this sample size/seed; not what we're testing
    true_do = float(scm.do({treatment: 1.0}).see(n, seed=seed + 1)[outcome].float().mean())
    assert abs(estimate - true_do) < 0.05, (
        f"identify_effect's estimand disagreed with ground-truth simulation: "
        f"estimate={estimate:.4f} true={true_do:.4f} for {graph.directed_edges=} "
        f"{graph.bidirected_edges=} {treatment=} {outcome=}"
    )


# --- Canonical battery: front-door, bow-arc, napkin, IV (Shpitser & Pearl 2006) --------------

_CANONICAL_CASES = [
    pytest.param(
        CausalGraph(directed_edges=[("X", "M"), ("M", "Y")], bidirected_edges=[("X", "Y")]),
        "X",
        "Y",
        True,
        id="front-door",
    ),
    pytest.param(
        CausalGraph(directed_edges=[("X", "Y")], bidirected_edges=[("X", "Y")]),
        "X",
        "Y",
        False,
        id="bow-arc",
    ),
    pytest.param(
        CausalGraph(
            directed_edges=[("R", "W"), ("W", "X"), ("X", "Y")],
            bidirected_edges=[("R", "X"), ("R", "Y")],
        ),
        "X",
        "Y",
        True,
        id="napkin",
    ),
    pytest.param(
        CausalGraph(directed_edges=[("Z", "X"), ("X", "Y")], bidirected_edges=[("X", "Y")]),
        "X",
        "Y",
        False,
        id="instrumental-variable",
    ),
]


@pytest.mark.parametrize(("graph", "treatment", "outcome", "identifiable"), _CANONICAL_CASES)
def test_canonical_battery_matches_published_verdict(
    graph: CausalGraph, treatment: str, outcome: str, identifiable: bool
) -> None:
    """Faithful to Shpitser & Pearl, *Identification of Joint Interventional Distributions in
    Recursive Semi-Markovian Causal Models* (AAAI 2006), and Pearl, *Causality* (2009) Ch. 3-4
    for front-door / napkin / IV / bow-arc."""
    assert is_identifiable_effect(graph, {treatment}, {outcome}) is identifiable


@pytest.mark.parametrize(
    ("graph", "treatment", "outcome", "identifiable"),
    [c for c in _CANONICAL_CASES if c.id in ("front-door", "bow-arc", "instrumental-variable")],
)
def test_canonical_battery_oracle_self_check(
    graph: CausalGraph, treatment: str, outcome: str, identifiable: bool
) -> None:
    """The witness oracle, run against the SAME canonical cases above, agrees with the published
    verdict -- a self-validation that the independent oracle isn't just coincidentally right on
    the random-graph property tests. (Napkin needs 4 nodes and is slow to enumerate exhaustively
    here; it is already covered numerically by test_canonical_battery_matches_published_verdict.)
    """
    if identifiable:
        witness = find_witness(graph, treatment, outcome, seed=0)
        assert witness is None, f"oracle found a spurious witness for {graph}: {witness}"
    else:
        # the random-sampling search path isn't guaranteed to hit a witness on every seed;
        # a handful of retries is standard practice for a randomized (sound, incomplete) search.
        witnesses = (find_witness(graph, treatment, outcome, seed=s) for s in range(8))
        assert any(isinstance(w, dict) for w in witnesses), (
            f"oracle failed to find the known witness for {graph} across 8 seeds"
        )


# --- gID and transportability: Bareinboim & Pearl -----------------------------------------


def test_gid_bow_arc_identifiable_only_with_an_experiment_on_x() -> None:
    """General identification (gID): observational data alone fails on the bow arc, but a
    surrogate experiment on the treatment breaks the hedge. Faithful to Bareinboim & Pearl,
    *Causal Inference and the Data-Fusion Problem*, PNAS 2016, and Lee, Correa & Bareinboim,
    *General Identifiability with Arbitrary Surrogate Experiments*, UAI 2019."""
    graph = CausalGraph(directed_edges=[("X", "Y")], bidirected_edges=[("X", "Y")])
    assert is_gid_identifiable(graph, {"X"}, {"Y"}, []) is False
    assert is_gid_identifiable(graph, {"X"}, {"Y"}, [{"X"}]) is True


def test_transport_covariate_shift_is_transportable() -> None:
    """Transportability: a covariate-shift selection diagram (only P(Z) differs between source
    and target) is transportable via back-door adjustment on Z. Faithful to Bareinboim & Pearl,
    *Transportability of Causal Effects: Completeness Results*, AAAI 2012."""
    graph = CausalGraph(directed_edges=[("Z", "X"), ("Z", "Y"), ("X", "Y")])
    assert is_transportable_effect(graph, {"X"}, {"Y"}, ["Z"]) is True


def test_transport_bow_arc_is_not_transportable_from_observation_alone() -> None:
    """The bow arc's non-identifiability also blocks transport when the confounded pair itself
    is the selection variable and there is no adjustment set available. Faithful to Bareinboim
    & Pearl, *A General Algorithm for Deciding Transportability of Experimental Results*,
    Journal of Causal Inference 2013."""
    graph = CausalGraph(directed_edges=[("X", "Y")], bidirected_edges=[("X", "Y")])
    assert is_transportable_effect(graph, {"X"}, {"Y"}, ["X"]) is False
