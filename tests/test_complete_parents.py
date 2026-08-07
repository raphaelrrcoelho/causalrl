"""Asserting that a node's observed parents are ALL its parents, and what that licenses.

The gap this closes: :class:`~causalrl.scm.fitters.PinnedMechanism` asserts a node's *equation*,
but nothing asserted the *absence of unobserved parents* -- and that absence is the part that turns
a bound into a point estimate. No observational test can supply it; only a design can.
"""

from __future__ import annotations

import numpy as np
import pytest

from causalrl import CausalGraph, Kind, NotIdentifiableError, identify_effect
from causalrl.exceptions import CausalGraphError
from causalrl.scm.bounded_fit import fit_scm_bounded


def _bow_arc() -> CausalGraph:
    """A <-> Y with A -> Y: the canonical non-identified effect."""
    return CausalGraph(directed_edges=[("A", "Y")], bidirected_edges=[("A", "Y")], nodes=["A", "Y"])


def test_the_assertion_turns_a_refusal_into_an_identified_effect() -> None:
    """The whole point: BOUNDED -> IDENTIFIED, and only because a design licensed it."""
    graph = _bow_arc()
    with pytest.raises(NotIdentifiableError):
        identify_effect(graph, ["A"], ["Y"], return_certificate=False)

    randomised = graph.assert_complete_parents("A", reason="arm assigned by the trial schedule")
    cert = identify_effect(randomised, ["A"], ["Y"])

    assert cert.kind is Kind.IDENTIFIED


def test_the_certificate_records_the_assertion_it_rests_on() -> None:
    """An unfalsifiable assertion must be visible, or the claim looks graph-earned."""
    randomised = _bow_arc().assert_complete_parents("A", reason="assigned by a feature flag")
    cert = identify_effect(randomised, ["A"], ["Y"])

    assumption = next(a for a in cert.assumptions if a.name == "parents-complete-by-design")
    assert assumption.checkable is False
    assert assumption.params == {"A": "assigned by a feature flag"}


def test_a_graph_earned_identification_carries_no_such_assumption() -> None:
    """The assumption appears only when it was actually used."""
    plain = CausalGraph(directed_edges=[("A", "Y")], nodes=["A", "Y"])
    cert = identify_effect(plain, ["A"], ["Y"])
    assert all(a.name != "parents-complete-by-design" for a in cert.assumptions)


def test_the_assertion_contradicts_a_bidirected_edge_at_that_node() -> None:
    """Declaring completeness while asserting a latent common cause is incoherent, not additive."""
    with pytest.raises(CausalGraphError, match="cannot both hold"):
        CausalGraph(
            directed_edges=[("A", "Y")],
            bidirected_edges=[("A", "Y")],
            nodes=["A", "Y"],
            complete_parents={"A": "randomised"},
        )


def test_assert_complete_parents_only_drops_edges_at_the_named_nodes() -> None:
    graph = CausalGraph(
        directed_edges=[("A", "Y"), ("B", "Y")],
        bidirected_edges=[("A", "Y"), ("B", "Y")],
        nodes=["A", "B", "Y"],
    )
    partial = graph.assert_complete_parents("A", reason="randomised")
    assert partial.bidirected_edges == [("B", "Y")]
    assert partial.has_complete_parents("A") and not partial.has_complete_parents("B")


def test_a_reason_is_mandatory_and_must_say_something() -> None:
    graph = _bow_arc()
    with pytest.raises(ValueError, match="reason must be non-empty"):
        graph.assert_complete_parents("A", reason="   ")
    with pytest.raises(ValueError, match="at least one node"):
        graph.assert_complete_parents(reason="randomised")


def test_the_assertion_makes_a_bounded_fit_identified() -> None:
    """It reaches the fitting layer too: no confounded parent left means no interval."""
    rng = np.random.default_rng(0)
    n = 600
    a = rng.integers(0, 2, size=n).astype(np.float64)
    data = {"A": a, "Y": 0.3 * a + rng.normal(scale=0.1, size=n)}

    confounded = fit_scm_bounded(data, graph=_bow_arc(), value_ranges={"Y": (-1.0, 2.0)})
    assert confounded.bounded_nodes == ("Y",)

    randomised = fit_scm_bounded(
        data, graph=_bow_arc().assert_complete_parents("A", reason="randomised assignment")
    )
    assert randomised.bounded_nodes == ()
    assert randomised.certificate().kind is Kind.IDENTIFIED
