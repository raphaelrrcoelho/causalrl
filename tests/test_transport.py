"""Transportability: m-separation oracles and the transport-formula decision."""

from __future__ import annotations

import pytest

from causalrl.exceptions import CausalGraphError
from causalrl.identification._separation import d_separated as d_sep
from causalrl.identification.transport import (
    SelectionDiagram,
    is_transportable,
    transport_formula,
)
from causalrl.scm.graph import CausalGraph


def test_m_separation_chain() -> None:
    g = CausalGraph(directed_edges=[("A", "B"), ("B", "C")])
    assert d_sep(g, {"A"}, {"C"}, {"B"})  # blocked by the middle of the chain
    assert not d_sep(g, {"A"}, {"C"}, set())  # open chain


def test_m_separation_fork() -> None:
    g = CausalGraph(directed_edges=[("B", "A"), ("B", "C")])
    assert d_sep(g, {"A"}, {"C"}, {"B"})  # blocked by the common cause
    assert not d_sep(g, {"A"}, {"C"}, set())


def test_m_separation_collider() -> None:
    g = CausalGraph(directed_edges=[("A", "B"), ("C", "B")])
    assert d_sep(g, {"A"}, {"C"}, set())  # collider closed when unconditioned
    assert not d_sep(g, {"A"}, {"C"}, {"B"})  # conditioning the collider opens the path


def test_m_separation_bidirected() -> None:
    connected = CausalGraph(directed_edges=[("A", "B")], bidirected_edges=[("A", "C")])
    assert not d_sep(connected, {"A"}, {"C"}, set())  # the latent links A and C
    collider = CausalGraph(directed_edges=[("A", "B")], bidirected_edges=[("B", "C")])
    assert d_sep(collider, {"A"}, {"C"}, set())  # A -> B <- L -> C is a closed collider at B


def test_adjustment_transport_with_shifted_covariate() -> None:
    g = CausalGraph(directed_edges=[("Z", "X"), ("Z", "Y"), ("X", "Y")])
    diagram = SelectionDiagram(g, frozenset({"Z"}))
    formula = transport_formula(diagram, treatment="X", outcome="Y")
    assert formula is not None
    assert formula.kind == "adjustment"
    assert formula.adjustment_set == frozenset({"Z"})
    assert is_transportable(diagram, treatment="X", outcome="Y")


def test_direct_transport_when_selection_off_interventional_path() -> None:
    # W -> X -> Y, selection on W: do(X) severs W -> X, so W cannot reach Y.
    g = CausalGraph(directed_edges=[("W", "X"), ("X", "Y")])
    diagram = SelectionDiagram(g, frozenset({"W"}))
    formula = transport_formula(diagram, treatment="X", outcome="Y")
    assert formula is not None
    assert formula.kind == "direct"


def test_no_selection_is_directly_transportable() -> None:
    g = CausalGraph(directed_edges=[("Z", "X"), ("Z", "Y"), ("X", "Y")])
    diagram = SelectionDiagram(g, frozenset())
    formula = transport_formula(diagram, treatment="X", outcome="Y")
    assert formula is not None
    assert formula.kind == "direct"


def test_not_transportable_when_selection_on_outcome() -> None:
    # X -> Y with the target differing in Y's own mechanism: no admissible adjustment exists.
    g = CausalGraph(directed_edges=[("X", "Y")])
    diagram = SelectionDiagram(g, frozenset({"Y"}))
    assert transport_formula(diagram, treatment="X", outcome="Y") is None
    assert not is_transportable(diagram, treatment="X", outcome="Y")


def test_unknown_nodes_raise() -> None:
    diagram = SelectionDiagram(CausalGraph(directed_edges=[("X", "Y")]), frozenset())
    with pytest.raises(CausalGraphError):
        transport_formula(diagram, treatment="X", outcome="Q")
    with pytest.raises(CausalGraphError):
        SelectionDiagram(CausalGraph(directed_edges=[("X", "Y")]), frozenset({"Z"}))
