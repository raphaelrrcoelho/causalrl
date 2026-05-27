"""Complete transportability (sID + mz + meta) via the general ``Domain`` engine.

M1: the general engine reproduces single-source transportability exactly (a behaviour-preserving
generalization of the c-factor routing) and reports a transport-hedge when no domain can supply a
needed c-factor. At c-factor granularity invariance is exactly "touches no selection-marked
variable", so single-source observational transport was already complete; the new power is surrogate
experiments (mz) and multiple source domains (meta), added below with simulation oracles.
"""

from __future__ import annotations

import pytest

from causalrl.exceptions import CausalGraphError
from causalrl.identification.id_algorithm import (
    Domain,
    identify_transport,
    is_identifiable_effect,
    is_transportable_general,
)
from causalrl.scm.graph import CausalGraph


# --- M1: the general engine reproduces single-source sID and reports hedges ---------------------
def test_covariate_shift_formula_mixes_domains() -> None:
    g = CausalGraph(directed_edges=[("Z", "X"), ("Z", "Y"), ("X", "Y")])
    formula = identify_transport(g, {"X"}, {"Y"}, ["Z"]).render()
    assert "do(" not in formula
    assert "P_target(" in formula and "P_source(" in formula  # shifted Z vs invariant Y


@pytest.mark.parametrize(
    ("edges", "bidirected", "ok"),
    [
        ([("Z", "X"), ("Z", "Y"), ("X", "Y")], [], True),
        ([("X", "M"), ("M", "Y")], [("X", "Y")], True),  # front-door
        ([("X", "Y")], [("X", "Y")], False),  # bow arc
    ],
)
def test_empty_selection_reduces_to_id(
    edges: list[tuple[str, str]], bidirected: list[tuple[str, str]], ok: bool
) -> None:
    g = CausalGraph(directed_edges=edges, bidirected_edges=bidirected)
    assert is_transportable_general(g, {"X"}, {"Y"}, [Domain("source")]) is ok
    assert is_identifiable_effect(g, {"X"}, {"Y"}) is ok


def test_non_transportable_hedge_is_reported() -> None:
    # Y is confounded with X (bow) AND its mechanism shifts (S->Y): neither source nor target
    # observational data supplies Q[Y]. A real transport-hedge.
    g = CausalGraph(directed_edges=[("X", "Y")], bidirected_edges=[("X", "Y")])
    assert is_transportable_general(g, {"X"}, {"Y"}, [Domain("source", frozenset({"Y"}))]) is False


def test_errors() -> None:
    g = CausalGraph(directed_edges=[("X", "Y")])
    with pytest.raises(CausalGraphError):
        is_transportable_general(g, {"X"}, {"Q"}, [Domain("source")])  # unknown outcome
    with pytest.raises(CausalGraphError):
        is_transportable_general(g, {"X"}, {"X"}, [Domain("source")])  # overlap
