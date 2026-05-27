"""Transportability (sID, Task 4): identify a target effect across a selection diagram.

Regression: with no selection nodes, transport reduces to the ID algorithm. The numeric test uses a
source and a target SCM that share every mechanism except the marked one, and checks the transported
estimate matches the target's true do() distribution (and differs from naively reusing the source).
"""

from __future__ import annotations

import numpy as np
import pytest
from torch import Tensor
from torch.distributions import Bernoulli, Distribution, Uniform

from causalrl.identification.id_algorithm import (
    estimate_transported_effect,
    identify_transport,
    is_identifiable_effect,
    is_transportable_effect,
)
from causalrl.scm.graph import CausalGraph
from causalrl.scm.mechanisms import FunctionalMechanism, Mechanism
from causalrl.scm.scm import StructuralCausalModel

_N = 40_000


def _flip(u: Tensor, p: float) -> Tensor:
    return (u < p).float()


def _cols(samples: dict[str, Tensor], keep: list[str]) -> dict[str, np.ndarray]:
    return {name: samples[name].long().numpy() for name in keep}


@pytest.mark.parametrize(
    ("edges", "bidirected", "identifiable"),
    [
        ([("Z", "X"), ("Z", "Y"), ("X", "Y")], [], True),
        ([("X", "M"), ("M", "Y")], [("X", "Y")], True),  # front-door
        ([("X", "Y")], [("X", "Y")], False),  # bow arc
    ],
)
def test_no_selection_reduces_to_identification(
    edges: list[tuple[str, str]], bidirected: list[tuple[str, str]], identifiable: bool
) -> None:
    graph = CausalGraph(directed_edges=edges, bidirected_edges=bidirected)
    assert is_transportable_effect(graph, {"X"}, {"Y"}, []) is identifiable
    assert is_identifiable_effect(graph, {"X"}, {"Y"}) is identifiable


def test_not_transportable_when_a_shifted_factor_is_unidentifiable() -> None:
    # Y is confounded with X (bow) and its mechanism also differs across domains (S -> Y): the Y
    # c-factor is neither transferable from the source nor identifiable from the target.
    graph = CausalGraph(directed_edges=[("X", "Y")], bidirected_edges=[("X", "Y")])
    assert is_transportable_effect(graph, {"X"}, {"Y"}, ["Y"]) is False


def test_covariate_shift_formula_is_do_free_and_mixes_domains() -> None:
    graph = CausalGraph(directed_edges=[("Z", "X"), ("Z", "Y"), ("X", "Y")])
    formula = identify_transport(graph, {"X"}, {"Y"}, ["Z"]).render()
    assert "do(" not in formula
    assert "P_target(" in formula and "P_source(" in formula  # the shifted Z vs invariant Y


def _shift_scm(p_z: float) -> StructuralCausalModel:
    """``Z -> X``, ``Z -> Y``, ``X -> Y``; only ``Z``'s prior (``p_z``) changes between domains."""
    graph = CausalGraph(directed_edges=[("Z", "X"), ("Z", "Y"), ("X", "Y")])
    mechanisms: dict[str, Mechanism] = {
        "Z": FunctionalMechanism([], lambda pa, u: u),
        "X": FunctionalMechanism(["Z"], lambda pa, u: (pa["Z"] + _flip(u, 0.2)) % 2),
        "Y": FunctionalMechanism(
            ["X", "Z"], lambda pa, u: ((((pa["X"] + pa["Z"]) > 0).float()) + _flip(u, 0.05)) % 2
        ),
    }
    exogenous: dict[str, Distribution] = {
        "Z": Bernoulli(p_z),
        "X": Uniform(0.0, 1.0),
        "Y": Uniform(0.0, 1.0),
    }
    return StructuralCausalModel(graph, mechanisms, exogenous)


def test_transported_estimate_matches_the_target_under_covariate_shift() -> None:
    source, target = _shift_scm(0.5), _shift_scm(0.85)  # Z's prior is the only difference (S -> Z)
    graph = CausalGraph(directed_edges=[("Z", "X"), ("Z", "Y"), ("X", "Y")])
    source_data = _cols(source.see(_N, seed=0), ["Z", "X", "Y"])
    target_data = _cols(target.see(_N, seed=1), ["Z", "X", "Y"])

    for value in (0, 1):
        estimate = estimate_transported_effect(
            graph, {"X"}, {"Y"}, ["Z"], source_data, target_data, do={"X": value}
        )[(1,)]
        target_truth = float(target.do({"X": float(value)}).see(_N, seed=7)["Y"].float().mean())
        assert estimate == pytest.approx(target_truth, abs=0.03)

    # The shift matters: reusing the source effect (source Z prior) is meaningfully wrong at X=0.
    transported = estimate_transported_effect(
        graph, {"X"}, {"Y"}, ["Z"], source_data, target_data, do={"X": 0}
    )[(1,)]
    source_truth = float(source.do({"X": 0.0}).see(_N, seed=7)["Y"].float().mean())
    assert abs(transported - source_truth) > 0.1
