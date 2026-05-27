"""General identification (gID, Task 4): identify P(y|do(x)) from data plus surrogate experiments.

Regression: with no experiments, gID must coincide exactly with the ID algorithm. Structural tests
check that the right experiment (and only the right one) breaks a hedge. Numeric tests confirm the
estimand evaluated on observational + experimental data matches the true do() distribution.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import Tensor
from torch.distributions import Bernoulli, Distribution, Uniform

from causalrl.identification.id_algorithm import (
    estimate_effect_with_experiments,
    identify_effect_with_experiments,
    is_gid_identifiable,
    is_identifiable_effect,
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
        ([("Z", "X"), ("Z", "Y"), ("X", "Y")], [], True),  # back-door
        ([("X", "M"), ("M", "Y")], [("X", "Y")], True),  # front-door
        ([("X", "Y")], [("X", "Y")], False),  # bow arc
        ([("Z", "X"), ("X", "Y")], [("X", "Y")], False),  # instrumental variable
        ([("X", "Y")], [], True),  # plain DAG
    ],
)
def test_gid_without_experiments_matches_id(
    edges: list[tuple[str, str]], bidirected: list[tuple[str, str]], identifiable: bool
) -> None:
    graph = CausalGraph(directed_edges=edges, bidirected_edges=bidirected)
    assert is_gid_identifiable(graph, {"X"}, {"Y"}, []) is identifiable
    assert is_identifiable_effect(graph, {"X"}, {"Y"}) is identifiable


def test_bow_arc_needs_an_experiment_on_x() -> None:
    graph = CausalGraph(directed_edges=[("X", "Y")], bidirected_edges=[("X", "Y")])
    assert is_gid_identifiable(graph, {"X"}, {"Y"}, [{"X"}]) is True  # do(X) breaks the hedge
    assert is_gid_identifiable(graph, {"X"}, {"Y"}, [{"Y"}]) is False  # do(Y) is useless here
    assert is_gid_identifiable(graph, {"X"}, {"Y"}, []) is False  # observation alone fails


def test_estimand_references_the_surrogate_experiment() -> None:
    graph = CausalGraph(directed_edges=[("X", "Y")], bidirected_edges=[("X", "Y")])
    formula = identify_effect_with_experiments(graph, {"X"}, {"Y"}, [{"X"}]).render()
    assert "do(X)" in formula


def test_gid_combines_observational_and_experimental_c_factors() -> None:
    # X -> Y1 (confounded by X<->Y1) and X -> Y2 (clean). P(Y1,Y2 | do(X)) is not observationally
    # identifiable (the Y1 hedge), but with do(X) the Y1 factor comes from the experiment while the
    # Y2 factor stays observational.
    graph = CausalGraph(directed_edges=[("X", "Y1"), ("X", "Y2")], bidirected_edges=[("X", "Y1")])
    assert is_gid_identifiable(graph, {"X"}, {"Y1", "Y2"}, []) is False
    assert is_gid_identifiable(graph, {"X"}, {"Y1", "Y2"}, [{"X"}]) is True


def _bow_scm() -> StructuralCausalModel:
    graph = CausalGraph(directed_edges=[("U", "X"), ("U", "Y"), ("X", "Y")])
    mechanisms: dict[str, Mechanism] = {
        "U": FunctionalMechanism([], lambda pa, u: u),
        "X": FunctionalMechanism(["U"], lambda pa, u: (pa["U"] + _flip(u, 0.2)) % 2),
        "Y": FunctionalMechanism(
            ["X", "U"], lambda pa, u: ((((pa["X"] + pa["U"]) > 0).float()) + _flip(u, 0.05)) % 2
        ),
    }
    exogenous: dict[str, Distribution] = {
        "U": Bernoulli(0.5),
        "X": Uniform(0.0, 1.0),
        "Y": Uniform(0.0, 1.0),
    }
    return StructuralCausalModel(graph, mechanisms, exogenous)


def _confounded_mediator_scm() -> StructuralCausalModel:
    # X -> M -> Y with X<->Y (via U) and M<->Y (via W): front-door fails, but do(X) identifies it.
    graph = CausalGraph(
        directed_edges=[("U", "X"), ("U", "Y"), ("W", "M"), ("W", "Y"), ("X", "M"), ("M", "Y")]
    )
    mechanisms: dict[str, Mechanism] = {
        "U": FunctionalMechanism([], lambda pa, u: u),
        "W": FunctionalMechanism([], lambda pa, u: u),
        "X": FunctionalMechanism(["U"], lambda pa, u: (pa["U"] + _flip(u, 0.2)) % 2),
        "M": FunctionalMechanism(
            ["X", "W"], lambda pa, u: ((((pa["X"] + pa["W"]) > 0).float()) + _flip(u, 0.1)) % 2
        ),
        "Y": FunctionalMechanism(
            ["M", "U", "W"],
            lambda pa, u: ((((pa["M"] + pa["U"] + pa["W"]) > 1).float()) + _flip(u, 0.05)) % 2,
        ),
    }
    exogenous: dict[str, Distribution] = {
        "U": Bernoulli(0.5),
        "W": Bernoulli(0.5),
        "X": Uniform(0.0, 1.0),
        "M": Uniform(0.0, 1.0),
        "Y": Uniform(0.0, 1.0),
    }
    return StructuralCausalModel(graph, mechanisms, exogenous)


def _randomized_experiment(scm: StructuralCausalModel, target: str, keep: list[str]) -> dict:
    """A perfect experiment on ``target``: data with the target randomized over {0, 1}."""
    low = scm.do({target: 0.0}).see(_N, seed=1)
    high = scm.do({target: 1.0}).see(_N, seed=2)
    joined = {name: torch.cat([low[name], high[name]]) for name in low}
    return _cols(joined, keep)


def _true_do(scm: StructuralCausalModel, target: str, value: int) -> float:
    return float(scm.do({target: float(value)}).see(_N, seed=7)["Y"].float().mean().item())


def test_bow_arc_estimate_matches_simulation() -> None:
    scm = _bow_scm()
    graph = CausalGraph(directed_edges=[("X", "Y")], bidirected_edges=[("X", "Y")])
    obs = _cols(scm.see(_N, seed=0), ["X", "Y"])
    experiments = {frozenset({"X"}): _randomized_experiment(scm, "X", ["X", "Y"])}
    for value in (0, 1):
        estimate = estimate_effect_with_experiments(
            graph, {"X"}, {"Y"}, obs, experiments, do={"X": value}
        )[(1,)]
        assert estimate == pytest.approx(_true_do(scm, "X", value), abs=0.03)


def test_confounded_mediator_estimate_matches_simulation() -> None:
    scm = _confounded_mediator_scm()
    graph = CausalGraph(
        directed_edges=[("X", "M"), ("M", "Y")], bidirected_edges=[("X", "Y"), ("M", "Y")]
    )
    assert is_identifiable_effect(graph, {"X"}, {"Y"}) is False  # not observationally identifiable
    obs = _cols(scm.see(_N, seed=0), ["X", "M", "Y"])
    experiments = {frozenset({"X"}): _randomized_experiment(scm, "X", ["X", "M", "Y"])}
    for value in (0, 1):
        estimate = estimate_effect_with_experiments(
            graph, {"X"}, {"Y"}, obs, experiments, do={"X": value}
        )[(1,)]
        assert estimate == pytest.approx(_true_do(scm, "X", value), abs=0.03)
