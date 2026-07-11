"""The ID algorithm (Task 4): identifiability decisions and estimands validated by simulation.

Each identifiable case is checked by comparing the estimand's numeric evaluation on observational
data against the ground-truth interventional distribution obtained by simulating ``do(x)`` on the
same SCM. Non-identifiable cases (bow arc, instrumental variable) must raise.
"""

from __future__ import annotations

import numpy as np
import pytest
from torch import Tensor
from torch.distributions import Bernoulli, Distribution, Uniform

from causalrl.exceptions import CausalGraphError, NotIdentifiableError
from causalrl.identification.id_algorithm import (
    estimate_effect,
    identify_effect,
    is_identifiable_effect,
)
from causalrl.scm.graph import CausalGraph
from causalrl.scm.mechanisms import FunctionalMechanism, Mechanism
from causalrl.scm.scm import StructuralCausalModel

_N = 60_000


def _flip(u: Tensor, p: float) -> Tensor:
    return (u < p).float()


def _columns(samples: dict[str, Tensor], keep: list[str]) -> dict[str, np.ndarray]:
    return {name: samples[name].long().numpy() for name in keep}


def _backdoor_scm() -> StructuralCausalModel:
    """Confounded ``Z -> X``, ``Z -> Y``, ``X -> Y`` (Z observed): a back-door example."""
    graph = CausalGraph(directed_edges=[("Z", "X"), ("Z", "Y"), ("X", "Y")])
    mechanisms: dict[str, Mechanism] = {
        "Z": FunctionalMechanism([], lambda pa, u: u),
        "X": FunctionalMechanism(["Z"], lambda pa, u: (pa["Z"] + _flip(u, 0.2)) % 2),
        "Y": FunctionalMechanism(
            ["X", "Z"], lambda pa, u: ((((pa["X"] + pa["Z"]) > 0).float()) + _flip(u, 0.05)) % 2
        ),
    }
    exogenous: dict[str, Distribution] = {
        "Z": Bernoulli(0.5),
        "X": Uniform(0.0, 1.0),
        "Y": Uniform(0.0, 1.0),
    }
    return StructuralCausalModel(graph, mechanisms, exogenous)


def _frontdoor_scm() -> StructuralCausalModel:
    """``X -> M -> Y`` with a latent ``U`` confounding ``X`` and ``Y`` (the front-door example)."""
    graph = CausalGraph(directed_edges=[("U", "X"), ("U", "Y"), ("X", "M"), ("M", "Y")])
    mechanisms: dict[str, Mechanism] = {
        "U": FunctionalMechanism([], lambda pa, u: u),
        "X": FunctionalMechanism(["U"], lambda pa, u: (pa["U"] + _flip(u, 0.2)) % 2),
        "M": FunctionalMechanism(["X"], lambda pa, u: (pa["X"] + _flip(u, 0.1)) % 2),
        "Y": FunctionalMechanism(
            ["M", "U"], lambda pa, u: ((((pa["M"] + pa["U"]) > 0).float()) + _flip(u, 0.05)) % 2
        ),
    }
    exogenous: dict[str, Distribution] = {
        "U": Bernoulli(0.5),
        "X": Uniform(0.0, 1.0),
        "M": Uniform(0.0, 1.0),
        "Y": Uniform(0.0, 1.0),
    }
    return StructuralCausalModel(graph, mechanisms, exogenous)


def _true_do(scm: StructuralCausalModel, treatment: str, value: int, outcome: str) -> float:
    """Ground-truth ``P(outcome = 1 | do(treatment = value))`` by simulating the mutilated SCM."""
    samples = scm.do({treatment: float(value)}).see(_N, seed=7)
    return float(samples[outcome].float().mean().item())


def test_backdoor_estimand_matches_simulation_and_corrects_confounding() -> None:
    scm = _backdoor_scm()
    graph = CausalGraph(directed_edges=[("Z", "X"), ("Z", "Y"), ("X", "Y")])
    obs = _columns(scm.see(_N, seed=0), ["Z", "X", "Y"])

    for value in (0, 1):
        estimate = estimate_effect(graph, {"X"}, {"Y"}, obs, do={"X": value})[(1,)]
        assert estimate == pytest.approx(_true_do(scm, "X", value, "Y"), abs=0.03)

    # The adjustment matters: the naive conditional P(Y=1 | X=0) is badly confounded by Z.
    naive = float(obs["Y"][obs["X"] == 0].mean())
    causal = estimate_effect(graph, {"X"}, {"Y"}, obs, do={"X": 0})[(1,)]
    assert abs(naive - causal) > 0.1


def test_frontdoor_is_identifiable_and_matches_simulation() -> None:
    scm = _frontdoor_scm()
    graph = CausalGraph(directed_edges=[("X", "M"), ("M", "Y")], bidirected_edges=[("X", "Y")])
    assert is_identifiable_effect(graph, {"X"}, {"Y"})
    obs = _columns(scm.see(_N, seed=0), ["X", "M", "Y"])  # U is latent, dropped

    for value in (0, 1):
        estimate = estimate_effect(graph, {"X"}, {"Y"}, obs, do={"X": value})[(1,)]
        assert estimate == pytest.approx(_true_do(scm, "X", value, "Y"), abs=0.03)

    # Front-door recovers the effect despite latent confounding the naive estimate cannot.
    naive = float(obs["Y"][obs["X"] == 0].mean())
    causal = estimate_effect(graph, {"X"}, {"Y"}, obs, do={"X": 0})[(1,)]
    assert abs(naive - causal) > 0.1


def test_frontdoor_estimand_renders_a_do_free_formula() -> None:
    graph = CausalGraph(directed_edges=[("X", "M"), ("M", "Y")], bidirected_edges=[("X", "Y")])
    formula = identify_effect(graph, {"X"}, {"Y"}, return_certificate=False).render()
    assert "do(" not in formula
    assert "P(" in formula


def test_bow_arc_is_not_identifiable() -> None:
    graph = CausalGraph(directed_edges=[("X", "Y")], bidirected_edges=[("X", "Y")])
    assert not is_identifiable_effect(graph, {"X"}, {"Y"})
    with pytest.raises(NotIdentifiableError) as excinfo:
        identify_effect(graph, {"X"}, {"Y"})
    assert excinfo.value.witness is not None


def test_instrumental_variable_is_not_identifiable() -> None:
    # Z -> X -> Y with X <-> Y: the textbook instrument gives only bounds, not point ID.
    graph = CausalGraph(directed_edges=[("Z", "X"), ("X", "Y")], bidirected_edges=[("X", "Y")])
    assert not is_identifiable_effect(graph, {"X"}, {"Y"})


def test_plain_dag_effect_is_identifiable() -> None:
    graph = CausalGraph(directed_edges=[("X", "Y")])
    assert is_identifiable_effect(graph, {"X"}, {"Y"})


def test_malformed_queries_raise() -> None:
    graph = CausalGraph(directed_edges=[("X", "Y")])
    with pytest.raises(CausalGraphError):
        identify_effect(graph, {"X"}, {"Q"})  # unknown node
    with pytest.raises(CausalGraphError):
        identify_effect(graph, {"X"}, {"X"})  # overlapping treatment/outcome
    with pytest.raises(CausalGraphError):
        identify_effect(graph, {"X"}, set())  # empty outcome
