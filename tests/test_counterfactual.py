"""Closed-form oracles for the Layer-3 counterfactual estimands."""

from __future__ import annotations

import pytest
import torch
from torch.distributions import Bernoulli, Distribution, Uniform

from causalrl.exceptions import CausalGraphError
from causalrl.identification.counterfactual import (
    counterfactual_expectation,
    effect_of_treatment_on_treated,
)
from causalrl.scm.graph import CausalGraph
from causalrl.scm.mechanisms import FunctionalMechanism, Mechanism
from causalrl.scm.scm import StructuralCausalModel

# Y ~ Bernoulli(p(X, U)): p(1,1)=0.9, p(1,0)=0.2, p(0,1)=0.6, p(0,0)=0.1.
# ATE = E[Y_1] - E[Y_0] = (0.5*0.9 + 0.5*0.2) - (0.5*0.6 + 0.5*0.1) = 0.55 - 0.35 = 0.20.


def _reward(pa: dict[str, torch.Tensor], u: torch.Tensor) -> torch.Tensor:
    x, uu = pa["X"], pa["U"]
    p = torch.full_like(x, 0.1)
    p = torch.where((x == 1) & (uu == 1), torch.full_like(x, 0.9), p)
    p = torch.where((x == 1) & (uu == 0), torch.full_like(x, 0.2), p)
    p = torch.where((x == 0) & (uu == 1), torch.full_like(x, 0.6), p)
    return (u < p).float()


def _confounded_scm() -> StructuralCausalModel:
    """``U -> X``, ``U -> Y``, ``X -> Y`` with ``X = U`` (full confounding)."""
    graph = CausalGraph(directed_edges=[("U", "X"), ("U", "Y"), ("X", "Y")])
    mechanisms: dict[str, Mechanism] = {
        "U": FunctionalMechanism([], lambda pa, u: u),
        "X": FunctionalMechanism(["U"], lambda pa, u: pa["U"]),
        "Y": FunctionalMechanism(["X", "U"], _reward),
    }
    exogenous: dict[str, Distribution] = {
        "U": Bernoulli(0.5),
        "X": Uniform(0.0, 1.0),
        "Y": Uniform(0.0, 1.0),
    }
    return StructuralCausalModel(graph, mechanisms, exogenous)


def _unconfounded_scm() -> StructuralCausalModel:
    """Same reward, but ``X`` is exogenous and independent of ``U`` (no ``U -> X`` edge)."""
    graph = CausalGraph(directed_edges=[("U", "Y"), ("X", "Y")])
    mechanisms: dict[str, Mechanism] = {
        "U": FunctionalMechanism([], lambda pa, u: u),
        "X": FunctionalMechanism([], lambda pa, u: u),
        "Y": FunctionalMechanism(["X", "U"], _reward),
    }
    exogenous: dict[str, Distribution] = {
        "U": Bernoulli(0.5),
        "X": Bernoulli(0.5),
        "Y": Uniform(0.0, 1.0),
    }
    return StructuralCausalModel(graph, mechanisms, exogenous)


def test_counterfactual_expectation_matches_closed_form() -> None:
    scm = _confounded_scm()
    # X = 0 abducts U = 0; do(X=1) gives p(1, 0) = 0.2.
    v1 = counterfactual_expectation(
        scm, outcome="Y", intervention={"X": 1.0}, evidence={"X": 0.0}, n=60_000, seed=0
    )
    assert abs(v1 - 0.2) < 0.02
    # X = 1 abducts U = 1; do(X=0) gives p(0, 1) = 0.6.
    v2 = counterfactual_expectation(
        scm, outcome="Y", intervention={"X": 0.0}, evidence={"X": 1.0}, n=60_000, seed=1
    )
    assert abs(v2 - 0.6) < 0.02


def test_ett_matches_closed_form_and_differs_from_ate() -> None:
    scm = _confounded_scm()
    # ETT on the treated (X=1 abducts U=1): E[Y_1|U=1] - E[Y_0|U=1] = 0.9 - 0.6 = 0.30.
    ett = effect_of_treatment_on_treated(
        scm, treatment="X", outcome="Y", treated=1.0, control=0.0, n=60_000, seed=7
    )
    assert abs(ett - 0.30) < 0.02
    ate = counterfactual_expectation(
        scm, outcome="Y", intervention={"X": 1.0}, evidence={}, n=60_000, seed=2
    ) - counterfactual_expectation(
        scm, outcome="Y", intervention={"X": 0.0}, evidence={}, n=60_000, seed=3
    )
    assert abs(ate - 0.20) < 0.02
    assert abs(ett - ate) > 0.05  # confounding makes them differ


def test_ett_equals_ate_without_confounding() -> None:
    scm = _unconfounded_scm()
    ett = effect_of_treatment_on_treated(
        scm, treatment="X", outcome="Y", treated=1.0, control=0.0, n=60_000, seed=5
    )
    # With X independent of U, conditioning on X=1 leaves U marginal, so ETT == ATE == 0.20.
    assert abs(ett - 0.20) < 0.03


def test_unknown_node_raises() -> None:
    scm = _confounded_scm()
    with pytest.raises(CausalGraphError):
        counterfactual_expectation(scm, outcome="Z", intervention={"X": 1.0}, evidence={}, n=10)
    with pytest.raises(CausalGraphError):
        counterfactual_expectation(scm, outcome="Y", intervention={"W": 1.0}, evidence={}, n=10)
