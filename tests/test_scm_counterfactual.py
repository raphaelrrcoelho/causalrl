import pytest
import torch
from torch.distributions import Bernoulli, Normal

from causalrl.exceptions import RealizabilityError
from causalrl.scm.graph import CausalGraph
from causalrl.scm.mechanisms import FunctionalMechanism
from causalrl.scm.scm import StructuralCausalModel


def build_scm() -> StructuralCausalModel:
    # U -> X -> Y ; Y = X xor U  (so flipping X flips Y for a fixed U)
    graph = CausalGraph(directed_edges=[("U", "X"), ("X", "Y"), ("U", "Y")])
    mechanisms = {
        "U": FunctionalMechanism([], lambda pa, u: u),
        "X": FunctionalMechanism(["U"], lambda pa, u: pa["U"]),
        "Y": FunctionalMechanism(["X", "U"], lambda pa, u: (pa["X"] != pa["U"]).float()),
    }
    exogenous = {"U": Bernoulli(0.5), "X": Normal(0, 1), "Y": Normal(0, 1)}
    return StructuralCausalModel(graph, mechanisms, exogenous)


def test_counterfactual_flips_outcome():
    # Observed X=0 (so U=0, Y = 0 xor 0 = 0). Counterfactual: had X been 1, Y = 1 xor 0 = 1.
    scm = build_scm()
    cf = scm.counterfactual(evidence={"X": 0.0}, interventions={"X": 1.0}, n=2000, seed=0)
    assert torch.allclose(cf["Y"].mean(), torch.tensor(1.0), atol=1e-6)


def test_counterfactual_consistent_with_evidence_when_no_intervention():
    scm = build_scm()
    cf = scm.counterfactual(evidence={"X": 1.0}, interventions={"X": 1.0}, n=2000, seed=1)
    # U=1 (since X=U=1), Y = 1 xor 1 = 0
    assert torch.allclose(cf["Y"].mean(), torch.tensor(0.0), atol=1e-6)


def test_counterfactual_no_intervention_returns_factual_consistent():
    # Empty interventions exercises the `else self` branch: pure abduction + prediction.
    # Evidence X=0 -> U=0, so with no intervention X stays 0 and Y = 0 xor 0 = 0.
    scm = build_scm()
    cf = scm.counterfactual(evidence={"X": 0.0}, interventions={}, n=2000, seed=3)
    assert torch.allclose(cf["X"].mean(), torch.tensor(0.0), atol=1e-6)
    assert torch.allclose(cf["Y"].mean(), torch.tensor(0.0), atol=1e-6)


def test_impossible_evidence_raises():
    scm = build_scm()
    with pytest.raises(RealizabilityError):
        scm.counterfactual(evidence={"X": 0.5}, interventions={"X": 1.0}, n=500, seed=2)
