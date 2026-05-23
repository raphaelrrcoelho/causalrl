import torch
from torch.distributions import Bernoulli, Normal

from causalrl.scm.graph import CausalGraph
from causalrl.scm.mechanisms import FunctionalMechanism
from causalrl.scm.scm import StructuralCausalModel


def build_scm() -> StructuralCausalModel:
    # D -> X -> Y, with Y = X + D
    graph = CausalGraph(directed_edges=[("D", "X"), ("X", "Y")])
    mechanisms = {
        "D": FunctionalMechanism([], lambda pa, u: u),
        "X": FunctionalMechanism(["D"], lambda pa, u: pa["D"]),
        "Y": FunctionalMechanism(["X"], lambda pa, u: pa["X"]),
    }
    exogenous = {"D": Bernoulli(0.5), "X": Normal(0, 1), "Y": Normal(0, 1)}
    return StructuralCausalModel(graph, mechanisms, exogenous)


def test_do_overrides_node_value():
    scm = build_scm()
    intervened = scm.do({"X": 1.0})
    s = intervened.see(100, seed=0)
    assert torch.allclose(s["X"], torch.ones(100))
    assert torch.allclose(s["Y"], torch.ones(100))  # Y = X = 1


def test_do_makes_node_independent_of_parents():
    # property: under do(X), X no longer depends on D
    scm = build_scm()
    s = scm.do({"X": 0.0}).see(500, seed=3)
    # X is constant 0 regardless of D
    assert torch.allclose(s["X"], torch.zeros(500))


def test_do_does_not_mutate_original():
    scm = build_scm()
    _ = scm.do({"X": 1.0})
    s = scm.see(200, seed=5)
    # original X still equals D (not forced to 1)
    assert torch.allclose(s["X"], s["D"])
