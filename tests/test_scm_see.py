import torch
from torch.distributions import Bernoulli, Normal

from causalrl.scm.graph import CausalGraph
from causalrl.scm.mechanisms import FunctionalMechanism
from causalrl.scm.scm import StructuralCausalModel


def build_xor_scm() -> StructuralCausalModel:
    # D, B independent Bernoulli(0.5); I = D xor B
    graph = CausalGraph(directed_edges=[("D", "I"), ("B", "I")])
    mechanisms = {
        "D": FunctionalMechanism([], lambda pa, u: u),
        "B": FunctionalMechanism([], lambda pa, u: u),
        "I": FunctionalMechanism(["D", "B"], lambda pa, u: (pa["D"] != pa["B"]).float()),
    }
    exogenous = {
        "D": Bernoulli(0.5),
        "B": Bernoulli(0.5),
        "I": Normal(0.0, 1.0),  # unused by the deterministic mechanism
    }
    return StructuralCausalModel(graph, mechanisms, exogenous)


def test_see_returns_all_nodes_with_batch_shape():
    scm = build_xor_scm()
    sample = scm.see(64, seed=0)
    assert set(sample) == {"D", "B", "I"}
    assert sample["I"].shape == (64,)


def test_see_respects_xor_relation():
    scm = build_xor_scm()
    s = scm.see(1000, seed=1)
    expected = (s["D"] != s["B"]).float()
    assert torch.allclose(s["I"], expected)


def test_see_is_reproducible_with_seed():
    scm = build_xor_scm()
    a = scm.see(32, seed=7)
    b = scm.see(32, seed=7)
    assert torch.allclose(a["D"], b["D"])
