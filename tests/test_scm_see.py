import pytest
import torch
from torch.distributions import Bernoulli, Normal

from causalrl.exceptions import CausalGraphError
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


def test_seeded_see_does_not_mutate_global_torch_rng_state():
    torch.manual_seed(991)
    before = torch.random.get_rng_state().clone()
    _ = build_xor_scm().see(32, seed=7)
    assert torch.equal(torch.random.get_rng_state(), before)


def test_unseeded_see_uses_private_rng_without_mutating_global_state():
    torch.manual_seed(991)
    before = torch.random.get_rng_state().clone()
    scm = build_xor_scm()
    first = scm.see(32)
    second = scm.see(32)
    assert torch.equal(torch.random.get_rng_state(), before)
    assert not torch.equal(first["D"], second["D"])


def test_executable_scm_rejects_bidirected_admg_graphs():
    graph = CausalGraph(directed_edges=[], bidirected_edges=[("X", "Y")])
    mechanisms = {
        "X": FunctionalMechanism([], lambda pa, u: u),
        "Y": FunctionalMechanism([], lambda pa, u: u),
    }
    exogenous = {"X": Bernoulli(0.5), "Y": Bernoulli(0.5)}
    with pytest.raises(CausalGraphError, match="explicit latent"):
        StructuralCausalModel(graph, mechanisms, exogenous)


def test_executable_scm_rejects_missing_exogenous_distribution_at_construction():
    graph = CausalGraph(directed_edges=[("X", "Y")])
    mechanisms = {
        "X": FunctionalMechanism([], lambda pa, u: u),
        "Y": FunctionalMechanism(["X"], lambda pa, u: pa["X"]),
    }
    with pytest.raises(CausalGraphError, match="exogenous"):
        StructuralCausalModel(graph, mechanisms, {"X": Bernoulli(0.5)})


def test_executable_scm_rejects_mechanism_parent_graph_mismatch():
    graph = CausalGraph(directed_edges=[("X", "Y")])
    mechanisms = {
        "X": FunctionalMechanism([], lambda pa, u: u),
        "Y": FunctionalMechanism([], lambda pa, u: u),
    }
    exogenous = {"X": Bernoulli(0.5), "Y": Bernoulli(0.5)}
    with pytest.raises(CausalGraphError, match="parents"):
        StructuralCausalModel(graph, mechanisms, exogenous)
