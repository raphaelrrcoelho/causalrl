"""Property-based tests for StructuralCausalModel invariants (spec §5).

These use hypothesis to generate constants, batch sizes, and seeds, asserting structural
properties that must hold for *every* input rather than for hand-picked examples.
"""

import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from torch.distributions import Bernoulli, Normal

from causalrl.scm.graph import CausalGraph
from causalrl.scm.mechanisms import FunctionalMechanism
from causalrl.scm.scm import StructuralCausalModel


def _deterministic_chain() -> StructuralCausalModel:
    # D -> X -> Y, with X = D and Y = X, so every node is exactly checkable.
    graph = CausalGraph(directed_edges=[("D", "X"), ("X", "Y")])
    mechanisms = {
        "D": FunctionalMechanism([], lambda pa, u: u),
        "X": FunctionalMechanism(["D"], lambda pa, u: pa["D"]),
        "Y": FunctionalMechanism(["X"], lambda pa, u: pa["X"]),
    }
    exogenous = {"D": Bernoulli(0.5), "X": Normal(0, 1), "Y": Normal(0, 1)}
    return StructuralCausalModel(graph, mechanisms, exogenous)


constants = st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False)
batch_sizes = st.integers(min_value=1, max_value=64)
seeds = st.integers(min_value=0, max_value=2**31 - 1)


@given(c=constants, n=batch_sizes, seed=seeds)
@settings(max_examples=40, deadline=None)
def test_do_forces_intervened_node_to_constant(c: float, n: int, seed: int) -> None:
    # Invariant: under do(X=c), X is exactly c regardless of its parents, and the
    # downstream deterministic Y = X is also c.
    s = _deterministic_chain().do({"X": c}).see(n, seed=seed)
    assert torch.allclose(s["X"], torch.full((n,), c))
    assert torch.allclose(s["Y"], torch.full((n,), c))


@given(n=batch_sizes, seed=seeds)
@settings(max_examples=40, deadline=None)
def test_see_returns_every_node_with_batch_shape(n: int, seed: int) -> None:
    # Invariant: see(n) returns one (n,)-shaped tensor per graph node.
    s = _deterministic_chain().see(n, seed=seed)
    assert set(s) == {"D", "X", "Y"}
    assert all(t.shape == (n,) for t in s.values())


@given(c=constants, seed=seeds)
@settings(max_examples=40, deadline=None)
def test_do_does_not_mutate_original(c: float, seed: int) -> None:
    # Invariant: do() returns a new SCM; the original still has X = D.
    scm = _deterministic_chain()
    _ = scm.do({"X": c})
    s = scm.see(128, seed=seed)
    assert torch.allclose(s["X"], s["D"])


@given(seed=seeds)
@settings(max_examples=40, deadline=None)
def test_counterfactual_with_no_intervention_reproduces_evidence(seed: int) -> None:
    # Invariant: a counterfactual that conditions on D=d with no intervention must return
    # samples consistent with that evidence (abduction preserves the conditioned value).
    scm = _deterministic_chain()
    cf = scm.counterfactual(evidence={"D": 1.0}, interventions={}, n=2000, seed=seed)
    assert torch.allclose(cf["D"], torch.ones_like(cf["D"]))
    assert torch.allclose(cf["X"], torch.ones_like(cf["X"]))  # X = D = 1
