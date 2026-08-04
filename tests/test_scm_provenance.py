import pytest
from torch.distributions import Uniform

from causalrl.exceptions import NotIdentifiableError
from causalrl.scm.graph import CausalGraph
from causalrl.scm.mechanisms import FunctionalMechanism, Mechanism
from causalrl.scm.scm import StructuralCausalModel


def _scm(*, provenance: str, invertible: bool) -> StructuralCausalModel:
    graph = CausalGraph(directed_edges=[("A", "Y")])
    mech_y = FunctionalMechanism(["A"], lambda pa, u: pa["A"] + u)
    mech_y.invertible = invertible  # type: ignore[attr-defined]
    mechanisms: dict[str, Mechanism] = {
        "A": FunctionalMechanism([], lambda pa, u: u),
        "Y": mech_y,
    }
    exogenous = {"A": Uniform(0.0, 1.0), "Y": Uniform(0.0, 1.0)}
    return StructuralCausalModel(graph, mechanisms, exogenous, provenance=provenance)


def test_specified_scm_defaults_to_specified_provenance():
    scm = StructuralCausalModel(
        CausalGraph(directed_edges=[("A", "Y")]),
        {
            "A": FunctionalMechanism([], lambda pa, u: u),
            "Y": FunctionalMechanism(["A"], lambda pa, u: pa["A"] + u),
        },
        {"A": Uniform(0.0, 1.0), "Y": Uniform(0.0, 1.0)},
    )
    assert scm.provenance == "specified"
    assert scm.fit_report is None


def test_specified_scm_abduct_is_unguarded_even_when_non_invertible():
    scm = _scm(provenance="specified", invertible=False)
    post = scm.abduct(known={"A": 0.5}, n=8)
    assert len(post) == 8


def test_fitted_scm_with_invertible_nodes_abducts_normally():
    scm = _scm(provenance="fitted", invertible=True)
    post = scm.abduct(known={"A": 0.5}, n=8)
    assert len(post) == 8


def test_fitted_scm_with_non_invertible_node_raises_and_names_the_alternative():
    scm = _scm(provenance="fitted", invertible=False)
    with pytest.raises(NotIdentifiableError, match="counterfactual_interval"):
        scm.abduct(known={"A": 0.5}, n=8)


def test_internal_abduct_bypasses_the_guard():
    scm = _scm(provenance="fitted", invertible=False)
    post = scm._abduct(known={"A": 0.5}, n=8)
    assert len(post) == 8


def test_counterfactual_inherits_the_guard():
    scm = _scm(provenance="fitted", invertible=False)
    with pytest.raises(NotIdentifiableError, match="counterfactual_interval"):
        scm.counterfactual({"Y": 0.5}, {"A": 1.0}, n=8)
