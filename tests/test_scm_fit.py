import numpy as np
import pytest

from causalrl.exceptions import NotIdentifiableError
from causalrl.scm.fit import fit_scm
from causalrl.scm.fitters import LinearGaussianFit
from causalrl.scm.graph import CausalGraph


def _linear_data(n: int = 20_000, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    a = 0.8 * z + rng.normal(scale=0.5, size=n)
    y = 2.0 * a + 1.5 * z + rng.normal(scale=0.3, size=n)
    return {"Z": z, "A": a, "Y": y}


def _discrete_data(n: int = 40_000, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    z = (rng.random(n) < 0.5).astype(int)
    a = (rng.random(n) < np.where(z == 1, 0.8, 0.2)).astype(int)
    y = (rng.random(n) < np.where(a + z > 1, 0.9, 0.1)).astype(int)
    return {"Z": z, "A": a, "Y": y}


def test_fit_scm_returns_a_fitted_structural_causal_model():
    graph = CausalGraph(directed_edges=[("Z", "A"), ("Z", "Y"), ("A", "Y")])
    scm = fit_scm(_linear_data(), graph=graph)
    assert scm.provenance == "fitted"
    assert scm.fit_report is not None
    assert sorted(scm.graph.nodes) == ["A", "Y", "Z"]


def test_fitted_linear_scm_recovers_the_interventional_mean():
    graph = CausalGraph(directed_edges=[("Z", "A"), ("Z", "Y"), ("A", "Y")])
    scm = fit_scm(
        _linear_data(),
        graph=graph,
        families={"A": LinearGaussianFit(), "Y": LinearGaussianFit(), "Z": LinearGaussianFit()},
    )
    # True E[Y|do(A=1)] = 2*1 + 1.5*E[Z] = 2.0
    drawn = scm.do({"A": 1.0}).see(40_000, seed=0)["Y"]
    assert abs(float(drawn.mean()) - 2.0) < 0.1


def test_fitted_discrete_scm_recovers_the_backdoor_adjusted_effect():
    graph = CausalGraph(directed_edges=[("Z", "A"), ("Z", "Y"), ("A", "Y")])
    scm = fit_scm(_discrete_data(), graph=graph)
    # E[Y|do(A=1)] = sum_z P(z) P(Y=1|A=1,z) = 0.5*0.1 + 0.5*0.9 = 0.5
    drawn = scm.do({"A": 1.0}).see(40_000, seed=0)["Y"]
    assert abs(float(drawn.mean()) - 0.5) < 0.03


def test_fit_scm_dispatches_families_by_dtype():
    graph = CausalGraph(directed_edges=[("Z", "A"), ("Z", "Y"), ("A", "Y")])
    mixed = _discrete_data()
    mixed["Y"] = mixed["Y"].astype(float) + np.random.default_rng(1).normal(
        scale=0.1, size=len(mixed["Y"])
    )
    report = fit_scm(mixed, graph=graph).fit_report
    families = {node.node: node.family for node in report.nodes}
    assert families["Z"] == "tabular_cpt"
    assert families["Y"] == "anm"


def test_fit_scm_honours_a_per_node_family_override():
    graph = CausalGraph(directed_edges=[("Z", "A"), ("Z", "Y"), ("A", "Y")])
    report = fit_scm(_linear_data(), graph=graph, families={"Y": LinearGaussianFit()}).fit_report
    families = {node.node: node.family for node in report.nodes}
    assert families["Y"] == "linear_gaussian"


def test_fit_scm_reports_invertibility_per_node():
    graph = CausalGraph(directed_edges=[("Z", "A"), ("Z", "Y"), ("A", "Y")])
    report = fit_scm(_discrete_data(), graph=graph).fit_report
    assert all(node.invertible is False for node in report.nodes)


def test_fit_scm_refuses_a_graph_with_bidirected_edges():
    graph = CausalGraph(directed_edges=[("A", "Y")], bidirected_edges=[("A", "Y")])
    with pytest.raises(NotIdentifiableError, match="latent confounding"):
        fit_scm({"A": np.zeros(10), "Y": np.zeros(10)}, graph=graph)


def test_fit_scm_rejects_data_missing_a_graph_node():
    graph = CausalGraph(directed_edges=[("A", "Y")])
    with pytest.raises(KeyError, match="Y"):
        fit_scm({"A": np.zeros(10)}, graph=graph)


def test_fit_report_summary_lists_every_node():
    graph = CausalGraph(directed_edges=[("Z", "A"), ("Z", "Y"), ("A", "Y")])
    summary = fit_scm(_linear_data(), graph=graph).fit_report.summary()
    for node in ("Z", "A", "Y"):
        assert node in summary


def test_fitted_scm_do_preserves_provenance_and_the_l3_guard():
    # F1 regression: do() used to return StructuralCausalModel(graph, mechanisms, exogenous)
    # with no provenance/fit_report, so a mutilated fitted SCM silently reverted to
    # provenance="specified" and abduct's L3 guard never fired for the un-intervened
    # non-invertible nodes it still carried.
    graph = CausalGraph(directed_edges=[("Z", "A"), ("Z", "Y"), ("A", "Y")])
    scm = fit_scm(_discrete_data(), graph=graph)
    mutilated = scm.do({"A": 1.0})
    assert mutilated.provenance == "fitted"
    assert mutilated.fit_report is scm.fit_report
    with pytest.raises(NotIdentifiableError, match="counterfactual_interval"):
        mutilated.abduct(n=8)


def test_fit_scm_discrete_scm_refuses_point_counterfactuals():
    # F2 regression: test_fit_scm_reports_invertibility_per_node only reads
    # report.nodes[i].invertible, which comes straight from FittedMechanism.invertible and says
    # nothing about whether fit.py actually wired that flag onto the mechanism OBJECT (the
    # setattr/assignment at fit.py's node loop). This test exercises the real consumers --
    # non_invertible_nodes() and abduct()'s L3 guard -- that read the flag off the mechanism.
    graph = CausalGraph(directed_edges=[("Z", "A"), ("Z", "Y"), ("A", "Y")])
    scm = fit_scm(_discrete_data(), graph=graph)
    assert scm.non_invertible_nodes()
    with pytest.raises(NotIdentifiableError, match="counterfactual_interval"):
        scm.abduct(known={"A": 0.5}, n=8)
