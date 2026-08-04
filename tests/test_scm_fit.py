import time

import numpy as np
import pytest
import torch
from torch.distributions import Uniform

from causalrl.discovery import CPDAG
from causalrl.exceptions import NotIdentifiableError
from causalrl.scm.fit import fit_scm, fit_scm_mec
from causalrl.scm.fitters import (
    ANMFit,
    FittedMechanism,
    LinearGaussianFit,
    PoissonGLMFit,
    evaluate_holdout,
)
from causalrl.scm.graph import CausalGraph
from causalrl.scm.mechanisms import FunctionalMechanism


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


def test_fit_scm_auto_selected_cpt_refuses_continuous_parents():
    # I5 regression through the route users actually hit: _is_discrete inspects only the CHILD
    # column, so the commonest real-data shape -- a binary treatment with continuous confounders
    # -- auto-selects TabularCPT for A and asks for one row per (X1, X2) configuration:
    # 640 x 640 = 409,600 rows from 800 samples. The error must name the way out.
    graph = CausalGraph(directed_edges=[("X1", "A"), ("X2", "A")])
    rng = np.random.default_rng(0)
    n = 800
    data = {
        "X1": rng.normal(size=n),
        "X2": rng.normal(size=n),
        "A": (rng.random(n) < 0.5).astype(int),
    }
    with pytest.raises(ValueError, match="families="):
        fit_scm(data, graph=graph)


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


def test_fit_scm_holdout_score_penalizes_misspecification():
    # Sanity check on evaluate_holdout's basic behaviour, NOT a regression guard for the
    # refit-vs-deployed-mechanism defect below: this ~0.79 gap is model-FAMILY capability (a
    # nonlinear ANM beats a linear fit on a nonlinear DGP) and holds equally under the correct
    # evaluate_holdout() path and the old buggy "refit fresh on holdout, score that in-sample"
    # path (reviewer-verified: NEW=0.9771/OLD=0.9764 for ANM, NEW=0.1863/OLD=0.1869 for linear,
    # on this exact data/split -- both scoring methods agree here). Kept because it still catches
    # gross breakage (a crash, or both families reporting the same/wrong-signed number); the test
    # below is what actually guards F3's fix.
    graph = CausalGraph(directed_edges=[("X", "Y")])
    rng = np.random.default_rng(2)
    x = rng.uniform(-2.0, 2.0, size=8000)
    y = np.sin(3.0 * x) + rng.normal(scale=0.1, size=8000)
    data = {"X": x, "Y": y}
    well_specified = fit_scm(data, graph=graph, families={"X": LinearGaussianFit(), "Y": ANMFit()})
    misspecified = fit_scm(
        data, graph=graph, families={"X": LinearGaussianFit(), "Y": LinearGaussianFit()}
    )
    well_score = next(n.holdout_score for n in well_specified.fit_report.nodes if n.node == "Y")
    mis_score = next(n.holdout_score for n in misspecified.fit_report.nodes if n.node == "Y")
    assert well_score > mis_score + 0.3


def test_fit_scm_holdout_score_is_pessimistic_about_a_high_capacity_fit_not_a_refit_in_disguise():
    # F3 regression, the actual guard: X and Y are independent (no true relationship), so the
    # best ANY correctly-scored DEPLOYED mechanism can do on genuinely unseen data is ~R^2=0 --
    # regardless of model family. But a high-capacity fitter (60 RBF features, ~61 parameters)
    # refit FRESH on a small 30-point holdout partition can nearly interpolate that partition's
    # own noise, giving an inflated, optimistic R^2. That is exactly the old bug: scoring a fresh
    # refit on holdout instead of the mechanism actually deployed (trained on the other 120
    # points, which cannot know this partition's noise pattern).
    #
    # Calibrated on this exact data/split (see task-6-report.md): the correct (deployed-mechanism)
    # score is -0.28; a fresh refit on the same holdout partition scores +0.33. 0.1 sits with a
    # wide margin (>=0.18) on both sides of that gap, robust to run-to-run numerical noise.
    graph = CausalGraph(directed_edges=[("X", "Y")])
    rng = np.random.default_rng(0)
    n = 150
    x = rng.normal(size=n)
    y = rng.normal(size=n)  # independent of x: no true X -> Y relationship whatsoever
    scm = fit_scm(
        {"X": x, "Y": y},
        graph=graph,
        families={"X": LinearGaussianFit(), "Y": ANMFit(n_features=60, seed=0)},
    )
    score = next(n.holdout_score for n in scm.fit_report.nodes if n.node == "Y")
    assert score < 0.1


def test_evaluate_holdout_raises_a_clear_error_when_log_prob_is_missing():
    # Minor hardening (fix round 2): a user-supplied MechanismFitter reporting invertible=False
    # without attaching log_prob would otherwise crash inside evaluate_holdout with an opaque,
    # unexplained AttributeError. Verify the raised error is clear and names what's missing.
    broken = FittedMechanism(
        mechanism=FunctionalMechanism([], lambda pa, u: u),
        noise=Uniform(0.0, 1.0),
        invertible=False,
        score=0.0,
    )
    with pytest.raises(AttributeError, match="log_prob"):
        evaluate_holdout(broken, {}, np.zeros(4))


@pytest.mark.parametrize("holdout", [0.0, -0.5, 1.0, 1.5])
def test_fit_scm_rejects_a_holdout_outside_the_unit_interval(holdout):
    # I6 regression: 1.5 and -0.5 were both accepted, and holdout=0.0 silently reported the
    # IN-SAMPLE score under NodeFit.holdout_score, whose docstring promises data the fit never
    # saw. holdout >= 1.0 left a one-row training set.
    graph = CausalGraph(directed_edges=[("A", "Y")])
    rng = np.random.default_rng(0)
    data = {"A": rng.normal(size=200), "Y": rng.normal(size=200)}
    with pytest.raises(ValueError, match="holdout"):
        fit_scm(data, graph=graph, holdout=holdout)


def test_fit_scm_mec_refuses_a_cpdag_whose_enumeration_would_not_finish():
    # I4 regression: _enumerate_mec materialised all 2**k orientations, constructing a CausalGraph
    # per candidate, BEFORE the max_members cap was consulted -- so an ordinary PC output on 10
    # variables (~25 undirected edges) hung instead of raising the documented error. 2**25 = 33.5M
    # candidates at the ~37us each measured here is roughly 20 minutes. max_members caps the class,
    # not the search, so raising it must NOT re-open the hang: the budget is checked independently.
    variables = tuple(f"V{i}" for i in range(26))
    cpdag = CPDAG(
        variables,
        frozenset(),
        frozenset(frozenset((f"V{i}", f"V{i + 1}")) for i in range(25)),
    )
    data = {v: np.zeros(4) for v in variables}
    start = time.perf_counter()
    with pytest.raises(ValueError, match="_MAX_MEC_ENUMERATION"):
        fit_scm_mec(data, cpdag=cpdag, max_members=10**9)
    assert time.perf_counter() - start < 5.0


def test_fit_scm_mec_fits_every_orientation_of_a_single_undirected_edge():
    cpdag = CPDAG(("A", "Y"), frozenset(), frozenset({frozenset(("A", "Y"))}))
    rng = np.random.default_rng(0)
    a = rng.normal(size=4000)
    data = {"A": a, "Y": 2.0 * a + rng.normal(scale=0.5, size=4000)}
    models = fit_scm_mec(data, cpdag=cpdag)
    assert len(models) == 2
    assert {tuple(m.graph.directed_edges[0]) for m in models} == {("A", "Y"), ("Y", "A")}
    assert all(m.provenance == "fitted" for m in models)


def test_fit_scm_mec_excludes_orientations_that_create_a_new_collider():
    # A - B - C chain with A, C non-adjacent: A -> B <- C is a v-structure the CPDAG lacks.
    cpdag = CPDAG(
        ("A", "B", "C"),
        frozenset(),
        frozenset({frozenset(("A", "B")), frozenset(("B", "C"))}),
    )
    rng = np.random.default_rng(1)
    b = rng.normal(size=2000)
    data = {"A": b + rng.normal(size=2000), "B": b, "C": b + rng.normal(size=2000)}
    models = fit_scm_mec(data, cpdag=cpdag)
    for model in models:
        parents_of_b = model.graph.parents("B")
        assert set(parents_of_b) != {"A", "C"}
    assert len(models) == 3


def test_fit_scm_mec_raises_above_the_cap_and_names_the_real_size():
    cpdag = CPDAG(
        ("A", "B", "C"),
        frozenset(),
        frozenset({frozenset(("A", "B")), frozenset(("B", "C"))}),
    )
    rng = np.random.default_rng(2)
    data = {k: rng.normal(size=500) for k in ("A", "B", "C")}
    with pytest.raises(ValueError, match="3 member"):
        fit_scm_mec(data, cpdag=cpdag, max_members=2)


def test_fit_scm_accepts_lag_embedded_node_names():
    # Lag-unrolled frames use names like `M1_0@t-2`. Nothing may parse a node name.
    rng = np.random.default_rng(0)
    n = 4000
    names = ["M1_0@t-2", "M1_0@t-1", "M1_0@t-0"]
    x2 = rng.normal(size=n)
    x1 = 0.7 * x2 + rng.normal(scale=0.5, size=n)
    x0 = 0.4 * x1 + 0.3 * x2 + rng.normal(scale=0.5, size=n)
    data = {names[0]: x2, names[1]: x1, names[2]: x0}
    graph = CausalGraph(
        directed_edges=[(names[0], names[1]), (names[1], names[2]), (names[0], names[2])]
    )
    scm = fit_scm(data, graph=graph, families=dict.fromkeys(names, LinearGaussianFit()))
    assert [f.node for f in scm.fit_report.nodes] == names
    assert scm.fit_report.nodes[2].parents == (names[1], names[0]) or set(
        scm.fit_report.nodes[2].parents
    ) == {names[0], names[1]}
    # E[x0 | do(x2=1)] = 0.4*0.7 + 0.3 = 0.58
    drawn = scm.do({names[0]: 1.0}).see(20_000, seed=0)[names[2]]
    assert abs(float(drawn.mean()) - 0.58) < 0.05


def test_fit_scm_wires_a_poisson_glm_node_end_to_end():
    # The four direct-fit tests in test_scm_fitters.py all call PoissonGLMFit().fit(...) directly,
    # so none of them exercise the wiring the brief's "Not auto-dispatched" section documents as
    # the real use case: fit_scm(..., families={"Y": PoissonGLMFit()}). That path is what actually
    # sets mechanism.invertible (fit.py), what the L3 guard in scm.py reads, what evaluate_holdout
    # dispatches on for holdout_score, and what do()/see() sample through -- all untested until now.
    graph = CausalGraph(directed_edges=[("X", "Y")])
    rng = np.random.default_rng(0)
    n = 4000
    x = rng.normal(size=n)
    y = rng.poisson(np.exp(0.2 + 0.5 * x))
    scm = fit_scm({"X": x, "Y": y}, graph=graph, families={"Y": PoissonGLMFit()})
    node = next(f for f in scm.fit_report.nodes if f.node == "Y")
    assert node.invertible is False
    assert node.family == "poisson_glm"  # pins fit.py's _FAMILY_NAMES entry, not the fallback
    with pytest.raises(NotIdentifiableError, match="counterfactual_interval"):
        scm.abduct(known={"Y": 2.0}, n=8)
    drawn = scm.do({"X": 1.0}).see(2000, seed=0)["Y"]
    assert torch.allclose(drawn, drawn.round())
