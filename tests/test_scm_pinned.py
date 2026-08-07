"""PinnedMechanism — deploying a known structural equation alongside learned ones."""

import numpy as np
import pytest
import torch

from causalrl.exceptions import NotIdentifiableError
from causalrl.scm.fit import fit_scm
from causalrl.scm.fitters import ANMFit, PinnedMechanism, TabularCPT
from causalrl.scm.graph import CausalGraph

Tensor = torch.Tensor
_GRAPH = CausalGraph(directed_edges=[("X", "Y")])


def _data(n: int = 400, *, slope: float = 3.0, seed: int = 0) -> dict[str, np.ndarray]:
    """``X ~ N(0, 1)``, ``Y = slope * X + noise`` — a mechanism a caller could know exactly."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    return {"X": x, "Y": slope * x + rng.normal(scale=0.1, size=n)}


def _times(factor: float):
    def mean(parents: dict[str, Tensor]) -> Tensor:
        return factor * parents["X"]

    return mean


def test_a_pinned_node_is_marked_and_the_model_is_mixed() -> None:
    model = fit_scm(_data(), graph=_GRAPH, families={"Y": PinnedMechanism(_times(3.0))})
    assert model.provenance == "mixed"
    report = model.fit_report
    assert report is not None
    assert report.pinned_nodes == ("Y",)
    assert {fit.node: fit.family for fit in report.nodes}["Y"] == "pinned"


def test_an_all_pinned_model_is_specified() -> None:
    def constant(parents: dict[str, Tensor]) -> Tensor:
        return torch.zeros(1)

    model = fit_scm(
        _data(),
        graph=_GRAPH,
        families={"X": PinnedMechanism(constant), "Y": PinnedMechanism(_times(3.0))},
    )
    # Nothing about the equations came from the data, so this is an assertion by its author in
    # exactly the way a hand-built SCM is.
    assert model.provenance == "specified"


def test_an_unpinned_model_is_still_fitted() -> None:
    assert fit_scm(_data(), graph=_GRAPH).provenance == "fitted"


def test_the_pinned_equation_is_deployed_exactly() -> None:
    model = fit_scm(_data(), graph=_GRAPH, families={"Y": PinnedMechanism(_times(3.0))})
    mechanism = model.mechanisms["Y"]
    x = torch.tensor([1.0, 2.0, -1.0])
    got = mechanism({"X": x}, torch.zeros(3))
    assert torch.allclose(got, 3.0 * x)


def test_a_correct_pinned_equation_scores_well_out_of_sample() -> None:
    model = fit_scm(_data(), graph=_GRAPH, families={"Y": PinnedMechanism(_times(3.0))})
    report = model.fit_report
    assert report is not None
    scores = {fit.node: fit.holdout_score for fit in report.nodes}
    assert scores["Y"] > 0.99


def test_a_wrong_pinned_equation_is_exposed_by_its_holdout_score() -> None:
    # The point of scoring a pinned node: the number tests the assertion. Nothing else in the
    # pipeline would notice that the supplied equation is wrong.
    model = fit_scm(_data(), graph=_GRAPH, families={"Y": PinnedMechanism(_times(-7.0))})
    report = model.fit_report
    assert report is not None
    scores = {fit.node: fit.holdout_score for fit in report.nodes}
    assert scores["Y"] < 0.0


def test_a_pinned_node_is_invertible_so_its_noise_is_recoverable() -> None:
    model = fit_scm(_data(), graph=_GRAPH, families={"Y": PinnedMechanism(_times(3.0))})
    report = model.fit_report
    assert report is not None
    assert {fit.node: fit.invertible for fit in report.nodes}["Y"]
    residual = model.mechanisms["Y"].residual  # type: ignore[attr-defined]
    x = torch.tensor([2.0])
    assert torch.allclose(residual({"X": x}, torch.tensor([6.5])), torch.tensor([0.5]))


def test_a_supplied_noise_distribution_is_used_verbatim() -> None:
    from torch.distributions import Normal

    noise = Normal(0.0, 2.5)
    model = fit_scm(
        _data(), graph=_GRAPH, families={"Y": PinnedMechanism(_times(3.0), noise=noise)}
    )
    assert model.exogenous["Y"] is noise


def test_a_constant_equation_broadcasts_to_the_column_it_stands_for() -> None:
    def constant(parents: dict[str, Tensor]) -> Tensor:
        return torch.tensor([2.0])

    model = fit_scm(_data(n=50), graph=_GRAPH, families={"X": PinnedMechanism(constant)})
    assert model.provenance == "mixed"


def test_a_mismatched_column_length_is_refused() -> None:
    def wrong(parents: dict[str, Tensor]) -> Tensor:
        return torch.zeros(3)

    with pytest.raises(ValueError, match="must map the parent columns"):
        fit_scm(_data(n=50), graph=_GRAPH, families={"Y": PinnedMechanism(wrong)})


def test_pinning_leaves_the_other_nodes_learned() -> None:
    model = fit_scm(_data(), graph=_GRAPH, families={"Y": PinnedMechanism(_times(3.0))})
    report = model.fit_report
    assert report is not None
    families = {fit.node: fit.family for fit in report.nodes}
    assert families["X"] == "anm"  # X was still fitted from the data
    assert families["Y"] == "pinned"


def test_the_report_summary_marks_pinned_nodes() -> None:
    model = fit_scm(_data(), graph=_GRAPH, families={"Y": PinnedMechanism(_times(3.0))})
    report = model.fit_report
    assert report is not None
    summary = report.summary()
    assert "PINNED" in summary
    assert summary.count("PINNED") == 1


def test_a_mixed_model_is_still_gated_for_point_counterfactuals() -> None:
    # Pinning one equation does not identify the noise-to-value coupling at a node that was
    # still learned, so L3 must stay refused.
    rng = np.random.default_rng(0)
    x = rng.integers(0, 2, size=300)
    data = {"X": x.astype(float), "Y": (2.0 * x + rng.normal(scale=0.1, size=300))}
    graph = CausalGraph(directed_edges=[("X", "Y")])
    model = fit_scm(
        data,
        graph=graph,
        families={"X": TabularCPT(), "Y": PinnedMechanism(_times(2.0))},
    )
    assert model.provenance == "mixed"
    assert model.non_invertible_nodes() == ["X"]
    with pytest.raises(NotIdentifiableError, match="mixed SCM"):
        model.abduct({"Y": 2.0})


def test_do_preserves_mixed_provenance() -> None:
    model = fit_scm(_data(), graph=_GRAPH, families={"Y": PinnedMechanism(_times(3.0))})
    assert model.do({"X": 1.0}).provenance == "mixed"


def test_a_pinned_model_samples_through_the_supplied_equation() -> None:
    model = fit_scm(_data(), graph=_GRAPH, families={"Y": PinnedMechanism(_times(3.0))})
    drawn = model.do({"X": 2.0}).see(200, seed=0)
    # Y = 3 * 2 + residual, and the residuals are the (small) fit residuals of the true DGP.
    assert float(np.mean(np.asarray(drawn["Y"]))) == pytest.approx(6.0, abs=0.1)


def test_pinning_composes_with_an_explicit_family_elsewhere() -> None:
    model = fit_scm(
        _data(),
        graph=_GRAPH,
        families={"X": ANMFit(), "Y": PinnedMechanism(_times(3.0))},
    )
    report = model.fit_report
    assert report is not None
    assert report.pinned_nodes == ("Y",)


def test_pinned_mechanism_is_reachable_from_the_scm_subpackage() -> None:
    # Every other fitter is re-exported from `causalrl.scm`; this one was not, so the documented
    # `from causalrl.scm import <Fitter>` pattern raised ImportError for exactly one name.
    from causalrl.scm import PinnedMechanism as FromSubpackage

    assert FromSubpackage is PinnedMechanism


def test_empty_fit_list_is_not_labelled_fitted() -> None:
    # `_provenance` counted pinned nodes and fell through to "fitted" when there were none --
    # including when there were no nodes at all, asserting a provenance the model does not have.
    from causalrl.scm.fit import _provenance

    assert _provenance([]) == "specified"
