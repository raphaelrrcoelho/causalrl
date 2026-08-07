"""The fitted model must say when it does not describe the regime being queried."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from causalrl import CausalGraph, Kind, fit_scm
from causalrl.scm.fidelity import certify_fitted_query
from causalrl.scm.fitters import PinnedMechanism


def _data(n: int = 3000, seed: int = 0) -> dict[str, np.ndarray]:
    """``Y = 2 + 3.5 * S + noise``: under S=1 the outcome sits near 5.5, under S=0 near 2."""
    rng = np.random.default_rng(seed)
    s = rng.integers(0, 2, size=n).astype(np.float64)
    y = 2.0 + 3.5 * s + rng.normal(scale=0.3, size=n)
    return {"S": s, "Y": y}


def _flat_at(value: float) -> PinnedMechanism:
    """A mechanism that ignores its parents entirely -- wrong here, and wrong in a visible way."""

    def mean(parents: dict[str, torch.Tensor]) -> torch.Tensor:
        n = len(next(iter(parents.values())))
        return torch.full((n,), value, dtype=torch.float64)

    return PinnedMechanism(mean)


def _graph() -> CausalGraph:
    return CausalGraph(directed_edges=[("S", "Y")], nodes=["S", "Y"])


def test_a_well_fitted_model_passes_the_gate() -> None:
    data = _data()
    model = fit_scm(data, graph=_graph())

    cert = certify_fitted_query(model, data, intervention={"S": 1.0}, outcome="Y")

    assert cert.hedge is None
    assert cert.kind is Kind.IDENTIFIED
    diagnostic = cert.assumptions[0].diagnostic
    assert diagnostic is not None and diagnostic["support"] > 0


def test_a_mechanism_that_mispredicts_the_regime_is_hedged_and_downgraded() -> None:
    """The motivating failure: the model answers 1.77 where the data plainly say 5.50."""
    data = _data()
    # Pin a wrong mechanism: it ignores S entirely, so under S=1 it predicts ~1.8 not ~5.5.
    wrong = _flat_at(1.77)
    model = fit_scm(data, graph=_graph(), families={"Y": wrong})

    cert = certify_fitted_query(model, data, intervention={"S": 1.0}, outcome="Y")

    assert cert.hedge is not None
    assert "model-fidelity" in cert.hedge.reason
    assert cert.hedge.downgraded_from == "fitted"
    assert cert.kind is Kind.EMPIRICAL
    assert cert.hedge.detail is not None
    # The data under S=1 sit near 5.5; the pinned equation cannot get there. (It does not predict
    # a flat 1.77 either: PinnedMechanism with noise=None resamples the residuals Y - g(parents),
    # which are large precisely because the equation is wrong -- so the gap, not the constant, is
    # the thing to assert.)
    assert abs(cert.hedge.detail["observed_mean"] - 5.5) < 0.3
    assert cert.hedge.detail["predicted_mean"] < 4.5
    diagnostic = cert.assumptions[0].diagnostic
    assert diagnostic is not None and diagnostic["standardised_error"] > 2.0


def test_a_regime_with_no_factual_support_is_flagged_as_extrapolation() -> None:
    """Querying where the logs contain nothing is untested, not merely uncertain."""
    data = _data()
    model = fit_scm(data, graph=_graph())

    cert = certify_fitted_query(model, data, intervention={"S": 7.0}, outcome="Y")

    assert cert.hedge is not None
    assert "no-factual-support" in cert.hedge.reason
    assert cert.kind is Kind.EMPIRICAL


def test_the_answer_is_always_reported_even_when_gated() -> None:
    """Gating changes what the answer is claimed to be, not whether it is available."""
    data = _data()
    wrong = _flat_at(1.77)
    model = fit_scm(data, graph=_graph(), families={"Y": wrong})

    cert = certify_fitted_query(model, data, intervention={"S": 1.0}, outcome="Y")

    assert cert.hedge is not None  # gated ...
    assert "E[Y | do(S=1.0)] =" in cert.claim  # ... and still answered
    assert "FAILS" in cert.claim


def test_the_tolerance_is_the_knob_and_it_is_in_observed_standard_deviations() -> None:
    data = _data()
    wrong = _flat_at(1.77)
    model = fit_scm(data, graph=_graph(), families={"Y": wrong})

    strict = certify_fitted_query(model, data, intervention={"S": 1.0}, outcome="Y", tolerance=0.5)
    lax = certify_fitted_query(model, data, intervention={"S": 1.0}, outcome="Y", tolerance=1e6)

    assert strict.hedge is not None
    assert lax.hedge is None


def test_it_refuses_an_empty_intervention_and_unknown_columns() -> None:
    data = _data(n=200)
    model = fit_scm(data, graph=_graph())
    with pytest.raises(ValueError, match="at least one variable"):
        certify_fitted_query(model, data, intervention={}, outcome="Y")
    with pytest.raises(KeyError):
        certify_fitted_query(model, data, intervention={"S": 1.0}, outcome="Nope")
