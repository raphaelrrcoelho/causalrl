# tests/test_bayesian_fit.py
"""Posterior mechanism fitting. Runs only on the dedicated ``[numpyro]`` CI lane."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("numpyro")
pytest.importorskip("jax")

from causalrl.scm.continuous.bayesian_fit import BayesianLinearFit
from causalrl.scm.fit import fit_scm
from causalrl.scm.graph import CausalGraph


def test_posterior_mean_recovers_the_generating_coefficients():
    # X on a native scale far from 1 (mean 10, sd 5): standardisation-then-unstandardisation must
    # round-trip correctly here, or the recovered coefficients drift off 2.0 / 1.0. Regression
    # coverage for both the slope's `/ safe_scale_x[i]` and the intercept's mean-shift correction.
    # mean_x=10 (not larger): the slope's posterior spread (theoretical sd ~= 0.5 / (scale_x *
    # sqrt(n)) ~= 0.0022) amplifies onto the intercept by a factor of mean_x, so keeping mean_x
    # modest keeps that amplified sd (~0.022 here) far inside the 0.1 tolerance below (~4.5 sd of
    # headroom) instead of trading test power for tolerance-widening.
    rng = np.random.default_rng(0)
    n = 2000
    x = 10.0 + 5.0 * rng.normal(size=n)
    y = 2.0 * x + 1.0 + rng.normal(scale=0.5, size=n)
    fitted = BayesianLinearFit(draws=500, warmup=500, seed=0).fit({"X": x}, y)
    posterior = fitted.mechanism.posterior  # type: ignore[attr-defined]
    assert isinstance(posterior["X"], np.ndarray)
    assert abs(float(np.mean(posterior["X"])) - 2.0) < 0.1
    assert abs(float(np.mean(posterior["intercept"])) - 1.0) < 0.1
    assert fitted.invertible is True
    # In-sample R^2 of the posterior-mean mechanism -- the same field, on the same scale, that
    # LinearGaussianFit reports and that fit_scm falls back to when the holdout split is empty.
    # signal var (5 * 2)^2 = 100 against residual var 0.25, so R^2 ~= 0.9975.
    assert fitted.score > 0.9


def test_posterior_sigma_recovers_the_true_noise_scale_at_large_child_scale():
    # y's native scale is ~200 residual sd here, far from the HalfNormal(1) prior's unit scale.
    # Regression coverage for sigma's `* safe_scale_y` unstandardisation: without it, sigma's
    # posterior collapses toward the prior's scale instead of tracking the true residual sd.
    rng = np.random.default_rng(3)
    n = 2000
    true_sigma = 200.0
    x = rng.normal(size=n)
    y = 50.0 * x + 300.0 + rng.normal(scale=true_sigma, size=n)
    fitted = BayesianLinearFit(draws=500, warmup=500, seed=0).fit({"X": x}, y)
    posterior = fitted.mechanism.posterior  # type: ignore[attr-defined]
    assert abs(float(np.mean(posterior["sigma"])) - true_sigma) < 0.1 * true_sigma
    # ...and the same number must reach `fitted.noise`, which is the field fit_scm installs as this
    # node's exogenous distribution (fit.py) and which every see()/do() draw is sampled from. The
    # posterior dict is only reporting; the noise is deployed. Dropping the `* safe_scale_y`
    # unstandardisation from the noise alone would leave the assertion above green while the SCM
    # drew residuals ~200x too small here, so this is the assertion that pins the standardise /
    # unstandardise hazard on the side that ships. Same shape as the point-estimate sibling's
    # `abs(float(fitted.noise.stddev) - 0.5) < 0.05` in test_scm_fitters.py.
    assert abs(float(fitted.noise.stddev) - true_sigma) < 0.1 * true_sigma


def test_credible_interval_covers_the_truth_and_narrows_with_data():
    def width(n: int, seed: int) -> float:
        # Own rng per call (not a shared stream advanced across both invocations): keeps this
        # helper's data independent of call order, so a later refactor that reorders the two
        # width() calls -- or adds an unrelated call -- cannot silently re-roll the 95% coverage
        # assertion below at both sizes simultaneously.
        rng = np.random.default_rng(seed)
        x = rng.normal(size=n)
        y = 2.0 * x + rng.normal(scale=0.5, size=n)
        draws = BayesianLinearFit(draws=500, warmup=500, seed=0).fit({"X": x}, y)
        posterior = draws.mechanism.posterior["X"]  # type: ignore[attr-defined]
        lo, hi = np.percentile(posterior, [2.5, 97.5])
        assert lo <= 2.0 <= hi, (lo, hi)
        return float(hi - lo)

    assert width(2000, seed=1) < width(200, seed=2)


def test_fit_rejects_a_parent_name_that_collides_with_a_reserved_posterior_key():
    # "sigma" (and "intercept") are the fitted mechanism's own posterior keys; a parent sharing
    # either name would silently overwrite it in the `posterior` dict. Fails before any NUTS
    # sampling, so small draws/warmup here do not shrink any real assertion.
    rng = np.random.default_rng(4)
    x = rng.normal(size=50)
    y = 2.0 * x + rng.normal(scale=0.1, size=50)
    with pytest.raises(ValueError, match="sigma"):
        BayesianLinearFit(draws=10, warmup=10, seed=0).fit({"sigma": x}, y)


def test_posterior_mean_mechanism_round_trips_like_its_siblings():
    import torch

    rng = np.random.default_rng(2)
    n = 1500
    x = rng.normal(size=n)
    y = 1.5 * x - 0.5 + rng.normal(scale=0.3, size=n)
    fitted = BayesianLinearFit(draws=400, warmup=400, seed=0).fit({"X": x}, y)
    parents = {"X": torch.tensor([0.4], dtype=torch.float32)}
    value = torch.tensor([1.1], dtype=torch.float32)
    noise = fitted.mechanism.residual(parents, value)  # type: ignore[attr-defined]
    assert torch.allclose(fitted.mechanism(parents, noise), value, rtol=0.0, atol=1e-4)


def test_fit_scm_wires_a_bayesian_linear_node_end_to_end():
    # The direct-fit tests above all call BayesianLinearFit().fit(...) directly, so none of them
    # exercise the wiring that is the real use case: fit_scm(..., families={"Y":
    # BayesianLinearFit()}). That path is what sets mechanism.invertible (fit.py), what the L3
    # guard in scm.py reads, what evaluate_holdout dispatches on for holdout_score, what
    # _FAMILY_NAMES labels in the FitReport, and -- above all -- what installs `fitted.noise` as
    # the node's exogenous distribution that do()/see() sample from. The same hole PoissonGLMFit
    # closed in test_scm_fit.py::test_fit_scm_wires_a_poisson_glm_node_end_to_end.
    #
    # y's residual scale (5.0) is deliberately far from y's own scale (sd ~= 30.4): the deployed
    # noise is unstandardised by safe_scale_y, so a regression there would show up as rollout
    # spread ~6x too small while every mean below stayed correct.
    graph = CausalGraph(directed_edges=[("X", "Y")])
    rng = np.random.default_rng(0)
    n = 2500
    x = rng.normal(size=n)
    y = 30.0 * x + 20.0 + rng.normal(scale=5.0, size=n)
    scm = fit_scm(
        {"X": x, "Y": y}, graph=graph, families={"Y": BayesianLinearFit(draws=400, warmup=400)}
    )

    node = next(f for f in scm.fit_report.nodes if f.node == "Y")
    assert node.family == "bayesian_linear"  # pins fit.py's _FAMILY_NAMES entry, not the fallback
    assert node.invertible is True
    # evaluate_holdout's invertible branch: R^2 on rows the fit never saw, 900 / 925 ~= 0.973.
    assert node.holdout_score > 0.9

    # The L3 guard's permissive side: an all-invertible fitted SCM licenses point counterfactuals
    # (the mirror of the Poisson node's NotIdentifiableError).
    scm.abduct(known={"Y": 0.0}, n=8)

    drawn = scm.do({"X": 1.0}).see(8000, seed=0)["Y"]
    assert abs(float(drawn.mean()) - 50.0) < 1.0  # 30 * 1 + 20, from the posterior-mean mechanism
    assert abs(float(drawn.std()) - 5.0) < 1.0  # from fitted.noise -- the deployed exogenous draw
