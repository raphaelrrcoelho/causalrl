# tests/test_bayesian_fit.py
"""Posterior mechanism fitting. Runs only on the dedicated ``[numpyro]`` CI lane."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("numpyro")
pytest.importorskip("jax")

from causalrl.scm.continuous.bayesian_fit import BayesianLinearFit


def test_posterior_mean_recovers_the_generating_coefficients():
    # X on a native scale far from 1 (mean 100, sd 5): standardisation-then-unstandardisation must
    # round-trip correctly here, or the recovered coefficients drift off 2.0 / 1.0. Regression
    # coverage for both the slope's `/ safe_scale_x[i]` and the intercept's mean-shift correction.
    rng = np.random.default_rng(0)
    n = 2000
    x = 100.0 + 5.0 * rng.normal(size=n)
    y = 2.0 * x + 1.0 + rng.normal(scale=0.5, size=n)
    fitted = BayesianLinearFit(draws=500, warmup=500, seed=0).fit({"X": x}, y)
    posterior = fitted.mechanism.posterior  # type: ignore[attr-defined]
    assert isinstance(posterior["X"], np.ndarray)
    assert abs(float(np.mean(posterior["X"])) - 2.0) < 0.1
    # A wider tolerance than the slope's: at mean_x=100, intercept = ... - weight * mean_x, so the
    # slope's own posterior spread (theoretical sd ~= 0.5 / (scale_x * sqrt(n)) ~= 0.0022) is
    # amplified ~100x onto the intercept (theoretical sd ~= mean_x * that ~= 0.22). 0.3 stays far
    # tighter than either mutation's failure mode (Mutation A fails the slope assertion above
    # first, at ~10 vs true 2.0; Mutation B lands the intercept at ~201 vs true 1.0).
    assert abs(float(np.mean(posterior["intercept"])) - 1.0) < 0.3
    assert fitted.invertible is True


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
