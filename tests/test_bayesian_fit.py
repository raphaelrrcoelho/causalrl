# tests/test_bayesian_fit.py
"""Posterior mechanism fitting. Runs only on the dedicated ``[numpyro]`` CI lane."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("numpyro")
pytest.importorskip("jax")

from causalrl.scm.continuous.bayesian_fit import BayesianLinearFit


def test_posterior_mean_recovers_the_generating_coefficients():
    rng = np.random.default_rng(0)
    n = 2000
    x = rng.normal(size=n)
    y = 2.0 * x + 1.0 + rng.normal(scale=0.5, size=n)
    fitted = BayesianLinearFit(draws=500, warmup=500, seed=0).fit({"X": x}, y)
    posterior = fitted.mechanism.posterior  # type: ignore[attr-defined]
    assert abs(float(np.mean(posterior["X"])) - 2.0) < 0.1
    assert abs(float(np.mean(posterior["intercept"])) - 1.0) < 0.1
    assert fitted.invertible is True


def test_credible_interval_covers_the_truth_and_narrows_with_data():
    rng = np.random.default_rng(1)

    def width(n: int) -> float:
        x = rng.normal(size=n)
        y = 2.0 * x + rng.normal(scale=0.5, size=n)
        draws = BayesianLinearFit(draws=500, warmup=500, seed=0).fit({"X": x}, y)
        posterior = draws.mechanism.posterior["X"]  # type: ignore[attr-defined]
        lo, hi = np.percentile(posterior, [2.5, 97.5])
        assert lo <= 2.0 <= hi, (lo, hi)
        return float(hi - lo)

    assert width(2000) < width(200)


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
