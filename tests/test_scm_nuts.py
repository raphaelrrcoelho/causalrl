"""Plan §7.1 (deferred): NUTS / NumPyro posterior abduction.

Runs only on the dedicated ``[numpyro]`` CI lane (NumPyro + JAX are absent from the main test
matrix, so this whole module skips there). The correctness anchor is that NUTS recovers the exact
noise posterior of an invertible location-scale forward: its posterior mean matches the analytic
inversion.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("numpyro")
pytest.importorskip("jax")

from causalrl.certify.certificate import Kind
from causalrl.scm.continuous.nuts import (
    NUTSNoisePosterior,
    abduct_nuts,
    certify_nuts_counterfactual,
    nuts_posterior_predictive_check,
)

_B = 0.5


def _forward(pv: dict[str, np.ndarray], u: np.ndarray) -> np.ndarray:
    """Invertible location-scale forward (linear ⇒ works on both NumPy and jax.numpy arrays)."""
    return 2.0 * pv["X"] + _B * u + 1.0


def test_nuts_recovers_linear_gaussian_noise_posterior() -> None:
    rng = np.random.default_rng(0)
    n = 200
    x = rng.standard_normal(n)
    u_true = rng.standard_normal(n)
    y = _forward({"X": x}, u_true)  # near-deterministic outcome

    post = abduct_nuts(
        _forward, {"X": x}, y, noise_scale=0.05, num_warmup=300, num_samples=500, seed=0
    )
    assert isinstance(post, NUTSNoisePosterior)

    u_exact = (y - 2.0 * x - 1.0) / _B  # analytic inversion
    assert float(np.corrcoef(post.mean, u_exact)[0, 1]) > 0.99
    assert float(np.mean(np.abs(post.mean - u_exact))) < 0.2

    draws = post.sample(3)["U"]
    assert draws.shape == (3, n)

    ppc = nuts_posterior_predictive_check(_forward, {"X": x}, y, post.mean)
    assert ppc["ppc_rmse"] < 0.2  # posterior mean reconstructs the observed outcome


def test_nuts_counterfactual_certificate_is_empirical() -> None:
    cert = certify_nuts_counterfactual("cf via nuts", {"ppc_rmse": 0.05, "ppc_bias": 0.0})
    assert cert.kind is Kind.EMPIRICAL
    assert cert.method == "nuts"
    assert cert.assumptions[0].name == "abduction"
    assert cert.assumptions[0].params["method"] == "nuts"
