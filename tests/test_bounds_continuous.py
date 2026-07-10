"""Phase 1 §7.3: estimated-propensity MSM (b) + heavy-tail/quantile targets (f).

All numpy/local. (b): the estimated-propensity bound reduces exactly to the shipped nominal kernel
and brackets the truth under bounded confounding. (f): the mean front door downgrades to a median on
an infinite-variance sample, and quantile CIs achieve nominal coverage on heavy tails.
"""

from __future__ import annotations

import numpy as np
import pytest

from causalrl.bounds.continuous import (
    certify_mean,
    certify_quantile,
    certify_sensitivity_bounds,
    moment_diagnostic,
    msm_sensitivity_bounds,
    tail_index_hill,
    weighted_quantile,
)
from causalrl.certify.certificate import Certificate, Kind
from causalrl.identification.bounds import Interval, ipw_sensitivity_bounds


def test_hill_detects_pareto_and_passes_gaussian() -> None:
    rng = np.random.default_rng(0)
    pareto = rng.pareto(1.5, 20000) + 1.0  # regularly-varying tail, index ~1.5 (< 2)
    gauss = rng.standard_normal(20000)
    assert 1.2 < tail_index_hill(pareto) < 2.0
    assert not moment_diagnostic(pareto).finite_variance
    assert moment_diagnostic(gauss).finite_variance


def test_weighted_quantile_matches_numpy_and_hand() -> None:
    rng = np.random.default_rng(1)
    v = rng.standard_normal(1000)
    assert weighted_quantile(v, 0.5) == pytest.approx(float(np.quantile(v, 0.5)))
    assert weighted_quantile([0.0, 10.0], 0.5, [1.0, 1.0]) == pytest.approx(5.0)


def test_msm_estimated_reduces_to_ipw_under_known_propensities() -> None:
    """Acceptance (b) reduction: propensities= path is exactly the shipped nominal kernel."""
    rng = np.random.default_rng(2)
    y = (rng.standard_normal(500) + 1.0).tolist()
    e = rng.uniform(0.2, 0.8, 500).tolist()
    assert msm_sensitivity_bounds(y, propensities=e, gamma=2.0) == ipw_sensitivity_bounds(
        y, e, gamma=2.0
    )


def test_msm_gamma_one_collapses_to_point() -> None:
    rng = np.random.default_rng(3)
    y = (rng.standard_normal(400) + 2.0).tolist()
    e = rng.uniform(0.3, 0.7, 400).tolist()
    iv = msm_sensitivity_bounds(y, propensities=e, gamma=1.0)
    assert iv.upper - iv.lower < 1e-9


def test_msm_estimated_path_returns_valid_interval() -> None:
    rng = np.random.default_rng(8)
    n = 3000
    z = rng.standard_normal(n)
    t = rng.binomial(1, 1.0 / (1.0 + np.exp(-z))).astype(float)
    y = 2.0 + 0.5 * z + rng.standard_normal(n)  # full per-unit outcomes; fn subsets treated
    narrow = msm_sensitivity_bounds(y.tolist(), treatment=t.tolist(), covariates=z, gamma=1.0)
    wide = msm_sensitivity_bounds(y.tolist(), treatment=t.tolist(), covariates=z, gamma=2.0)
    assert narrow.lower <= narrow.upper <= wide.upper
    assert wide.lower <= narrow.lower  # bounds widen monotonically with gamma


def test_msm_brackets_truth_under_bounded_confounding() -> None:
    """Acceptance (b) validity: with confounding odds-ratio <= gamma, the bound contains E[Y(1)]."""
    rng = np.random.default_rng(7)
    n = 50000
    u = rng.binomial(1, 0.5, n)  # unmeasured confounder; nominal e0 = 0.5 ignores it
    e_true = np.where(u == 1, 3.0 / 4.0, 1.0 / 4.0)  # true odds-ratio vs nominal = 3
    t = rng.binomial(1, e_true)
    y1 = u.astype(float) + 0.1 * rng.standard_normal(n)  # E[Y(1)] = 0.5
    treated = t == 1
    e0 = [0.5] * int(treated.sum())
    iv = msm_sensitivity_bounds(y1[treated].tolist(), propensities=e0, gamma=3.5)
    assert iv.lower <= 0.5 <= iv.upper
    naive = msm_sensitivity_bounds(y1[treated].tolist(), propensities=e0, gamma=1.0)
    assert naive.lower > 0.5  # confounding really does bias the naive IPW point high


def test_certify_sensitivity_bounds_certificate() -> None:
    rng = np.random.default_rng(5)
    y = (rng.standard_normal(300) + 1.0).tolist()
    e = rng.uniform(0.2, 0.8, 300).tolist()
    cert = certify_sensitivity_bounds(y, propensities=e, gamma=2.0)
    assert cert.kind is Kind.BOUNDED
    assert isinstance(cert.value, Interval)
    names = {a.name for a in cert.assumptions}
    assert "MSM" in names and "known-propensity" in names
    assert Certificate.from_json(cert.to_json()).to_dict() == cert.to_dict()


def test_certify_mean_downgrades_on_heavy_tail() -> None:
    """Acceptance (f): mean request on infinite-variance data downgrades to a median (I3)."""
    rng = np.random.default_rng(6)
    pareto = (rng.pareto(1.3, 8000) + 1.0).tolist()
    cert = certify_mean(pareto, seed=0)
    assert cert.hedge is not None and cert.hedge.downgraded_from == "mean"
    assert cert.estimand.target == "quantile"

    gauss = rng.standard_normal(8000).tolist()
    keep = certify_mean(gauss, seed=0)
    assert keep.hedge is None
    assert keep.estimand.target == "mean" and keep.kind is Kind.IDENTIFIED


def test_certify_quantile_median_coverage() -> None:
    """Acceptance (f): bootstrap median CIs cover the true median at nominal rate on heavy tails."""
    a = 1.5
    true_median = 2.0 ** (1.0 / a) - 1.0  # median of numpy's pareto(a) (Lomax)
    rng = np.random.default_rng(20260711)
    reps, n, hits = 400, 500, 0
    for _ in range(reps):
        sample = rng.pareto(a, n).tolist()
        cert = certify_quantile(sample, 0.5, alpha=0.05, n_boot=500, seed=0)
        assert cert.ci is not None
        if cert.ci.lower <= true_median <= cert.ci.upper:
            hits += 1
    coverage = hits / reps
    assert 0.90 <= coverage <= 0.99, f"median coverage {coverage:.3f} off nominal"


def test_bounds_package_reexports() -> None:
    from causalrl.bounds import (
        Interval as I2,
    )
    from causalrl.bounds import (
        ipw_sensitivity_bounds as ipw2,
    )
    from causalrl.bounds import (
        msm_sensitivity_bounds as msm2,
    )

    assert callable(ipw2) and callable(msm2)
    assert I2(0.0, 1.0).upper == 1.0
