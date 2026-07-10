"""Phase 1 §7.4: conformal prediction intervals with finite-sample >= nominal coverage (accept c).

All numpy/local. Split conformal, CQR (heteroscedastic), and weighted conformal (covariate shift)
each attain at least the nominal marginal coverage; the certificate wrapper records exchangeability.
"""

from __future__ import annotations

import numpy as np
import pytest

from causalrl.certify.certificate import Certificate, Kind
from causalrl.conformal.core import (
    certify_conformal_interval,
    conformal_quantile,
    cqr_interval,
    split_conformal_interval,
)
from causalrl.identification.bounds import Interval


def test_conformal_quantile_finite_sample_formula() -> None:
    # n=9, alpha=0.1: ceil((n+1)(1-alpha)) = ceil(9.0) = 9 -> the 9th smallest score (index 8).
    scores = list(range(9))
    assert conformal_quantile(scores, 0.1) == 8.0
    # too small for the level -> +inf (a trivially valid interval)
    assert conformal_quantile(scores, 0.05) == float("inf")


def test_conformal_quantile_validation() -> None:
    with pytest.raises(ValueError, match="at least one"):
        conformal_quantile([], 0.1)
    with pytest.raises(ValueError, match="alpha"):
        conformal_quantile([1.0, 2.0], 1.5)


def test_split_conformal_coverage_is_at_least_nominal() -> None:
    """Acceptance (c): split conformal covers a fresh point at >= 1 - alpha (marginal)."""
    rng = np.random.default_rng(2026)
    alpha, reps, n_cal, hits = 0.1, 400, 100, 0
    for _ in range(reps):
        cal_noise = rng.standard_normal(n_cal)  # residuals of a correct predictor
        interval = split_conformal_interval(0.0, cal_noise.tolist(), [0.0] * n_cal, alpha)
        y_test = float(rng.standard_normal())
        if interval.lower <= y_test <= interval.upper:
            hits += 1
    coverage = hits / reps
    assert 0.88 <= coverage <= 0.97, f"split coverage {coverage:.3f}"


def test_cqr_coverage_heteroscedastic() -> None:
    """Acceptance (c): CQR keeps >= nominal coverage under heteroscedastic noise + a crude band."""
    rng = np.random.default_rng(7)
    n_cal, n_test, alpha = 1000, 4000, 0.1
    x_cal = rng.standard_normal(n_cal)
    scale_cal = 0.5 + np.abs(x_cal)
    y_cal = scale_cal * rng.standard_normal(n_cal)
    band = 1.2816  # nominal 10/90 z, applied WITHOUT the scale (deliberately miscalibrated)
    interval_lo = cqr_interval(-band, band, y_cal.tolist(), [-band] * n_cal, [band] * n_cal, alpha)
    x_test = rng.standard_normal(n_test)
    y_test = (0.5 + np.abs(x_test)) * rng.standard_normal(n_test)
    coverage = float(np.mean((y_test >= interval_lo.lower) & (y_test <= interval_lo.upper)))
    assert coverage >= 0.88, f"CQR coverage {coverage:.3f}"


def test_weighted_conformal_corrects_covariate_shift() -> None:
    """Acceptance (c) weighted: under covariate shift the unweighted interval under-covers; the
    likelihood-ratio-weighted one restores coverage."""
    rng = np.random.default_rng(11)
    n_cal, n_test, alpha = 600, 3000, 0.1
    x_cal = rng.standard_normal(n_cal)  # calibration X ~ N(0, 1)
    resid_cal = (0.3 + 0.4 * np.abs(x_cal)) * np.abs(rng.standard_normal(n_cal))
    w_cal = np.exp(1.5 * x_cal - 1.5**2 / 2.0)  # dN(1.5,1)/dN(0,1)

    x_test = 1.5 + rng.standard_normal(n_test)  # test X ~ N(1.5, 1): larger |X| -> larger residuals
    resid_test = (0.3 + 0.4 * np.abs(x_test)) * np.abs(rng.standard_normal(n_test))

    q_unw = conformal_quantile(resid_cal.tolist(), alpha)
    unweighted = float(np.mean(resid_test <= q_unw))

    weighted_hits = 0
    for xt, rt in zip(x_test, resid_test, strict=True):
        wt = float(np.exp(1.5 * xt - 1.5**2 / 2.0))
        q = conformal_quantile(resid_cal.tolist(), alpha, weights=w_cal.tolist(), test_weight=wt)
        if rt <= q:
            weighted_hits += 1
    weighted = weighted_hits / n_test

    assert unweighted < 0.90  # ignoring the shift under-covers
    assert weighted >= 0.88  # weighting restores >= nominal
    assert weighted > unweighted + 0.02


def test_certify_conformal_interval_certificate() -> None:
    rng = np.random.default_rng(3)
    cal = rng.standard_normal(200)
    cert = certify_conformal_interval(1.0, cal.tolist(), [0.0] * 200, alpha=0.1)
    assert cert.kind is Kind.EMPIRICAL
    assert cert.value is None
    assert isinstance(cert.ci, Interval) and cert.ci.lower < 1.0 < cert.ci.upper
    assert cert.alpha == 0.1
    assert cert.assumptions[0].name == "exchangeability"
    assert Certificate.from_json(cert.to_json()).to_dict() == cert.to_dict()


def test_certify_conformal_marks_weighted_exchangeability() -> None:
    rng = np.random.default_rng(4)
    cal = rng.standard_normal(200)
    w = np.ones(200)
    cert = certify_conformal_interval(0.0, cal.tolist(), [0.0] * 200, alpha=0.1, weights=w.tolist())
    assert cert.assumptions[0].name == "weighted-exchangeability"
    assert cert.method == "weighted-split-conformal"
