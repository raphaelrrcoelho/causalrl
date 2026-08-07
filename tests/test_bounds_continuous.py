"""Phase 1 §7.3: estimated-propensity MSM.

All numpy/local. The estimated-propensity bound reduces exactly to the shipped nominal kernel and
brackets the truth under bounded confounding.
"""

from __future__ import annotations

import numpy as np

from causalrl.bounds.continuous import (
    certify_sensitivity_bounds,
    msm_sensitivity_bounds,
)
from causalrl.certify.certificate import Certificate, Kind
from causalrl.identification.bounds import Interval
from causalrl.ope.bounds import ipw_sensitivity_bounds


def test_msm_estimated_reduces_to_ipw_under_known_propensities() -> None:
    """Acceptance (b) reduction: propensities= path is exactly the shipped nominal kernel."""
    rng = np.random.default_rng(2)
    y = (rng.standard_normal(500) + 1.0).tolist()
    e = rng.uniform(0.2, 0.8, 500).tolist()
    assert msm_sensitivity_bounds(y, propensities=e, gamma=2.0) == ipw_sensitivity_bounds(
        y, e, gamma=2.0, return_certificate=False
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
