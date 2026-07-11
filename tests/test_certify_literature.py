"""Known-answer anchors: the certificate stack reproduces the published Tan / Zhao marginal-
sensitivity-model Hajek bounds. Formula-level checks (no code ported), so the sensitivity numbers
the decision certificate reports are provably trustworthy, not just internally consistent.
"""

from __future__ import annotations

import pytest

from causalrl import certify_estimate, ipw_sensitivity_bounds
from causalrl.identification.estimate import PolicyValueContrast


def test_ipw_msm_collapses_to_ipw_point_at_gamma_one():
    # Tan's MSM collapses to the IPW point estimate at Gamma = 1.
    iv = ipw_sensitivity_bounds([0.0, 1.0], [0.5, 0.5], gamma=1.0, return_certificate=False)
    assert iv.lower == pytest.approx(0.5) and iv.upper == pytest.approx(0.5)


def test_ipw_msm_hand_computed_band_at_gamma_two():
    # e0 = 0.5 => odds = 1; per-unit weights range over [1 + 1/2, 1 + 2] = [1.5, 3].
    # Hajek min: hi weight on y=0, lo on y=1 -> 1.5/4.5 = 1/3; max -> 3/4.5 = 2/3.
    iv = ipw_sensitivity_bounds([0.0, 1.0], [0.5, 0.5], gamma=2.0, return_certificate=False)
    assert iv.lower == pytest.approx(1 / 3) and iv.upper == pytest.approx(2 / 3)


def test_certify_estimate_naive_matches_ips_contrast():
    # Two disjoint one-hot arms over uniform logging: each arm's Gamma=1 self-normalised mean is
    # 0.5, so the contrast is exactly 0.0.
    y = [0.0, 1.0, 1.0, 0.0]
    e0 = [0.5, 0.5, 0.5, 0.5]
    on = [1.0, 1.0, 0.0, 0.0]
    off = [0.0, 0.0, 1.0, 1.0]
    c = PolicyValueContrast(outcomes=y, logging_propensities=e0, target_on=on, target_off=off)
    assert certify_estimate(c).naive_contrast == pytest.approx(0.0)
