"""Tests for certify_estimate — the general off-policy-contrast decision certificate.

certify_estimate generalises certify_decision from raw one-hot arms to any V(pi_on) - V(pi_off)
contrast carried by a PolicyValueContrast. These pin the one-hot special case (equivalence to the
raw-logs path), the general stochastic-policy path, and the naive-contrast definition.
"""

from __future__ import annotations

import numpy as np
import pytest

from causalrl import (
    DecisionCertificate,
    certify_decision,
    certify_estimate,
    msm_contribution_bounds,
)
from causalrl.identification.estimate import PolicyValueContrast


def test_one_hot_contrast_matches_certify_decision_msm():
    rng = np.random.default_rng(0)
    n = 3000
    f = rng.integers(0, 2, size=n)
    y = 0.5 * f + rng.uniform(0, 0.2, size=n)
    e0 = np.full(n, 0.5)
    ref = certify_decision(y, f, propensities=e0, gamma_max=20.0)
    c = PolicyValueContrast.from_binary(y, f, propensities=e0)
    got = certify_estimate(c, gamma_max=20.0, labels=("treated", "control"))
    assert isinstance(got, DecisionCertificate)
    assert got.tipping_gamma == ref.tipping_gamma
    assert got.naive_contrast == pytest.approx(ref.naive_contrast)
    assert got.summary == ref.summary


def test_general_stochastic_policy_certifies_and_widens():
    rng = np.random.default_rng(1)
    n = 2000
    y = rng.uniform(0, 1, size=n).tolist()
    e0 = np.full(n, 0.5).tolist()
    on = rng.uniform(0.4, 0.6, size=n).tolist()  # overlapping (stochastic) targets
    off = rng.uniform(0.4, 0.6, size=n).tolist()
    c = PolicyValueContrast(outcomes=y, logging_propensities=e0, target_on=on, target_off=off)
    cert = certify_estimate(c, gamma_max=5.0)
    assert isinstance(cert, DecisionCertificate)
    assert cert.decision in {"prefer pi_on", "prefer pi_off", "indifferent"}
    assert cert.pivotality is None  # no binary-arm reduction => pivotality layer does not run


def test_general_path_naive_is_gamma1_point():
    y = [0.0, 1.0, 0.0, 1.0]
    e0 = [0.5, 0.5, 0.5, 0.5]
    on = [1.0, 1.0, 0.0, 0.0]
    off = [0.0, 0.0, 1.0, 1.0]
    c = PolicyValueContrast(outcomes=y, logging_propensities=e0, target_on=on, target_off=off)
    point = msm_contribution_bounds(y, e0, on, off, gamma=1.0).lower
    assert certify_estimate(c).naive_contrast == pytest.approx(point)


def test_labels_flow_into_decision():
    y = [1.0, 0.0]
    e0 = [0.5, 0.5]
    on = [1.0, 0.0]
    off = [0.0, 1.0]
    c = PolicyValueContrast(outcomes=y, logging_propensities=e0, target_on=on, target_off=off)
    cert = certify_estimate(c, labels=("new", "old"))
    assert cert.decision in {"prefer new", "prefer old", "indifferent"}
