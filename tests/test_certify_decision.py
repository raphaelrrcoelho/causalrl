"""Tests for certify_decision — the one-call decision-certificate front door (v1.0).

certify_decision is an ergonomic orchestrator over the documented decision stack: it composes
pivotality_certificate (cheap sign-robustness under hidden confounding) and, when logging
propensities are supplied, the marginal-sensitivity-model tipping point (tipping_gamma over
msm_contribution_bounds). It adds no new theory; these tests pin the orchestration and the
human-readable verdict, not the underlying kernels (those are covered in test_pivotality.py /
test_msm_bounds.py).
"""

from __future__ import annotations

import numpy as np
import pytest

from causalrl import DecisionCertificate, certify_decision


def _confounded_rows(n: int, strength: float, seed: int = 0):
    """Binary Z -> (F, Y) with confounding `strength` (0 => F independent of Z)."""
    rng = np.random.default_rng(seed)
    z = rng.integers(0, 2, size=n)
    p_f = 0.3 + strength * (z - 0.5)  # P(F=1|Z)
    f = (rng.random(n) < p_f).astype(int)
    y = 1.0 * z + 0.5 * f + rng.normal(0, 0.1, size=n)  # Z confounds Y; real +0.5 treated effect
    return y, f, z


class TestStructuralLayer:
    def test_certifies_thin_channel_decision(self):
        # weak confounding, real +0.5 treated effect: the decision must certify
        y, f, z = _confounded_rows(20000, 0.0, seed=1)
        cert = certify_decision(y, f, confounder_bins=z)
        assert isinstance(cert, DecisionCertificate)
        assert cert.decision == "prefer treated"
        assert cert.certified
        assert cert.pivotality is not None and cert.pivotality.certified
        assert cert.tipping_gamma is None and cert.msm_certified is None  # MSM layer not run

    def test_does_not_certify_when_confounding_can_flip(self):
        # small NEGATIVE effect buried under strong positive Z bias: naive>0 but adjusted<0
        rng = np.random.default_rng(1)
        n = 20000
        z = rng.integers(0, 2, size=n)
        f = (rng.random(n) < 0.3 + 0.4 * (z - 0.5)).astype(int)
        y = 1.0 * z - 0.1 * f + rng.normal(0, 0.1, size=n)
        cert = certify_decision(y, f, confounder_bins=z)
        assert cert.decision == "prefer treated"
        assert not cert.certified

    def test_structural_cap_mode(self):
        # certify from the information-channel cap alone, no measured Z
        y, f, _ = _confounded_rows(20000, 0.0, seed=2)
        assert certify_decision(y, f, mi_cap=1e-6).certified
        assert not certify_decision(y, f, mi_cap=10.0).certified


class TestMsmLayer:
    def test_reports_tipping_gamma_with_propensities(self):
        # random logging (uniform propensities) => the IPS contrast equals the raw contrast
        rng = np.random.default_rng(4)
        n = 4000
        f = rng.integers(0, 2, size=n)
        y = 0.5 * f + rng.uniform(0, 0.2, size=n)  # treated genuinely better
        e0 = np.full(n, 0.5)
        cert = certify_decision(y, f, propensities=e0, gamma_max=20.0)
        assert cert.decision == "prefer treated"
        assert cert.msm_certified is not None  # MSM layer ran
        assert cert.tipping_gamma is None or cert.tipping_gamma >= 1.0
        assert cert.pivotality is None  # structural layer not run

    def test_strong_decision_more_robust_than_borderline(self):
        rng = np.random.default_rng(5)
        n = 4000
        f = rng.integers(0, 2, size=n)
        e0 = np.full(n, 0.5)
        strong = certify_decision(1.0 * f, f, propensities=e0, gamma_max=20.0)  # clean separation
        y_border = 0.02 * f + rng.uniform(0, 1, size=n)  # near-indifferent
        border = certify_decision(y_border, f, propensities=e0, gamma_max=20.0)
        strong_g = float("inf") if strong.tipping_gamma is None else strong.tipping_gamma
        border_g = float("inf") if border.tipping_gamma is None else border.tipping_gamma
        assert strong_g > border_g  # a cleaner decision tips later (or never)


class TestOrchestration:
    def test_both_layers_run_when_both_evidence_given(self):
        y, f, z = _confounded_rows(8000, 0.0, seed=6)
        e0 = np.full(len(f), 0.5)
        cert = certify_decision(y, f, confounder_bins=z, propensities=e0)
        assert cert.pivotality is not None  # structural layer ran
        assert cert.msm_certified is not None  # MSM layer ran

    def test_decision_string_follows_sign(self):
        # control genuinely better => "prefer control"
        rng = np.random.default_rng(7)
        n = 6000
        z = rng.integers(0, 2, size=n)
        f = rng.integers(0, 2, size=n)
        y = 0.3 * z - 0.5 * f + rng.normal(0, 0.1, size=n)  # treated is worse
        cert = certify_decision(y, f, confounder_bins=z)
        assert cert.naive_contrast < 0
        assert cert.decision == "prefer control"

    def test_summary_is_human_readable(self):
        y, f, z = _confounded_rows(8000, 0.0, seed=8)
        cert = certify_decision(y, f, confounder_bins=z)
        summary = cert.summary
        assert isinstance(summary, str) and summary
        assert "prefer treated" in summary
        assert str(cert) == summary

    def test_str_returns_summary(self):
        y, f, z = _confounded_rows(100, 0.0)
        cert = certify_decision(y, f, confounder_bins=z)
        assert str(cert) == cert.summary

    def test_requires_some_evidence(self):
        y, f, _ = _confounded_rows(200, 0.0)
        with pytest.raises(ValueError, match=r"confounder_bins|mi_cap|propensities"):
            certify_decision(y, f)

    def test_requires_both_arms(self):
        y = np.array([0.1, 0.2, 0.3])
        f = np.array([1, 1, 1])  # no control arm
        with pytest.raises(ValueError, match="both arms"):
            certify_decision(y, f, mi_cap=0.1)


class TestEstimateDelegation:
    def test_certify_decision_equals_certify_estimate_over_random_inputs(self):
        from causalrl import certify_estimate
        from causalrl.identification.estimate import PolicyValueContrast

        rng = np.random.default_rng(11)
        for _ in range(5):
            n = 500
            f = rng.integers(0, 2, size=n)
            if not (f.any() and (~f).any()):
                continue
            y = 0.3 * f + rng.uniform(0, 1, size=n)
            e0 = np.full(n, 0.5)
            a = certify_decision(y, f, propensities=e0, gamma_max=15.0)
            b = certify_estimate(
                PolicyValueContrast.from_binary(y, f, propensities=e0),
                gamma_max=15.0,
                labels=("treated", "control"),
            )
            assert a == b

    def test_estimate_overload_matches_certify_estimate(self):
        from causalrl import certify_estimate
        from causalrl.identification.estimate import PolicyValueContrast

        y = [0.0, 1.0, 0.0, 1.0]
        e0 = [0.5, 0.5, 0.5, 0.5]
        on = [1.0, 1.0, 0.0, 0.0]
        off = [0.0, 0.0, 1.0, 1.0]
        c = PolicyValueContrast(outcomes=y, logging_propensities=e0, target_on=on, target_off=off)
        assert certify_decision(estimate=c) == certify_estimate(c)

    def test_estimate_and_raw_logs_are_mutually_exclusive(self):
        from causalrl.identification.estimate import PolicyValueContrast

        c = PolicyValueContrast.from_binary([0.0, 1.0], [1, 0], propensities=[0.5, 0.5])
        with pytest.raises(ValueError, match="either raw logs"):
            certify_decision([0.0, 1.0], [1, 0], estimate=c)

    def test_missing_all_inputs_raises(self):
        with pytest.raises(ValueError, match="outcomes and treated"):
            certify_decision()
