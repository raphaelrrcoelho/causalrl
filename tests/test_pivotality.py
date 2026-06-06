"""Tests for the decision-pivotality primitives (confounding_bias_bound, mi_flip_threshold,
pivotality_certificate) — the OVB-in-TV lemma, the Pinsker/MI relaxation, and the one-sided
sign-robustness certificate (docs/games/THEORY_pivotality.md)."""

from __future__ import annotations

import numpy as np
import pytest

from causalrl import (
    confounding_bias_bound,
    confounding_bias_per_step_bounds,
    mi_flip_threshold,
    pivotality_certificate,
)


def _confounded_rows(n: int, strength: float, seed: int = 0):
    """Binary Z -> (F, Y) with confounding `strength` (0 => F independent of Z)."""
    rng = np.random.default_rng(seed)
    z = rng.integers(0, 2, size=n)
    p_f = 0.3 + strength * (z - 0.5)  # P(F=1|Z)
    f = (rng.random(n) < p_f).astype(int)
    y = 1.0 * z + 0.5 * f + rng.normal(0, 0.1, size=n)  # Z confounds Y
    return y, f, z


def _adjusted(y, f, z):
    f = f.astype(bool)
    out = 0.0
    for b in np.unique(z):
        in_b = z == b
        out += in_b.mean() * (y[in_b & f].mean() - y[in_b & ~f].mean())
    return out


class TestConfoundingBiasBound:
    def test_lemma_holds_across_strengths_and_seeds(self):
        for strength in (0.0, 0.2, 0.4):
            for seed in range(5):
                y, f, z = _confounded_rows(4000, strength, seed)
                naive = y[f == 1].mean() - y[f == 0].mean()
                bias = abs(naive - _adjusted(y, f, z))
                b_tv = confounding_bias_bound(y, f, z, form="tv")
                b_mi = confounding_bias_bound(y, f, z, form="mi")
                assert bias <= b_tv + 1e-9
                assert b_tv <= b_mi + 1e-9  # Pinsker relaxation only loosens

    def test_independent_z_gives_near_zero_bound(self):
        y, f, z = _confounded_rows(20000, 0.0, seed=3)
        assert confounding_bias_bound(y, f, z, form="tv") < 0.05

    def test_positivity_violation_raises(self):
        y, f, z = _confounded_rows(200, 0.2, seed=0)
        f = f.copy()
        f[z == 1] = 1  # stratum 1 loses its control arm
        with pytest.raises(ValueError, match="positivity"):
            confounding_bias_bound(y, f, z)

    def test_bad_form_raises(self):
        y, f, z = _confounded_rows(200, 0.2)
        with pytest.raises(ValueError, match="form"):
            confounding_bias_bound(y, f, z, form="kl")


class TestMiFlipThreshold:
    def test_inverts_the_sharp_mi_form_bound(self):
        # at MI == MI_flip the sharp MI-form bound equals |naive| exactly
        naive, m1, m0, p = 0.4, 1.3, 0.7, 0.27
        mi = mi_flip_threshold(naive, m1, m0, p)
        bound = np.sqrt(mi / 2 * (m1**2 / p + m0**2 / (1 - p)))
        assert bound == pytest.approx(abs(naive), rel=1e-12)

    def test_sharp_dominates_additive_form(self):
        # the sharp threshold certifies strictly more than the additive (Cauchy-Schwarz) form
        naive, m1, m0, p = 0.4, 1.3, 0.7, 0.27
        additive = 2 * (abs(naive) / (m1 / np.sqrt(p) + m0 / np.sqrt(1 - p))) ** 2
        assert mi_flip_threshold(naive, m1, m0, p) > additive

    def test_zero_spans_certify_everything(self):
        assert mi_flip_threshold(0.4, 0.0, 0.0, 0.5) == float("inf")

    def test_invalid_p_raises(self):
        with pytest.raises(ValueError, match="p_treated"):
            mi_flip_threshold(0.4, 1.0, 1.0, 1.0)


class TestConverseFamily:
    """The tightness constructions of THEORY_pivotality.md, checked end-to-end."""

    @staticmethod
    def _bsc_dataset(eps: float, n: int = 40_000):
        """Exact-count dataset of the symmetric converse family: p=q=1/2, Y=Z,
        P(Z=1|F=1)=1-eps, P(Z=1|F=0)=eps. All cell probabilities are multiples of 1/n."""
        counts = {
            (1, 1): int(0.5 * (1 - eps) * n),
            (1, 0): int(0.5 * eps * n),
            (0, 1): int(0.5 * eps * n),
            (0, 0): int(0.5 * (1 - eps) * n),
        }
        f = np.concatenate([np.full(c, k[0]) for k, c in counts.items()])
        z = np.concatenate([np.full(c, k[1]) for k, c in counts.items()])
        return z.astype(float), f.astype(int), z.astype(int)  # Y = Z

    def test_lemma1_attained_with_equality(self):
        for eps in (0.1, 0.3, 0.45):
            y, f, z = self._bsc_dataset(eps)
            naive = y[f == 1].mean() - y[f == 0].mean()
            bias = abs(naive - _adjusted(y, f.astype(float), z))  # adjusted == 0 (Y dep Z only)
            assert confounding_bias_bound(y, f, z, form="tv") == pytest.approx(bias, abs=1e-9)

    def test_mi_form_asymptotically_sharp(self):
        # near eps=1/2 (small MI) the achieved bias approaches the sharp MI-form bound
        y, f, z = self._bsc_dataset(0.45)
        naive = y[f == 1].mean() - y[f == 0].mean()  # == 1 - 2 eps = 0.1
        bound_mi = confounding_bias_bound(y, f, z, form="mi")
        assert 0.98 <= abs(naive) / bound_mi <= 1.0 + 1e-9

    def test_mi_form_capped_at_trivial(self):
        # far from the small-MI regime the MI form never exceeds M1 + M0
        y, f, z = self._bsc_dataset(0.05)
        assert confounding_bias_bound(y, f, z, form="mi") <= 2.0 + 1e-9


class TestSequentialPerStep:
    def test_growing_channel_prediction(self):
        # 3-step DGP, per-step leak kappa_t increasing: step-1 credit is channel-protected,
        # step-3 is not; per-step bounds hold and grow with t (THEORY Theorem-seq)
        rng = np.random.default_rng(7)
        n, kappas = 60_000, (0.0, 0.5, 0.9)
        z = rng.integers(0, 2, size=n)
        fs = []
        for k in kappas:
            leak = np.where(rng.random(n) < k, z, rng.integers(0, 2, size=n))
            fs.append((rng.random(n) < 0.3 + 0.4 * (leak - 0.5)).astype(int))
        y = z.astype(float) + 0.2 * sum(fs) + rng.normal(0, 0.1, size=n)

        bounds = confounding_bias_per_step_bounds(y, fs, z, form="tv")
        biases = []
        for f_t in fs:
            f_b = f_t.astype(bool)
            naive = y[f_b].mean() - y[~f_b].mean()
            biases.append(abs(naive - _adjusted(y, f_t.astype(float), z)))
        for bias, bound in zip(biases, bounds):
            assert bias <= bound + 1e-9
        assert biases[0] < 0.05 < 0.2 < biases[2]   # protected early, confounded late
        assert bounds[0] < bounds[1] < bounds[2]     # the bounds see the growing channel


class TestPivotalityCertificate:
    def test_certifies_thin_channel(self):
        y, f, z = _confounded_rows(20000, 0.0, seed=1)
        cert = pivotality_certificate(y, f, z)
        assert cert.certified
        assert cert.mi_measured < cert.mi_flip
        assert abs(cert.naive - 0.5) < 0.1

    def test_abstains_when_confounding_can_flip_the_sign(self):
        # small NEGATIVE effect buried under strong positive Z-bias: naive is positive, the
        # adjusted contrast is negative — the certificate must refuse to certify
        rng = np.random.default_rng(1)
        n = 20000
        z = rng.integers(0, 2, size=n)
        f = (rng.random(n) < 0.3 + 0.4 * (z - 0.5)).astype(int)
        y = 1.0 * z - 0.1 * f + rng.normal(0, 0.1, size=n)
        cert = pivotality_certificate(y, f, z)
        assert cert.naive > 0 > _adjusted(y, f, z)  # a real sign flip is present
        assert not cert.certified

    def test_structural_cap_mode(self):
        y, f, _ = _confounded_rows(20000, 0.0, seed=2)
        tight = pivotality_certificate(y, f, mi_cap=1e-6)
        assert tight.certified and tight.mi_measured is None
        vacuous = pivotality_certificate(y, f, mi_cap=10.0)
        assert not vacuous.certified

    def test_requires_z_or_cap(self):
        y, f, _ = _confounded_rows(200, 0.0)
        with pytest.raises(ValueError, match="confounder_bins"):
            pivotality_certificate(y, f)

    def test_soundness_on_the_certificate_claim(self):
        # whenever certified, the true Z-adjusted contrast has the naive sign
        for strength in (0.0, 0.1, 0.2, 0.3):
            for seed in range(4):
                y, f, z = _confounded_rows(5000, strength, seed)
                cert = pivotality_certificate(y, f, z)
                if cert.certified:
                    assert np.sign(_adjusted(y, f, z)) == np.sign(cert.naive)
