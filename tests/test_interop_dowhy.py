"""Tests for the DoWhy -> PolicyValueContrast adapter.

The adapter duck-types a fitted propensity-based DoWhy estimate (it never imports dowhy), so these
tests run with a tiny fake estimate and need no third-party dependency.
"""

from __future__ import annotations

import numpy as np
import pytest

from causalrl import certify_estimate
from causalrl.identification.estimate import PolicyValueContrast
from causalrl.interop.dowhy import from_dowhy_estimate


class _FakeEstimator:
    def __init__(self, scores):
        self.propensity_scores = np.asarray(scores)


class _FakeEstimate:
    """DoWhy stores fitted propensities on the underlying estimator."""

    def __init__(self, scores):
        self.estimator = _FakeEstimator(scores)


class _FakeEstimateDirect:
    """Some versions expose propensity_scores on the estimate itself."""

    def __init__(self, scores):
        self.propensity_scores = np.asarray(scores)


def test_from_dowhy_estimate_reduces_to_from_binary():
    y = [0.0, 1.0, 0.0, 1.0]
    f = [1, 0, 1, 0]
    e0 = [0.6, 0.4, 0.5, 0.5]
    got = from_dowhy_estimate(_FakeEstimate(e0), outcomes=y, treated=f)
    want = PolicyValueContrast.from_binary(y, f, propensities=e0)
    assert certify_estimate(got) == certify_estimate(want)


def test_extract_propensities_from_estimate_directly():
    got = from_dowhy_estimate(_FakeEstimateDirect([0.5, 0.5]), outcomes=[0.0, 1.0], treated=[1, 0])
    assert list(got.logging_propensities) == [0.5, 0.5]
    assert got.has_msm


def test_pivotality_evidence_passes_through():
    got = from_dowhy_estimate(
        _FakeEstimate([0.5, 0.5]), outcomes=[0.0, 1.0], treated=[1, 0], confounder_bins=[0, 1]
    )
    assert got.has_msm and got.has_pivotality


def test_non_propensity_estimate_raises():
    with pytest.raises(TypeError, match="propensity"):
        from_dowhy_estimate(object(), outcomes=[0.0, 1.0], treated=[1, 0])
