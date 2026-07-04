"""Tests for the PolicyValueContrast seam — the typed object certify_estimate consumes.

These pin the validation contract and the from_binary reduction, not the underlying kernels
(those are covered in test_msm_bounds.py / test_pivotality.py).
"""

from __future__ import annotations

import pytest

from causalrl.identification.estimate import PolicyValueContrast


def test_from_binary_msm_only_sets_one_hot_targets():
    c = PolicyValueContrast.from_binary([0.0, 1.0], [1, 0], propensities=[0.5, 0.5])
    assert c.has_msm and not c.has_pivotality
    assert list(c.target_on) == [1.0, 0.0] and list(c.target_off) == [0.0, 1.0]


def test_from_binary_pivotality_only():
    c = PolicyValueContrast.from_binary([0.0, 1.0], [1, 0], confounder_bins=[0, 1])
    assert c.has_pivotality and not c.has_msm


def test_from_binary_both_layers():
    c = PolicyValueContrast.from_binary([0.0, 1.0], [1, 0], propensities=[0.5, 0.5], mi_cap=0.1)
    assert c.has_msm and c.has_pivotality


def test_no_evidence_raises():
    with pytest.raises(ValueError, match=r"confounder_bins|mi_cap|propensities"):
        PolicyValueContrast.from_binary([0.0, 1.0], [1, 0])


def test_both_arms_required():
    with pytest.raises(ValueError, match="both arms"):
        PolicyValueContrast.from_binary([0.1, 0.2, 0.3], [1, 1, 1], mi_cap=0.1)


def test_empty_outcomes_raises():
    with pytest.raises(ValueError, match="non-empty"):
        PolicyValueContrast(outcomes=[], logging_propensities=[], target_on=[], target_off=[])


def test_propensity_range_validated():
    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        PolicyValueContrast(
            outcomes=[0.0, 1.0],
            logging_propensities=[0.0, 0.5],
            target_on=[1.0, 0.0],
            target_off=[0.0, 1.0],
        )


def test_length_mismatch_raises():
    with pytest.raises(ValueError, match="equal length"):
        PolicyValueContrast(
            outcomes=[0.0, 1.0],
            logging_propensities=[0.5],
            target_on=[1.0, 0.0],
            target_off=[0.0, 1.0],
        )


def test_msm_requires_targets():
    with pytest.raises(ValueError, match="target_on and target_off"):
        PolicyValueContrast(outcomes=[0.0, 1.0], logging_propensities=[0.5, 0.5])


def test_treated_length_validated():
    with pytest.raises(ValueError, match="treated must match"):
        PolicyValueContrast(outcomes=[0.0, 1.0], treated=[1], mi_cap=0.1)
