"""Plan §10 (the 2.0 flip): the three shipped routines return a ``Certificate`` by default.

At 2.0 the certificate-default flip (I9) is executed: ``identify_effect``,
``ipw_sensitivity_bounds`` and ``msm_policy_value_bounds`` return a :class:`Certificate` when
``return_certificate`` is unset, with NO ``FutureWarning``; ``return_certificate=False`` still
returns the legacy :class:`Estimand` / :class:`Interval`. A byte-stability pin guards that the new
default path wraps exactly the legacy numerics, and that ``certify_decision`` is unchanged.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from causalrl.certify.certificate import Certificate, Kind
from causalrl.certify.routines import (
    identify_effect_certified,
    ipw_sensitivity_bounds_certified,
    msm_policy_value_bounds_certified,
)
from causalrl.estimate.compiler import certify_effect
from causalrl.identification.bounds import (
    Interval,
    ipw_sensitivity_bounds,
    msm_contribution_bounds,
    msm_per_step_bounds,
    msm_policy_value_bounds,
    msm_stratified_bounds,
)
from causalrl.identification.decision import DecisionCertificate, certify_decision
from causalrl.identification.id_algorithm import Estimand, identify_effect
from causalrl.scm.graph import CausalGraph

_Y = [1.0, 2.0, 3.0, 4.0]
_E = [0.5, 0.4, 0.6, 0.5]
_PT = [0.7, 0.3, 0.6, 0.4]


def _graph() -> CausalGraph:
    return CausalGraph([("Z", "X"), ("Z", "Y"), ("X", "Y")], [], nodes=["X", "Y", "Z"])


def test_default_returns_certificate_without_warning() -> None:
    g = _graph()
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # ANY warning (incl. a stray FutureWarning) fails the test
        id_cert = identify_effect(g, ["X"], ["Y"])
        ipw_cert = ipw_sensitivity_bounds(_Y, _E, gamma=1.5)
        pv_cert = msm_policy_value_bounds(_Y, _E, _PT, gamma=1.5)
    assert isinstance(id_cert, Certificate) and id_cert.kind is Kind.IDENTIFIED
    assert isinstance(ipw_cert, Certificate) and ipw_cert.kind is Kind.BOUNDED
    assert isinstance(pv_cert, Certificate) and pv_cert.kind is Kind.BOUNDED


def test_return_certificate_false_keeps_legacy_type() -> None:
    g = _graph()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        est = identify_effect(g, ["X"], ["Y"], return_certificate=False)
        iv = ipw_sensitivity_bounds(_Y, _E, gamma=1.5, return_certificate=False)
        pv = msm_policy_value_bounds(_Y, _E, _PT, gamma=1.5, return_certificate=False)
    assert isinstance(est, Estimand)
    assert isinstance(iv, Interval) and isinstance(pv, Interval)


def test_default_equals_explicit_true() -> None:
    g = _graph()
    assert (
        identify_effect(g, ["X"], ["Y"]).witness
        == identify_effect(g, ["X"], ["Y"], return_certificate=True).witness
    )
    default_iv = ipw_sensitivity_bounds(_Y, _E, gamma=1.5)
    assert isinstance(default_iv, Certificate)
    assert (
        default_iv.value == ipw_sensitivity_bounds(_Y, _E, gamma=1.5, return_certificate=True).value
    )


def test_byte_pin_default_wraps_legacy_numerics() -> None:
    """The new default certificate carries exactly the legacy numeric bound (byte-stable flip)."""
    legacy_iv = ipw_sensitivity_bounds(_Y, _E, gamma=1.5, return_certificate=False)
    default_cert = ipw_sensitivity_bounds(_Y, _E, gamma=1.5)
    assert isinstance(default_cert, Certificate) and isinstance(legacy_iv, Interval)
    assert default_cert.value == legacy_iv  # same Interval, wrapped
    assert default_cert.value == ipw_sensitivity_bounds_certified(_Y, _E, gamma=1.5).value

    legacy_pv = msm_policy_value_bounds(_Y, _E, _PT, gamma=1.5, return_certificate=False)
    default_pv = msm_policy_value_bounds(_Y, _E, _PT, gamma=1.5)
    assert isinstance(default_pv, Certificate)
    assert default_pv.value == legacy_pv
    assert default_pv.value == msm_policy_value_bounds_certified(_Y, _E, _PT, gamma=1.5).value

    est = identify_effect(_graph(), ["X"], ["Y"], return_certificate=False)
    cert = identify_effect(_graph(), ["X"], ["Y"])
    assert isinstance(cert, Certificate) and isinstance(est, Estimand)
    assert cert.witness == identify_effect_certified(_graph(), ["X"], ["Y"]).witness


def test_byte_pin_certify_decision_unchanged() -> None:
    """The decision layer is untouched by the flip: pinned golden decision + naive contrast."""
    outcomes = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
    treated = [1, 1, 1, 0, 0, 0]
    bins = [0, 1, 0, 1, 0, 1]  # confounder bins -> the pivotality layer runs
    cert = certify_decision(outcomes, treated, confounder_bins=bins)
    assert isinstance(cert, DecisionCertificate)
    assert cert.decision == "prefer action 1"
    # naive contrast E[Y|F=1] - E[Y|F=0] = 2/3 - 1/3, exactly (independent of the flip).
    assert cert.naive_contrast == pytest.approx(2.0 / 3.0 - 1.0 / 3.0, abs=1e-12)


def test_internal_wrappers_emit_no_warning_and_keep_working() -> None:
    """Every shipped wrapper opts out (return_certificate=False), so none warns post-flip."""
    rng = np.random.default_rng(0)
    n = 400
    z = rng.standard_normal(n)
    x = rng.binomial(1, 1.0 / (1.0 + np.exp(-0.8 * z))).astype(float)
    y = 1.5 * x + z + rng.standard_normal(n)
    data = {"X": x, "Y": y, "Z": z}
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        msm_per_step_bounds([_Y], [_E], gamma=1.5)
        msm_stratified_bounds(_Y, _E, [0, 0, 1, 1], {0: 0.5, 1: 0.5}, gamma=1.5)
        msm_contribution_bounds(_Y, _E, _PT, [0.3, 0.7, 0.4, 0.6], gamma=1.5)
        cert = certify_effect(_graph(), "X", "Y", data, method="aipw")
        ipw_sensitivity_bounds_certified(_Y, _E, gamma=1.5)
        msm_policy_value_bounds_certified(_Y, _E, _PT, gamma=1.5)
        identify_effect_certified(_graph(), ["X"], ["Y"])
    assert isinstance(cert, Certificate)
