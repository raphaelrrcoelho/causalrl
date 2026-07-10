"""Plan I9 (deferred): the pre-2.0 certificate-default-flip ``FutureWarning``.

The three shipped routines with a certificate variant (``identify_effect``,
``ipw_sensitivity_bounds``, ``msm_policy_value_bounds``) warn that 2.0 will return a ``Certificate``
by default unless the caller opts out with ``return_certificate=False``; ``return_certificate=True``
returns the certificate now. Crucially, the library's own wrappers pass ``return_certificate=False``
so they never emit the warning (the completeness gate below turns it into an error and asserts it).
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from causalrl.certify.certificate import Kind
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
from causalrl.identification.id_algorithm import Estimand, identify_effect
from causalrl.scm.graph import CausalGraph

_Y = [1.0, 2.0, 3.0, 4.0]
_E = [0.5, 0.4, 0.6, 0.5]
_PT = [0.7, 0.3, 0.6, 0.4]


def _graph() -> CausalGraph:
    return CausalGraph([("Z", "X"), ("Z", "Y"), ("X", "Y")], [], nodes=["X", "Y", "Z"])


def test_bare_routines_warn_by_default() -> None:
    g = _graph()
    with pytest.warns(FutureWarning, match="2.0"):
        identify_effect(g, ["X"], ["Y"])
    with pytest.warns(FutureWarning, match="return_certificate=False"):
        ipw_sensitivity_bounds(_Y, _E, gamma=1.5)
    with pytest.warns(FutureWarning, match="Certificate by default"):
        msm_policy_value_bounds(_Y, _E, _PT, gamma=1.5)


def test_return_certificate_false_is_silent_and_keeps_legacy_type() -> None:
    g = _graph()
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)  # any leak becomes a test failure
        est = identify_effect(g, ["X"], ["Y"], return_certificate=False)
        iv = ipw_sensitivity_bounds(_Y, _E, gamma=1.5, return_certificate=False)
        pv = msm_policy_value_bounds(_Y, _E, _PT, gamma=1.5, return_certificate=False)
    assert isinstance(est, Estimand)
    assert isinstance(iv, Interval) and isinstance(pv, Interval)


def test_return_certificate_true_returns_certificate() -> None:
    g = _graph()
    id_cert = identify_effect(g, ["X"], ["Y"], return_certificate=True)
    ipw_cert = ipw_sensitivity_bounds(_Y, _E, gamma=1.5, return_certificate=True)
    pv_cert = msm_policy_value_bounds(_Y, _E, _PT, gamma=1.5, return_certificate=True)
    assert id_cert.kind is Kind.IDENTIFIED
    assert ipw_cert.kind is Kind.BOUNDED and pv_cert.kind is Kind.BOUNDED
    # True path is equivalent to the shipped *_certified variants
    assert ipw_cert.value == ipw_sensitivity_bounds_certified(_Y, _E, gamma=1.5).value
    assert pv_cert.value == msm_policy_value_bounds_certified(_Y, _E, _PT, gamma=1.5).value
    assert id_cert.witness == identify_effect_certified(g, ["X"], ["Y"]).witness


def test_internal_wrappers_do_not_leak_the_warning() -> None:
    """Completeness gate: every shipped wrapper opts out, so none emits the flip FutureWarning."""
    rng = np.random.default_rng(0)
    n = 400
    z = rng.standard_normal(n)
    x = rng.binomial(1, 1.0 / (1.0 + np.exp(-0.8 * z))).astype(float)
    y = 1.5 * x + z + rng.standard_normal(n)
    data = {"X": x, "Y": y, "Z": z}

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        # wrappers over the three deprecated routines
        msm_per_step_bounds([_Y], [_E], gamma=1.5)
        msm_stratified_bounds(_Y, _E, [0, 0, 1, 1], {0: 0.5, 1: 0.5}, gamma=1.5)
        msm_contribution_bounds(_Y, _E, _PT, [0.3, 0.7, 0.4, 0.6], gamma=1.5)
        certify_effect(_graph(), "X", "Y", data, method="aipw")
        # the *_certified variants themselves (they delegate with return_certificate=False)
        ipw_sensitivity_bounds_certified(_Y, _E, gamma=1.5)
        msm_policy_value_bounds_certified(_Y, _E, _PT, gamma=1.5)
        identify_effect_certified(_graph(), ["X"], ["Y"])
