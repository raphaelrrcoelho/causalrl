"""Tests for the EconML CATE -> PolicyValueContrast adapter.

Duck-typed on ``.effect(X)``; never imports econml, so it runs with a tiny fake CATE estimator.
"""

from __future__ import annotations

import numpy as np

from causalrl import certify_estimate
from causalrl.interop.econml import policy_from_econml_cate


class _FakeCate:
    def __init__(self, tau):
        self._tau = np.asarray(tau, dtype=float)

    def effect(self, X):
        return self._tau


def test_policy_from_econml_cate_builds_msm_only_contrast():
    y = [1.0, 0.0, 1.0, 0.0]
    f = [1, 0, 0, 1]
    e0 = [0.5, 0.5, 0.5, 0.5]
    tau = [0.4, -0.2, 0.3, -0.1]  # induced policy treats units 0 and 2
    c = policy_from_econml_cate(
        _FakeCate(tau), np.zeros((4, 1)), outcomes=y, treated=f, logging_propensities=e0
    )
    assert c.has_msm and not c.has_pivotality
    # induced action pi = [1,0,1,0]; on = 1{f==pi} = [1,1,0,0]; off = 1{f==1-pi} = [0,0,1,1]
    assert list(c.target_on) == [1.0, 1.0, 0.0, 0.0]
    assert list(c.target_off) == [0.0, 0.0, 1.0, 1.0]
    cert = certify_estimate(c)
    assert cert.decision in {"prefer pi_on", "prefer pi_off", "indifferent"}
