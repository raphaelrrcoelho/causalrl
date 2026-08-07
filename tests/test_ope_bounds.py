"""Validated partial-identification / OPE bounds.

Manski no-assumptions bounds and the marginal sensitivity model (Tan's Γ) are checked against a
confounded SCM whose true ``do(X=1)`` effect is known: the unobserved ``U`` raises both the
propensity and the outcome, so the naive estimate is biased upward, the true effect lies inside the
Manski bounds, and the sensitivity interval contains the truth once Γ exceeds the true confounding
odds ratio (≈ 2.33 here).
"""

from __future__ import annotations

import numpy as np
import pytest
from torch.distributions import Bernoulli, Distribution, Uniform

from causalrl.identification.bounds import manski_bounds
from causalrl.ope.bounds import ipw_sensitivity_bounds
from causalrl.scm.graph import CausalGraph
from causalrl.scm.mechanisms import FunctionalMechanism, Mechanism
from causalrl.scm.scm import StructuralCausalModel

_N = 40_000


def _confounded_scm() -> StructuralCausalModel:
    graph = CausalGraph(directed_edges=[("U", "X"), ("U", "Y"), ("X", "Y")])
    mechanisms: dict[str, Mechanism] = {
        "U": FunctionalMechanism([], lambda pa, u: u),
        "X": FunctionalMechanism(["U"], lambda pa, u: (u < (0.3 + 0.4 * pa["U"])).float()),
        "Y": FunctionalMechanism(
            ["X", "U"], lambda pa, u: (u < (0.2 + 0.3 * pa["X"] + 0.4 * pa["U"])).float()
        ),
    }
    exogenous: dict[str, Distribution] = {
        "U": Bernoulli(0.5),
        "X": Uniform(0.0, 1.0),
        "Y": Uniform(0.0, 1.0),
    }
    return StructuralCausalModel(graph, mechanisms, exogenous)


def _true_do(scm: StructuralCausalModel, value: int) -> float:
    return float(scm.do({"X": float(value)}).see(_N, seed=1)["Y"].float().mean())


def test_manski_bounds_contain_truth_and_are_sharp() -> None:
    scm = _confounded_scm()
    sample = scm.see(_N, seed=0)
    data = {"X": sample["X"].long().numpy(), "Y": sample["Y"].long().numpy()}
    lo, hi = manski_bounds(data, treatment="X", outcome="Y", action=1)
    assert lo <= _true_do(scm, 1) <= hi
    p = float((data["X"] == 1).mean())
    assert (hi - lo) == pytest.approx(1.0 - p, abs=0.01)  # sharp width = mass of unobserved arm


def test_ipw_sensitivity_collapses_at_one_and_contains_truth() -> None:
    scm = _confounded_scm()
    sample = scm.see(_N, seed=0)
    x = sample["X"].long().numpy()
    y = sample["Y"].float().numpy()
    treated = x == 1
    y_treated = y[treated]
    propensities = np.full(y_treated.shape, float(treated.mean()))  # nominal (no-covariate) e
    truth = _true_do(scm, 1)

    lo1, hi1 = ipw_sensitivity_bounds(y_treated, propensities, gamma=1.0, return_certificate=False)
    assert lo1 == pytest.approx(hi1, abs=1e-9)  # collapses to the point estimate
    assert hi1 == pytest.approx(float(y_treated.mean()), abs=0.01)
    assert hi1 > truth + 0.02  # the confounded point estimate is biased upward

    lo3, hi3 = ipw_sensitivity_bounds(y_treated, propensities, gamma=3.0, return_certificate=False)
    assert lo3 <= truth <= hi3  # Γ > true odds ratio (~2.33): contains the truth
    assert lo3 < lo1 and hi3 > hi1  # monotone widening


def test_ipw_sensitivity_requires_valid_gamma() -> None:
    with pytest.raises(ValueError):
        ipw_sensitivity_bounds([1.0, 0.0], [0.5, 0.5], gamma=0.9, return_certificate=False)
