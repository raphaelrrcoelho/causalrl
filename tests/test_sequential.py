"""Plan §7.2 (deferred): finite-horizon sequential policy-value estimation (g-comp + sequential DR).

Locally verifiable (numpy): a known-truth T=2 linear-Gaussian sequential DGP where both estimators
recover the analytic policy value; the horizon-1 DR reduces to single-stage DR; per-stage overlap
failures hedge (I3); the identified certificate carries the sequential-ignorability witness and
round-trips through serialization.
"""

from __future__ import annotations

import numpy as np

from causalrl.certify.certificate import Certificate, Kind
from causalrl.estimate.estimators import estimate_ate
from causalrl.ope.sequential import (
    certify_sequential_value,
    estimate_sequential_value,
)

# Analytic policy value of "always treat" for the T=2 DGP below (see _seq_dgp): V = t1 + t2 + g2*b.
T1, T2, G1, G2, ALPHA, BETA = 1.0, 2.0, 0.5, 0.5, 0.7, 1.0
V_ALWAYS1 = T1 + T2 + G2 * BETA  # = 3.5


def _seq_dgp(rng: np.random.Generator, n: int) -> dict[str, np.ndarray]:
    """T=2 confounded sequential DGP; sequential ignorability holds given H1=L1, H2=[L1,L2,A1]."""
    l1 = rng.standard_normal(n)
    a1 = rng.binomial(1, 1.0 / (1.0 + np.exp(-l1))).astype(float)
    l2 = ALPHA * l1 + BETA * a1 + 0.5 * rng.standard_normal(n)
    a2 = rng.binomial(1, 1.0 / (1.0 + np.exp(-l2))).astype(float)
    y = T1 * a1 + T2 * a2 + G1 * l1 + G2 * l2 + 0.5 * rng.standard_normal(n)
    return {"L1": l1, "A1": a1, "L2": l2, "A2": a2, "Y": y}


def _always_one(d: dict[str, np.ndarray]) -> dict[str, list[np.ndarray]]:
    n = len(d["Y"])
    ones = np.ones(n)
    return {
        "histories": [d["L1"], np.column_stack([d["L1"], d["L2"], d["A1"]])],
        "treatments": [d["A1"], d["A2"]],
        "target_actions": [ones, ones],
    }


def test_dr_recovers_analytic_policy_value() -> None:
    rng = np.random.default_rng(0)
    d = _seq_dgp(rng, 6000)
    args = _always_one(d)
    est = estimate_sequential_value(
        args["histories"], args["treatments"], args["target_actions"], d["Y"], method="dr", seed=0
    )
    assert est.horizon == 2
    assert abs(est.value - V_ALWAYS1) < 0.15
    assert est.ci.lower <= V_ALWAYS1 <= est.ci.upper


def test_gcomp_recovers_analytic_policy_value() -> None:
    rng = np.random.default_rng(1)
    d = _seq_dgp(rng, 6000)
    args = _always_one(d)
    est = estimate_sequential_value(
        args["histories"], args["treatments"], args["target_actions"], d["Y"], method="gcomp"
    )
    assert abs(est.value - V_ALWAYS1) < 0.15


def test_horizon1_dr_reduces_to_single_stage_dr() -> None:
    """At T=1, V(always-1) - V(always-0) matches the shipped single-stage DR ATE estimate."""
    rng = np.random.default_rng(2)
    n = 6000
    z = rng.standard_normal(n)
    x = rng.binomial(1, 1.0 / (1.0 + np.exp(-z))).astype(float)
    y = 1.5 * x + 2.0 * z + 0.5 * rng.standard_normal(n)
    ones, zeros = np.ones(n), np.zeros(n)
    v1 = estimate_sequential_value([z], [x], [ones], y, method="dr", seed=7).value
    v0 = estimate_sequential_value([z], [x], [zeros], y, method="dr", seed=7).value
    ate_seq = v1 - v0
    ate_ref = estimate_ate({"X": x, "Y": y, "Z": z}, "X", "Y", ("Z",), method="dml", seed=7).value
    assert abs(ate_seq - 1.5) < 0.1
    assert abs(ate_seq - ate_ref) < 0.05  # the two DR estimators of the ATE agree


def test_certify_sequential_value_identified_and_roundtrips() -> None:
    rng = np.random.default_rng(3)
    d = _seq_dgp(rng, 4000)
    args = _always_one(d)
    # This DGP's stage-2 propensity occasionally reaches ~0.99, adequate for estimation; the strict
    # default positivity floor is exercised by test_overlap_violation_hedges below.
    cert = certify_sequential_value(
        args["histories"],
        args["treatments"],
        args["target_actions"],
        d["Y"],
        seed=0,
        overlap_eps=0.001,
        policy="all-1",
    )
    assert cert.kind is Kind.IDENTIFIED
    assert cert.hedge is None
    assert cert.value is not None and abs(cert.value - V_ALWAYS1) < 0.2
    assert cert.ci is not None
    names = {a.name for a in cert.assumptions}
    assert "sequential-ignorability" in names and "overlap" in names
    # non-checkable ignorability assumption is recorded, never silently assumed
    seq_assumption = next(a for a in cert.assumptions if a.name == "sequential-ignorability")
    assert seq_assumption.checkable is False
    assert Certificate.from_json(cert.to_json()).value == cert.value


def test_overlap_violation_hedges() -> None:
    """A stage with (near-)deterministic treatment destroys positivity -> hedge, never a point."""
    rng = np.random.default_rng(4)
    n = 2000
    l1 = rng.standard_normal(n)
    a1 = rng.binomial(1, 1.0 / (1.0 + np.exp(-8.0 * l1))).astype(float)  # extreme propensities
    y = 1.0 * a1 + l1 + 0.5 * rng.standard_normal(n)
    cert = certify_sequential_value([l1], [a1], [np.ones(n)], y, seed=0, overlap_eps=0.02)
    assert cert.value is None
    assert cert.hedge is not None and cert.hedge.reason == "overlap-violation"


def test_binary_validation() -> None:
    rng = np.random.default_rng(5)
    n = 100
    bad = rng.standard_normal(n)  # continuous "action"
    import pytest

    with pytest.raises(ValueError, match="binary"):
        estimate_sequential_value([bad], [bad], [np.ones(n)], bad)
