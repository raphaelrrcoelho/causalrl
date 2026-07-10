"""Phase 1 §7.2: identification-aware DR/DML estimation of the back-door ATE.

Covers the acceptance criteria that are locally verifiable (numpy): (a) nominal DML CI coverage over
many replications; (d) non-identified / unsupported / overlap-destroyed queries hedge and never
return a silent point estimate. Plus oracle (hand-computed IPW), nuisance, and serialization tests.
"""

from __future__ import annotations

import numpy as np
import pytest

from causalrl.certify.certificate import (
    Certificate,
    EstimandSpec,
    Kind,
    Provenance,
)
from causalrl.estimate._stats import norm_ppf
from causalrl.estimate.compiler import (
    EstimandNotSupportedError,
    certify_effect,
    compile_estimand,
)
from causalrl.estimate.estimators import estimate_ate
from causalrl.estimate.nuisance import LogisticRegressor, RidgeRegressor
from causalrl.exceptions import NotIdentifiableError
from causalrl.identification.bounds import Interval
from causalrl.scm.graph import CausalGraph

TRUE_ATE = 1.5


def _confounded(
    rng: np.random.Generator, n: int, *, confound: float = 1.5
) -> dict[str, np.ndarray]:
    """Z -> X, Z -> Y, X -> Y with true ATE = TRUE_ATE; Z is the observed confounder."""
    z = rng.standard_normal(n)
    e = 1.0 / (1.0 + np.exp(-confound * z))
    x = rng.binomial(1, e).astype(float)
    y = TRUE_ATE * x + 3.0 * z + rng.standard_normal(n)
    return {"X": x, "Y": y, "Z": z}


def _backdoor_graph() -> CausalGraph:
    return CausalGraph([("Z", "X"), ("Z", "Y"), ("X", "Y")], [], nodes=["X", "Y", "Z"])


# --------------------------------------------------------------------------- pure-numpy helpers


def test_norm_ppf_known_values() -> None:
    assert abs(float(norm_ppf(0.5))) < 1e-9
    assert abs(float(norm_ppf(0.975)) - 1.959963985) < 1e-6
    assert abs(float(norm_ppf(0.95)) - 1.644853627) < 1e-6
    assert float(norm_ppf(0.975)) == pytest.approx(-float(norm_ppf(0.025)), abs=1e-9)
    with pytest.raises(ValueError, match="0 < p < 1"):
        norm_ppf(0.0)


def test_ridge_recovers_linear_coefficients() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((200, 2))
    y = 1.0 + 2.0 * x[:, 0] - 3.0 * x[:, 1]  # noiseless
    model = RidgeRegressor(alpha=1e-10).fit(x, y)
    assert np.allclose(model.beta, [1.0, 2.0, -3.0], atol=1e-4)
    assert np.allclose(model.predict(x), y, atol=1e-4)


def test_logistic_predicts_ordered_probabilities() -> None:
    rng = np.random.default_rng(1)
    z = rng.standard_normal(500)
    x = rng.binomial(1, 1.0 / (1.0 + np.exp(-1.5 * z))).astype(float)
    model = LogisticRegressor().fit(z, x)
    p = model.predict_proba(z)
    assert p.shape == (500,)
    assert np.all((p > 0.0) & (p < 1.0))
    assert model.beta[1] > 0.0  # positive slope recovered
    assert p[x == 1].mean() > p[x == 0].mean()


# ---------------------------------------------------------------------------- estimator behaviour


def test_ipw_matches_hand_computed() -> None:
    # Intercept-only propensity => e = 0.5; Hajek IPW ATE = weighted treated mean - control mean.
    data = {"X": np.array([1.0, 1.0, 0.0, 0.0]), "Y": np.array([2.0, 4.0, 1.0, 3.0])}
    est = estimate_ate(data, "X", "Y", (), method="ipw")
    assert est.value == pytest.approx(1.0, abs=1e-9)  # (3) - (2)


def test_estimate_ate_rejects_nonbinary_treatment() -> None:
    data = {"X": np.array([0.0, 1.0, 2.0]), "Y": np.array([0.0, 1.0, 2.0])}
    with pytest.raises(ValueError, match="must be binary"):
        estimate_ate(data, "X", "Y", ())


def test_estimate_ate_missing_adjustment_raises() -> None:
    data = {"X": np.array([0.0, 1.0]), "Y": np.array([0.0, 1.0])}
    with pytest.raises(ValueError, match="not found in data"):
        estimate_ate(data, "X", "Y", ("Z",))


def test_unconfounded_all_methods_near_truth() -> None:
    rng = np.random.default_rng(2)
    n = 4000
    z = rng.standard_normal(n)
    x = rng.binomial(1, 0.5, n).astype(float)  # treatment independent of Z
    y = TRUE_ATE * x + 2.0 * z + rng.standard_normal(n)
    data = {"X": x, "Y": y, "Z": z}
    for method in ("plugin", "ipw", "aipw", "dml"):
        est = estimate_ate(data, "X", "Y", ("Z",), method=method, seed=0)
        assert abs(est.value - TRUE_ATE) < 0.2, method


def test_confounding_biases_naive_but_dml_recovers() -> None:
    rng = np.random.default_rng(3)
    data = _confounded(rng, 6000)
    naive = float(data["Y"][data["X"] == 1].mean() - data["Y"][data["X"] == 0].mean())
    assert naive > TRUE_ATE + 0.4  # confounding inflates the naive contrast
    est = estimate_ate(data, "X", "Y", ("Z",), method="dml", seed=0)
    assert abs(est.value - TRUE_ATE) < 0.25  # adjustment recovers the truth


def test_dml_ci_coverage_is_nominal() -> None:
    """Acceptance (a): DML 95% CIs cover the true ATE at nominal rate. Deterministic."""
    rng = np.random.default_rng(20260710)
    reps, n, hits = 500, 600, 0
    for _ in range(reps):
        data = _confounded(rng, n)
        est = estimate_ate(data, "X", "Y", ("Z",), method="dml", alpha=0.05, seed=0)
        if est.ci.lower <= TRUE_ATE <= est.ci.upper:
            hits += 1
    coverage = hits / reps
    assert 0.93 <= coverage <= 0.97, f"coverage {coverage:.3f} off nominal 0.95"


# ------------------------------------------------------------------- certificate front door


def test_certify_effect_identified_backdoor() -> None:
    rng = np.random.default_rng(4)
    data = _confounded(rng, 6000, confound=0.8)  # confounded but with adequate overlap
    cert = certify_effect(_backdoor_graph(), "X", "Y", data, method="dml", seed=0)
    assert cert.kind is Kind.IDENTIFIED
    assert cert.hedge is None
    assert cert.value is not None and abs(cert.value - TRUE_ATE) < 0.3
    assert cert.ci is not None and cert.ci.lower < cert.value < cert.ci.upper
    assert cert.witness is not None and cert.witness.detail["set"] == ["Z"]
    assert cert.provenance.graph_hash
    assert cert.provenance.data_fingerprint
    # round-trips losslessly
    assert Certificate.from_json(cert.to_json()).to_dict() == cert.to_dict()


def test_certify_effect_not_identifiable_hedges() -> None:
    """Acceptance (d): a bow arc (X<->Y, X->Y) is not identifiable -> hedge, never a point."""
    bow = CausalGraph([("X", "Y")], [("X", "Y")], nodes=["X", "Y"])
    data = {"X": np.array([0.0, 1.0]), "Y": np.array([0.0, 1.0])}
    cert = certify_effect(bow, "X", "Y", data)
    assert cert.value is None
    assert cert.hedge is not None and cert.hedge.reason == "not-identifiable"


def test_certify_effect_frontdoor_identified_but_unsupported_hedges() -> None:
    """Identifiable by front-door but NOT parent adjustment -> honest hedge, not a bad estimate."""
    fd = CausalGraph([("X", "M"), ("M", "Y")], [("X", "Y")], nodes=["X", "M", "Y"])
    data = {"X": np.array([0.0, 1.0]), "Y": np.array([0.0, 1.0]), "M": np.array([0.0, 1.0])}
    cert = certify_effect(fd, "X", "Y", data)
    assert cert.value is None
    assert cert.hedge is not None and cert.hedge.reason == "estimand-unsupported"


def test_certify_effect_overlap_violation_hedges() -> None:
    """Acceptance (d)/I3: destroyed positivity downgrades to a hedge, not an unstable point."""
    rng = np.random.default_rng(5)
    data = _confounded(rng, 3000, confound=5.0)  # near-deterministic propensity -> min e < eps
    cert = certify_effect(_backdoor_graph(), "X", "Y", data, method="dml", seed=0, overlap_eps=0.01)
    assert cert.value is None
    assert cert.hedge is not None and cert.hedge.reason == "overlap-violation"


def test_compile_estimand_backdoor_and_errors() -> None:
    plan = compile_estimand(_backdoor_graph(), "X", "Y")
    assert plan.adjustment_set == ("Z",)
    assert plan.estimand_render
    with pytest.raises(NotIdentifiableError):
        compile_estimand(CausalGraph([("X", "Y")], [("X", "Y")], nodes=["X", "Y"]), "X", "Y")
    with pytest.raises(EstimandNotSupportedError):
        compile_estimand(
            CausalGraph([("X", "M"), ("M", "Y")], [("X", "Y")], nodes=["X", "M", "Y"]), "X", "Y"
        )


# ---------------------------------------------------------------------------- serialization


def test_certificate_ci_roundtrips() -> None:
    base = dict(
        claim="c",
        estimand=EstimandSpec(query="do"),
        kind=Kind.IDENTIFIED,
        value=0.5,
        alpha=0.05,
        assumptions=(),
        method="dml",
        witness=None,
        hedge=None,
        provenance=Provenance.create(),
    )
    with_ci = Certificate(**base, ci=Interval(0.1, 0.9))
    restored = Certificate.from_json(with_ci.to_json())
    assert restored.ci == Interval(0.1, 0.9)

    without_ci = Certificate(**base)
    assert without_ci.ci is None
    assert Certificate.from_json(without_ci.to_json()).ci is None
