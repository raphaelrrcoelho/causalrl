"""Phase 0 acceptance #2: shipped certificates adapt to + round-trip through unified Certificate."""

import pytest

from causalrl.certify import Certificate, Kind, as_certificate
from causalrl.identification.bounds import Interval, PivotalityCertificate
from causalrl.identification.decision import DecisionCertificate
from causalrl.identification.transport_regret import TransportRegretCertificate


def _roundtrips(cert: Certificate) -> bool:
    return Certificate.from_json(cert.to_json()) == cert


def test_pivotality_adapts_and_roundtrips() -> None:
    c = as_certificate(PivotalityCertificate(True, 0.4, 0.1, 2.0, 0.5))
    assert isinstance(c, Certificate)
    assert c.kind is Kind.BOUNDED
    assert c.hedge is None  # certified -> no hedge
    assert any(a.name == "mi-cap" for a in c.assumptions)
    assert _roundtrips(c)


def test_pivotality_uncertified_hedges() -> None:
    c = as_certificate(PivotalityCertificate(False, 0.1, 0.5, 0.2, None))
    assert c.hedge is not None
    assert _roundtrips(c)


def test_decision_act_has_no_hedge() -> None:
    p = PivotalityCertificate(True, 0.4, 0.1, 2.0, 0.5)
    d = DecisionCertificate("prefer treated", 0.4, True, p, None, None, "robust")
    c = as_certificate(d)
    assert c.kind is Kind.BOUNDED
    assert c.hedge is None
    assert c.claim == "robust"
    assert _roundtrips(c)


def test_decision_abstain_has_hedge_and_msm_assumption() -> None:
    d = DecisionCertificate("prefer control", -0.2, False, None, 1.05, False, "not robust")
    c = as_certificate(d)
    assert c.hedge is not None
    assert any(a.name == "MSM" for a in c.assumptions)
    assert _roundtrips(c)


def test_decision_refused_only_by_the_downside_gate_is_not_blamed_on_confounding() -> None:
    """The MSM layer certified; the finite-sample conformal gate is what refused. Naming the
    confounding layer would be exactly the false-provenance bug the certificate layer exists to
    prevent."""
    d = DecisionCertificate("prefer treated", 0.4, False, None, None, True, "gated", -5.0)
    c = as_certificate(d)
    assert c.hedge is not None and c.hedge.reason == "downside-not-certified"
    assert c.hedge.detail is not None and c.hedge.detail["conformal_lcb"] == -5.0
    assert any(a.name == "weighted-exchangeability" for a in c.assumptions)
    assert _roundtrips(c)


def test_transport_regret_adapts_and_roundtrips() -> None:
    c0 = TransportRegretCertificate(
        transportable=True,
        formula=None,
        non_transportable_witness=frozenset({"Z"}),
        reweight_required=True,
        decision_dependence=0.3,
        value_range=(0.0, 1.0),
        regret_bound=Interval(0.0, 0.3),
    )
    c = as_certificate(c0)
    assert c.kind is Kind.BOUNDED
    assert c.value == Interval(0.0, 0.3)
    assert c.witness is not None
    assert c.witness.kind == "transport-formula"
    assert _roundtrips(c)


def test_transport_zero_regret_no_hedge() -> None:
    c0 = TransportRegretCertificate(
        transportable=True,
        formula=None,
        non_transportable_witness=frozenset(),
        reweight_required=False,
        decision_dependence=0.0,
        value_range=(0.0, 1.0),
        regret_bound=Interval(0.0, 0.0),
    )
    c = as_certificate(c0)
    assert c.hedge is None
    assert _roundtrips(c)


def test_as_certificate_idempotent_on_certificate() -> None:
    c = as_certificate(PivotalityCertificate(True, 0.4, 0.1, 2.0, 0.5))
    assert as_certificate(c) is c


def test_as_certificate_rejects_unknown() -> None:
    with pytest.raises(TypeError):
        as_certificate(object())
