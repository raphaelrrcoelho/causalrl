"""Phase 0: unified Certificate protocol + serialization (§5.2; invariants I1-I3)."""

import dataclasses
import datetime as dt
import json

from causalrl.certify import (
    Assumption,
    Certificate,
    EstimandSpec,
    Hedge,
    Kind,
    Provenance,
    Witness,
)
from causalrl.identification.bounds import Interval


def _prov() -> Provenance:
    return Provenance(
        library_version="1.3.0",
        seeds=(0, 1),
        data_fingerprint="abc",
        graph_hash="def",
        timestamp="2026-07-10T00:00:00+00:00",
    )


def _cert(value: float | Interval | None) -> Certificate:
    return Certificate(
        claim="E[Y|do(X=1)] is identified",
        estimand=EstimandSpec(query="do", target="mean", policy="pi", domains=("source", "target")),
        kind=Kind.IDENTIFIED,
        value=value,
        alpha=0.05,
        assumptions=(
            Assumption(
                name="backdoor", params={"set": ["Z"]}, checkable=True, diagnostic={"overlap": 0.9}
            ),
        ),
        method="identify_effect@1.3.0",
        witness=Witness(kind="adjustment-set", detail={"set": ["Z"]}),
        hedge=None,
        provenance=_prov(),
    )


def test_kind_values() -> None:
    assert {k.value for k in Kind} == {"identified", "bounded", "empirical"}


def test_estimandspec_defaults() -> None:
    e = EstimandSpec(query="do")
    assert e.target == "mean"
    assert e.policy is None
    assert e.domains == ()


def test_roundtrip_float_value() -> None:
    c = _cert(1.5)
    assert Certificate.from_json(c.to_json()) == c


def test_roundtrip_interval_value() -> None:
    c = _cert(Interval(0.2, 0.8))
    r = Certificate.from_json(c.to_json())
    assert r == c
    assert isinstance(r.value, Interval)
    assert (r.value.lower, r.value.upper) == (0.2, 0.8)


def test_roundtrip_none_value() -> None:
    c = _cert(None)
    assert Certificate.from_json(c.to_json()) == c


def test_roundtrip_with_hedge_and_downgrade() -> None:
    c = Certificate(
        claim="mean refused; quantile reported",
        estimand=EstimandSpec(query="do", target="quantile"),
        kind=Kind.BOUNDED,
        value=Interval(0.0, 1.0),
        alpha=None,
        assumptions=(Assumption("MSM", {"gamma": 1.5}),),
        method="msm@1.3.0",
        witness=None,
        hedge=Hedge(
            reason="moment-condition-failed", detail={"tail_index": 1.3}, downgraded_from="mean"
        ),
        provenance=_prov(),
    )
    r = Certificate.from_json(c.to_json())
    assert r == c
    assert r.hedge is not None
    assert r.hedge.downgraded_from == "mean"


def test_to_json_is_string_and_parses() -> None:
    s = _cert(1.0).to_json()
    assert isinstance(s, str)
    json.loads(s)  # valid JSON


def test_str_contains_claim_and_kind() -> None:
    s = str(_cert(1.0))
    assert "identified" in s.lower()
    assert "E[Y|do(X=1)]" in s


def test_provenance_create_fills_version_and_iso_timestamp() -> None:
    p = Provenance.create(seeds=(7,), graph_hash="h")
    assert p.seeds == (7,)
    assert p.graph_hash == "h"
    assert isinstance(p.library_version, str) and p.library_version
    dt.datetime.fromisoformat(p.timestamp)  # ISO-8601 parseable


def test_certificates_equal_by_value() -> None:
    assert _cert(1.0) == _cert(1.0)
    assert _cert(1.0) != _cert(2.0)


def test_kind_is_a_distinguishing_field() -> None:
    a = _cert(1.0)
    b = dataclasses.replace(a, kind=Kind.BOUNDED)
    assert a != b
