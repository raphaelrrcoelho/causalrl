"""Phase 0 acceptance #3: certificate-returning variants (one ID, two bounds) with correct kind."""

import pytest

from causalrl.certify import Certificate, Kind
from causalrl.certify.routines import (
    identify_effect_certified,
    ipw_sensitivity_bounds_certified,
    msm_policy_value_bounds_certified,
)
from causalrl.exceptions import NotIdentifiableError
from causalrl.identification.bounds import Interval
from causalrl.scm.graph import CausalGraph


def _roundtrips(c: Certificate) -> bool:
    return Certificate.from_json(c.to_json()) == c


def test_identify_effect_certified_is_identified() -> None:
    c = identify_effect_certified(CausalGraph([("X", "Y")]), ["X"], ["Y"])
    assert isinstance(c, Certificate)
    assert c.kind is Kind.IDENTIFIED
    assert c.witness is not None
    assert "formula" in c.witness.detail
    assert c.provenance.graph_hash
    assert _roundtrips(c)


def test_identify_effect_certified_raises_on_bow_graph() -> None:
    # X -> Y with a latent confounder X <-> Y: P(Y | do(X)) is not identifiable.
    g = CausalGraph([("X", "Y")], bidirected_edges=[("X", "Y")])
    with pytest.raises(NotIdentifiableError):
        identify_effect_certified(g, ["X"], ["Y"])


def test_ipw_sensitivity_bounds_certified_is_bounded() -> None:
    c = ipw_sensitivity_bounds_certified([1.0, 0.0, 1.0], [0.5, 0.5, 0.5], gamma=1.5)
    assert c.kind is Kind.BOUNDED
    assert isinstance(c.value, Interval)
    assert any(a.name == "MSM" and a.params.get("gamma") == 1.5 for a in c.assumptions)
    assert _roundtrips(c)


def test_msm_policy_value_bounds_certified_is_bounded() -> None:
    c = msm_policy_value_bounds_certified(
        [1.0, 0.0, 1.0], [0.5, 0.5, 0.5], [0.6, 0.4, 0.6], gamma=2.0
    )
    assert c.kind is Kind.BOUNDED
    assert isinstance(c.value, Interval)
    assert c.estimand.query == "policy_value"
    assert _roundtrips(c)


def test_kinds_are_distinct_across_routines() -> None:
    idc = identify_effect_certified(CausalGraph([("X", "Y")]), ["X"], ["Y"])
    bnd = ipw_sensitivity_bounds_certified([1.0, 0.0], [0.5, 0.5], gamma=1.2)
    assert idc.kind is Kind.IDENTIFIED
    assert bnd.kind is Kind.BOUNDED
