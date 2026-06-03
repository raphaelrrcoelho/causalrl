"""Transport-regret certificate (identification/transport_regret.py).

Synthetic policies with *known* witness-dependence so the analytic transfer regret is closed-form:

  (T1) invariant query on the factored diagram -> transportable, witness empty, bound == [0,0];
  (T2) value routed through the witness -> witness == {C}, reweight_required;
  (T3) COVERAGE: the bound contains the analytic transfer regret for any drop profile;
  (T4) abstain mask == exactly the truly witness-dependent units;
  (T5) NON-VACUITY at moderate dependence;
  (T6) negative control: an invariant policy -> mu == 0, bound [0,0], empty mask;
  (T7) degenerate always-flip policy -> bound saturates to the span (still valid, flagged vacuous).
"""

from __future__ import annotations

import numpy as np
import pytest

from causalrl.identification.transport import SelectionDiagram
from causalrl.identification.transport_regret import (
    decision_abstain_mask,
    decision_flip_rate,
    transport_regret_certificate,
)
from causalrl.scm.graph import CausalGraph

ACTION, VALUE, P, C = "action", "value", "P", "C"
VRANGE = (0.0, 1.0)
SPAN = VRANGE[1] - VRANGE[0]
UNITS = list(range(200))


def factored_diagram() -> SelectionDiagram:
    """Value does NOT route through the selection-marked C (the invariant/true graph)."""
    g = CausalGraph(directed_edges=[(ACTION, VALUE), (P, VALUE), (P, C)])
    return SelectionDiagram(g, frozenset({C}))


def monolithic_diagram() -> SelectionDiagram:
    """Value is free to route through the selection-marked C."""
    g = CausalGraph(directed_edges=[(ACTION, VALUE), (P, VALUE), (C, VALUE), (P, C)])
    return SelectionDiagram(g, frozenset({C}))


def dependent_units(fraction: float, seed: int = 0) -> set[int]:
    rng = np.random.default_rng(seed)
    return {u for u in UNITS if rng.random() < fraction}


def make_policy(dep: set[int]):
    """trace/greedy pair whose behaviour depends on the witness exactly on ``dep``."""

    def trace(u: int, w: int):
        return (u, w) if u in dep else (u, 0)

    def greedy(u: int, w: int):
        return w if u in dep else 0

    return trace, greedy


def test_invariant_query_transportable_zero_bound():
    trace, _ = make_policy(set())  # invariant policy
    mu = decision_flip_rate(trace, UNITS)
    cert = transport_regret_certificate(
        factored_diagram(),
        action=ACTION,
        value=VALUE,
        value_range=VRANGE,
        decision_dependence=mu,
    )
    assert cert.transportable is True
    assert cert.non_transportable_witness == frozenset()
    assert cert.reweight_required is False
    assert cert.decision_dependence == 0.0
    assert cert.regret_bound == (0.0, 0.0)


def test_witness_detected_when_value_routes_through_selection():
    cert = transport_regret_certificate(
        monolithic_diagram(),
        action=ACTION,
        value=VALUE,
        value_range=VRANGE,
        decision_dependence=0.3,
    )
    assert cert.non_transportable_witness == frozenset({C})
    assert cert.reweight_required is True
    assert cert.formula is not None
    assert C in cert.formula.adjustment_set


def test_bound_contains_analytic_transfer_regret():
    """Coverage: R_shift = mean per-unit value drop <= span * mu, for any drop profile."""
    rng = np.random.default_rng(7)
    for frac in (0.1, 0.3, 0.6):
        dep = dependent_units(frac, seed=int(frac * 100))
        trace, _ = make_policy(dep)
        mu = decision_flip_rate(trace, UNITS)
        # analytic regret: each dependent unit loses drop_u in [0, span]; others lose 0.
        drops = {u: float(rng.uniform(0.0, SPAN)) for u in dep}
        r_true = sum(drops.values()) / len(UNITS)
        cert = transport_regret_certificate(
            monolithic_diagram(),
            action=ACTION,
            value=VALUE,
            value_range=VRANGE,
            decision_dependence=mu,
        )
        lo, hi = cert.regret_bound
        assert lo == 0.0
        assert r_true <= hi + 1e-12, f"UNDER-COVERAGE frac={frac}: {r_true} > {hi}"
        assert hi <= SPAN + 1e-12


def test_abstain_mask_equals_dependent_set():
    dep = dependent_units(0.25, seed=3)
    _, greedy = make_policy(dep)
    mask = decision_abstain_mask(greedy, UNITS)
    truth = np.array([u in dep for u in UNITS])
    assert np.array_equal(mask, truth)


def test_bound_non_vacuous_at_moderate_dependence():
    dep = dependent_units(0.3, seed=5)
    trace, _ = make_policy(dep)
    mu = decision_flip_rate(trace, UNITS)
    cert = transport_regret_certificate(
        monolithic_diagram(),
        action=ACTION,
        value=VALUE,
        value_range=VRANGE,
        decision_dependence=mu,
    )
    assert 0.0 < cert.regret_bound.upper < 0.9 * SPAN
    assert not cert.is_vacuous(frac=0.9)


def test_negative_control_invariant_policy_silent():
    trace, greedy = make_policy(set())
    assert decision_flip_rate(trace, UNITS) == 0.0
    assert decision_abstain_mask(greedy, UNITS).sum() == 0
    cert = transport_regret_certificate(
        monolithic_diagram(),
        action=ACTION,
        value=VALUE,  # even the permissive graph
        value_range=VRANGE,
        decision_dependence=0.0,
    )
    assert cert.regret_bound.upper == 0.0  # no false alarm


def test_always_flip_saturates_to_span_and_flags_vacuous():
    trace, _ = make_policy(set(UNITS))
    mu = decision_flip_rate(trace, UNITS)
    assert mu == 1.0
    cert = transport_regret_certificate(
        monolithic_diagram(),
        action=ACTION,
        value=VALUE,
        value_range=VRANGE,
        decision_dependence=mu,
    )
    assert abs(cert.regret_bound.upper - SPAN) < 1e-12
    assert cert.is_vacuous(frac=0.9)


def test_empty_units_and_bad_inputs():
    trace, _ = make_policy(set())
    assert decision_flip_rate(trace, []) == 0.0
    with pytest.raises(ValueError):
        transport_regret_certificate(
            monolithic_diagram(),
            action=ACTION,
            value=VALUE,
            value_range=VRANGE,
            decision_dependence=1.5,
        )
    with pytest.raises(ValueError):
        transport_regret_certificate(
            monolithic_diagram(),
            action=ACTION,
            value=VALUE,
            value_range=(1.0, 0.0),
            decision_dependence=0.5,
        )
