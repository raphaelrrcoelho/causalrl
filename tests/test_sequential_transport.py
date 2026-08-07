"""Plan §7.5 (deferred): sequential/policy-value transport estimation (hedge-first).

Locally verifiable (numpy): the one identified subcase (a baseline population shift, downstream
mechanisms shared) recovers the analytic *target* policy value by reweighting the source sequential
g-computation to the target baseline marginal; anything else (a selection node on a time-varying
covariate) hedges (I3), never returns a silent point.
"""

from __future__ import annotations

import numpy as np

from causalrl.certify.certificate import Certificate, Kind
from causalrl.identification.transport import SelectionDiagram
from causalrl.ope.sequential import sequential_ice_values
from causalrl.scm.graph import CausalGraph
from causalrl.transport.estimate import certify_transported_policy_value

# Shared-mechanism T=2 DGP with a DISCRETE baseline B; only P(B) differs across domains.
T1, T2, G1, G2, ALPHA, BETA = 1.0, 2.0, 0.5, 0.5, 0.7, 1.0
# V(always-treat | E[B]) = T1 + T2 + BETA*G2 + (G1 + G2*ALPHA) * E[B]
_SLOPE = G1 + G2 * ALPHA  # 0.85
_INTERCEPT = T1 + T2 + BETA * G2  # 3.5


def _v(p_b1: float) -> float:
    return _INTERCEPT + _SLOPE * p_b1


def _seq_domain(rng: np.random.Generator, n: int, p_b1: float) -> dict[str, np.ndarray]:
    b = rng.binomial(1, p_b1, n).astype(float)
    a1 = rng.binomial(1, 1.0 / (1.0 + np.exp(-(b - 0.5)))).astype(float)
    l2 = ALPHA * b + BETA * a1 + 0.5 * rng.standard_normal(n)
    a2 = rng.binomial(1, 1.0 / (1.0 + np.exp(-l2))).astype(float)
    y = T1 * a1 + T2 * a2 + G1 * b + G2 * l2 + 0.5 * rng.standard_normal(n)
    return {"B": b, "A1": a1, "L2": l2, "A2": a2, "Y": y}


def _graph() -> CausalGraph:
    return CausalGraph(
        directed_edges=[
            ("B", "A1"),
            ("B", "L2"),
            ("B", "Y"),
            ("A1", "L2"),
            ("A1", "Y"),
            ("L2", "A2"),
            ("L2", "Y"),
            ("A2", "Y"),
        ],
        nodes=["B", "A1", "L2", "A2", "Y"],
    )


_STAGES = (
    {"history": ("B",), "treatment": "A1"},
    {"history": ("B", "L2", "A1"), "treatment": "A2"},
)


def test_baseline_shift_transports_to_target_value() -> None:
    """Selection confined to the baseline B (population shift): recover the TARGET policy value."""
    rng = np.random.default_rng(0)
    source = _seq_domain(rng, 8000, p_b1=0.5)  # source E[B] = 0.5
    target = _seq_domain(rng, 8000, p_b1=0.8)  # target E[B] = 0.8
    diagram = SelectionDiagram(_graph(), frozenset({"B"}))

    cert = certify_transported_policy_value(
        diagram, source, target, stages=_STAGES, outcome="Y", target_actions=[1.0, 1.0]
    )
    assert cert.kind is Kind.IDENTIFIED and cert.hedge is None
    assert cert.value is not None
    assert abs(cert.value - _v(0.8)) < 0.15  # transported to the target baseline distribution

    # The correction matters: the naive (source-averaged) value tracks the SOURCE baseline instead.
    naive = float(
        sequential_ice_values(
            [source["B"], np.column_stack([source["B"], source["L2"], source["A1"]])],
            [source["A1"], source["A2"]],
            [np.ones(len(source["Y"])), np.ones(len(source["Y"]))],
            source["Y"],
        ).mean()
    )
    assert abs(naive - _v(0.5)) < 0.15
    assert abs(cert.value - naive) > 0.15  # transport moved the estimate


def test_selection_on_time_varying_covariate_hedges() -> None:
    """A selection node on L2 (a time-varying mechanism) is research-grade -> hedge, not a point."""
    rng = np.random.default_rng(1)
    source = _seq_domain(rng, 3000, p_b1=0.5)
    target = _seq_domain(rng, 3000, p_b1=0.5)
    diagram = SelectionDiagram(_graph(), frozenset({"L2"}))

    cert = certify_transported_policy_value(
        diagram, source, target, stages=_STAGES, outcome="Y", target_actions=[1.0, 1.0]
    )
    assert cert.value is None
    assert cert.hedge is not None
    assert cert.hedge.reason == "non-transportable-sequential"
    assert cert.hedge.detail is not None
    assert cert.hedge.detail["fallback"] == "transport_regret_certificate"


def test_identified_transport_roundtrips() -> None:
    rng = np.random.default_rng(2)
    source = _seq_domain(rng, 4000, p_b1=0.4)
    target = _seq_domain(rng, 4000, p_b1=0.7)
    diagram = SelectionDiagram(_graph(), frozenset({"B"}))
    cert = certify_transported_policy_value(
        diagram,
        source,
        target,
        stages=_STAGES,
        outcome="Y",
        target_actions=[1.0, 1.0],
        policy="all-1",
    )
    names = {a.name for a in cert.assumptions}
    assert "sequential-ignorability" in names and "selection-diagram" in names
    assert cert.witness is not None and cert.witness.kind == "sequential-baseline-transport"
    assert Certificate.from_json(cert.to_json()).value == cert.value
