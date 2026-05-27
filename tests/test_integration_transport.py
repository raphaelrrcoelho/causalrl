"""Headline: the transport formula recovers the target effect; naive transfer is biased."""

from __future__ import annotations

from causalrl.envs.suite.transport import make_transport_domains
from causalrl.identification.counterfactual import counterfactual_expectation
from causalrl.identification.transport import transport_formula, transported_effect


def test_transported_effect_recovers_target_not_naive_source() -> None:
    source, target, diagram = make_transport_domains()
    formula = transport_formula(diagram, treatment="X", outcome="Y")
    assert formula is not None
    assert formula.kind == "adjustment"

    true_target = counterfactual_expectation(
        target, outcome="Y", intervention={"X": 1.0}, evidence={}, n=60_000, seed=0
    )
    transported = transported_effect(
        formula,
        treatment="X",
        treated_value=1.0,
        outcome="Y",
        source=source,
        target=target,
        n=60_000,
        seed=1,
    )
    naive_source = counterfactual_expectation(
        source, outcome="Y", intervention={"X": 1.0}, evidence={}, n=60_000, seed=2
    )

    assert abs(transported - true_target) < 0.03  # transport recovers the target effect
    assert abs(transported - 0.82) < 0.03
    assert abs(transported - naive_source) > 0.15  # naive transfer is badly biased
