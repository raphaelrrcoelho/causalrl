"""The two-domain transportability demo: covariate shift and the resulting effect gap."""

from __future__ import annotations

from causalrl.envs.suite.transport import make_transport_domains
from causalrl.identification.counterfactual import counterfactual_expectation


def test_domains_differ_only_in_covariate_distribution() -> None:
    source, target, _ = make_transport_domains()
    assert abs(source.see(60_000, seed=0)["Z"].float().mean().item() - 0.2) < 0.02
    assert abs(target.see(60_000, seed=1)["Z"].float().mean().item() - 0.8) < 0.02


def test_true_target_and_biased_source_effects() -> None:
    source, target, _ = make_transport_domains()
    # E*[Y|do(X=1)] = 0.9*P*(Z=1) + 0.5*P*(Z=0) = 0.9*0.8 + 0.5*0.2 = 0.82
    target_effect = counterfactual_expectation(
        target, outcome="Y", intervention={"X": 1.0}, evidence={}, n=60_000, seed=2
    )
    assert abs(target_effect - 0.82) < 0.03
    # E[Y|do(X=1)] in the source = 0.9*0.2 + 0.5*0.8 = 0.58 (biased for the target).
    source_effect = counterfactual_expectation(
        source, outcome="Y", intervention={"X": 1.0}, evidence={}, n=60_000, seed=3
    )
    assert abs(source_effect - 0.58) < 0.03
