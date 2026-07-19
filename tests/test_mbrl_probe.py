"""M0 kill-gate harness: shape of the report (the verdict itself is read, not asserted)."""

from __future__ import annotations

from causalrl.eval.benchmark import BenchmarkEstimate
from causalrl.eval.mbrl_probe import (
    run_m0_kill_gate,
    run_m1_discovery_gate,
    run_m1b_dtr_gate,
    run_m2_phase_diagram,
    run_m3_function_approx_gate,
)


def test_harness_reports_causal_naive_optimal() -> None:
    report = run_m0_kill_gate(seeds=(0, 1), n=1000)
    assert set(report) == {"causal", "naive", "optimal"}
    for est in report.values():
        assert isinstance(est, BenchmarkEstimate)
        assert 0.0 <= est.mean <= 1.0
        assert len(est.values) == 2


def test_m1_discovery_harness_reports_discovery_naive_optimal() -> None:
    report = run_m1_discovery_gate(seeds=(0, 1), n=2000)
    assert set(report) == {"discovery", "naive", "optimal"}
    for est in report.values():
        assert isinstance(est, BenchmarkEstimate)
        assert 0.0 <= est.mean <= 1.0
        assert len(est.values) == 2


def test_m1b_dtr_harness_reports_causal_naive_optimal() -> None:
    report = run_m1b_dtr_gate(seeds=(0,), horizon=2, n_episodes=800)
    assert set(report) == {"causal", "naive", "optimal"}
    for est in report.values():
        assert isinstance(est, BenchmarkEstimate)
        assert est.mean >= 0.0
        assert len(est.values) == 1


def test_m2_phase_diagram_shows_the_confounding_signature() -> None:
    report = run_m2_phase_diagram(
        gammas=(0.0, 1.0), shifts=(0.0, 0.6), seeds=(0, 1, 2, 3, 4), n=4000
    )
    assert report.gammas == (0.0, 1.0)
    assert report.shifts == (0.0, 0.6)
    assert set(report.gap) == {(0.0, 0.0), (0.0, 0.6), (1.0, 0.0), (1.0, 0.6)}
    for est in report.gap.values():
        assert isinstance(est, BenchmarkEstimate)
        assert len(est.values) == 5
    # No confounding + no shift -> no advantage; strong confounding + shift -> a clear advantage.
    assert report.gap[(0.0, 0.0)].mean < 0.05
    assert report.gap[(1.0, 0.6)].mean > 0.05
    # The advantage grows (weakly) along both axes -- the phase-diagram signature.
    assert report.monotone_in_gamma is True
    assert report.monotone_in_shift is True


def test_m3_function_approx_gate_reports_causal_naive_optimal() -> None:
    report = run_m3_function_approx_gate(seeds=(0, 1), n=3000)
    assert set(report) == {"causal", "naive", "optimal"}
    for est in report.values():
        assert isinstance(est, BenchmarkEstimate)
        assert 0.0 <= est.mean <= 1.0
        assert len(est.values) == 2
    # The function-approx back-door agent keeps the optimum; the confounded marginal does not.
    assert report["causal"].mean > report["naive"].mean
