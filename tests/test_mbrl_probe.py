"""M0 kill-gate harness: shape of the true-value report (verdict itself is not asserted)."""

from __future__ import annotations

from causalrl.eval.benchmark import BenchmarkEstimate
from causalrl.eval.mbrl_probe import run_m0_kill_gate


def test_harness_returns_four_true_value_estimates() -> None:
    report = run_m0_kill_gate(seeds=(0, 1), n_episodes=400)
    assert set(report) == {"causal_source", "naive_source", "causal_shifted", "naive_shifted"}
    for est in report.values():
        assert isinstance(est, BenchmarkEstimate)
        assert 0.0 <= est.mean <= 1.0
        assert len(est.values) == 2
