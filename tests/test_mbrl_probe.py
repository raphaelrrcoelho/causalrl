"""M0 kill-gate harness: shape of the report (the verdict itself is read, not asserted)."""

from __future__ import annotations

from causalrl.eval.benchmark import BenchmarkEstimate
from causalrl.eval.mbrl_probe import run_m0_kill_gate


def test_harness_reports_causal_naive_optimal() -> None:
    report = run_m0_kill_gate(seeds=(0, 1), n=1000)
    assert set(report) == {"causal", "naive", "optimal"}
    for est in report.values():
        assert isinstance(est, BenchmarkEstimate)
        assert 0.0 <= est.mean <= 1.0
        assert len(est.values) == 2
