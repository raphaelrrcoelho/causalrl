from causalrl.eval.benchmark import (
    BenchmarkEstimate,
    report_to_dict,
    run_confounded_chain_benchmark,
    run_frontdoor_benchmark,
)


def test_benchmark_estimate_reports_mean_spread_and_interval():
    estimate = BenchmarkEstimate.from_values("agent", seeds=(1, 2, 3), values=(0.2, 0.5, 0.8))
    assert estimate.mean == 0.5
    assert estimate.std > 0.0
    assert estimate.ci95_low < estimate.mean < estimate.ci95_high
    assert report_to_dict({"agent": estimate})["agent"]["values"] == [0.2, 0.5, 0.8]


def test_confounded_chain_report_is_exactly_reproducible_for_equal_seeds():
    kwargs = {"seeds": (1, 2), "n_steps": 300, "tail_window": 100, "n_mc": 100}
    first = run_confounded_chain_benchmark(**kwargs)
    second = run_confounded_chain_benchmark(**kwargs)
    assert first == second
    assert set(first) == {"pomis", "brute_force", "fixed_set"}


def test_confounded_chain_report_preserves_expected_agent_ordering():
    report = run_confounded_chain_benchmark(seeds=(1, 2), n_steps=2000, tail_window=500, n_mc=200)
    assert report["pomis"].mean > report["fixed_set"].mean + 0.25


def test_frontdoor_report_names_manipulability_comparison():
    report = run_frontdoor_benchmark(seeds=(1,), n_steps=20, tail_window=20, n_mc=50)
    assert set(report) == {"manipulability_aware", "naive_filter"}
    assert report["manipulability_aware"].seeds == (1,)
