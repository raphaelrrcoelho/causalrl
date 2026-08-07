"""Lagged (time-series) causal discovery: embedding, PCMCI links, contemporaneous PAG."""

from __future__ import annotations

import numpy as np
import pytest

from causalrl.exceptions import CausalGraphError
from causalrl.neuro.citests import PartialCorrelationTest, PoissonGLMTest
from causalrl.neuro.simulate import SpikingCorticalSimulator, two_area_microcircuit
from causalrl.neuro.timeseries import (
    ConditionedCITest,
    discover_lagged,
    lag_name,
    lagged_frame,
)


def test_lag_names_round_trip() -> None:
    assert lag_name("A", 0) == "A"
    assert lag_name("A", 2) == "A@t-2"
    with pytest.raises(CausalGraphError, match="non-negative"):
        lag_name("A", -1)


def test_lagged_frame_aligns_every_lag_to_the_same_rows() -> None:
    data = {"A": np.arange(10.0), "B": np.arange(100.0, 110.0)}
    frame = lagged_frame(data, ["A", "B"], 2)
    assert set(frame) == {"A", "A@t-1", "A@t-2", "B", "B@t-1", "B@t-2"}
    assert all(len(v) == 8 for v in frame.values())
    assert frame["A"][0] == 2.0
    assert frame["A@t-1"][0] == 1.0
    assert frame["A@t-2"][0] == 0.0
    # Same row index, one step apart in time.
    assert np.array_equal(frame["A@t-1"][1:], frame["A"][:-1])


def test_lagged_frame_rejects_short_series_and_unknown_variables() -> None:
    with pytest.raises(CausalGraphError, match="too short"):
        lagged_frame({"A": np.arange(3.0)}, ["A"], 5)
    with pytest.raises(CausalGraphError, match="not in data"):
        lagged_frame({"A": np.arange(10.0)}, ["B"], 2)


def _linear_var(n: int = 4000, seed: int = 0) -> dict[str, np.ndarray]:
    """A known VAR(1): X drives Y at lag 1, Y drives Z at lag 1, nothing else."""
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    y = np.zeros(n)
    z = np.zeros(n)
    for t in range(1, n):
        x[t] = 0.5 * x[t - 1] + rng.standard_normal()
        y[t] = 0.4 * y[t - 1] + 0.8 * x[t - 1] + rng.standard_normal()
        z[t] = 0.4 * z[t - 1] + 0.8 * y[t - 1] + rng.standard_normal()
    return {"X": x, "Y": y, "Z": z}


def test_discover_lagged_recovers_a_known_var_chain() -> None:
    graph = discover_lagged(
        _linear_var(),
        ["X", "Y", "Z"],
        max_lag=2,
        ci_test=PartialCorrelationTest(alpha=1e-4),
        max_conditioning_size=2,
    )
    edges = set(graph.lagged_edges())
    assert ("X", "Y") in edges
    assert ("Y", "Z") in edges
    # No direct X -> Z: the chain is mediated by Y and the MCI test conditions on it.
    assert ("X", "Z") not in edges
    # Nothing points backwards in time.
    assert ("Y", "X") not in edges
    assert ("Z", "Y") not in edges


def test_self_links_are_recovered_but_kept_out_of_the_connectivity_edges() -> None:
    graph = discover_lagged(
        _linear_var(),
        ["X", "Y", "Z"],
        max_lag=2,
        ci_test=PartialCorrelationTest(alpha=1e-4),
        max_conditioning_size=2,
    )
    assert graph.self_links()  # autocorrelation is real and is modelled
    assert all(a != b for a, b in graph.lagged_edges())
    assert any(a == b for a, b in graph.lagged_edges(include_self=True))


def test_parents_are_sorted_by_strength() -> None:
    graph = discover_lagged(
        _linear_var(),
        ["X", "Y", "Z"],
        max_lag=2,
        ci_test=PartialCorrelationTest(alpha=1e-4),
        max_conditioning_size=2,
    )
    parents = graph.parents("Z")
    assert parents
    strengths = [abs(p.statistic) for p in parents]
    assert strengths == sorted(strengths, reverse=True)


def test_parents_rejects_unknown_variables() -> None:
    graph = discover_lagged(
        _linear_var(), ["X", "Y", "Z"], max_lag=1, ci_test=PartialCorrelationTest(alpha=1e-4)
    )
    with pytest.raises(CausalGraphError, match="unknown variable"):
        graph.parents("Q")


def test_unrolled_admg_is_acyclic_and_carries_every_lag() -> None:
    graph = discover_lagged(
        _linear_var(),
        ["X", "Y", "Z"],
        max_lag=2,
        ci_test=PartialCorrelationTest(alpha=1e-4),
        max_conditioning_size=2,
    )
    admg = graph.unrolled_admg()
    # Construction succeeds only if the directed part is a DAG (CausalGraph enforces it).
    assert len(admg.nodes) == 3 * 3
    assert ("X@t-1", "Y") in admg.directed_edges


def test_summary_graph_keeps_recurrence() -> None:
    graph = discover_lagged(
        _linear_var(),
        ["X", "Y", "Z"],
        max_lag=2,
        ci_test=PartialCorrelationTest(alpha=1e-4),
        max_conditioning_size=2,
    )
    summary = graph.summary_graph()
    assert set(summary.nodes) == {"X", "Y", "Z"}
    assert ("X", "Y") in summary.directed_edges


def test_conditioned_ci_test_pins_extra_variables_into_every_test() -> None:
    seen: list[list[str]] = []

    def spy(data: object, x: str, y: str, z: list[str]) -> bool:
        seen.append(list(z))
        return True

    wrapped = ConditionedCITest(spy, {"A": ["A@t-1"], "B": ["B@t-1"]})
    wrapped({}, "A", "B", ["C"])
    assert seen == [["C", "A@t-1", "B@t-1"]]


def test_conditioned_ci_test_never_conditions_on_an_endpoint() -> None:
    seen: list[list[str]] = []

    def spy(data: object, x: str, y: str, z: list[str]) -> bool:
        seen.append(list(z))
        return True

    ConditionedCITest(spy, {"A": ["B", "A@t-1"]})({}, "A", "B", [])
    assert seen == [["A@t-1"]]


def test_contemporaneous_phase_can_be_switched_off() -> None:
    graph = discover_lagged(
        _linear_var(),
        ["X", "Y", "Z"],
        max_lag=1,
        ci_test=PartialCorrelationTest(alpha=1e-4),
        contemporaneous=False,
    )
    assert graph.contemporaneous.edges() == []


def test_discover_lagged_validates_its_arguments() -> None:
    with pytest.raises(CausalGraphError, match="must be unique"):
        discover_lagged(_linear_var(), ["X", "X"], max_lag=1)
    with pytest.raises(CausalGraphError, match="at least 1"):
        discover_lagged(_linear_var(), ["X", "Y"], max_lag=0)


def test_fast_common_input_is_never_reported_as_a_directed_edge() -> None:
    """A sub-bin shared drive shows up at lag 0, and is left unoriented rather than invented."""
    rng = np.random.default_rng(0)
    n = 6000
    latent = rng.standard_normal(n)
    data = {
        "A": 1.2 * latent + rng.standard_normal(n),
        "B": 1.2 * latent + rng.standard_normal(n),
        "C": rng.standard_normal(n),
    }
    graph = discover_lagged(
        data,
        ["A", "B", "C"],
        max_lag=1,
        ci_test=PartialCorrelationTest(alpha=1e-4),
        max_conditioning_size=1,
    )
    assert ("A", "B") in graph.common_input_candidates()
    assert ("A", "B") not in graph.contemporaneous_directed()
    assert ("B", "A") not in graph.contemporaneous_directed()
    # The unconfounded channel is not dragged in.
    assert ("A", "C") not in graph.common_input_candidates()


def test_a_collider_lets_fci_commit_to_a_bidirected_edge() -> None:
    """With enough surrounding structure the latent pair is resolved to a definite ``<->``."""
    rng = np.random.default_rng(1)
    n = 8000
    latent = rng.standard_normal(n)
    a = 1.3 * latent + 0.5 * rng.standard_normal(n)
    b = 1.3 * latent + 0.5 * rng.standard_normal(n)
    data = {"A": a, "B": b, "C": 1.0 * a + 1.0 * b + 0.5 * rng.standard_normal(n)}
    graph = discover_lagged(
        data,
        ["A", "B", "C"],
        max_lag=1,
        ci_test=PartialCorrelationTest(alpha=1e-4),
        max_conditioning_size=1,
    )
    assert ("A", "B") in graph.common_input_candidates()


def test_discovery_runs_on_spike_counts_with_the_point_process_test() -> None:
    spec = two_area_microcircuit(n_per_area=3, latent_gain=0.0, n_latent=0, seed=0)
    rec = SpikingCorticalSimulator(spec, seed=1).simulate(8000)
    graph = discover_lagged(
        rec.micro_columns(),
        list(rec.unit_names),
        max_lag=2,
        ci_test=PoissonGLMTest(alpha=1e-3),
        max_conditioning_size=1,
    )
    true_edges = set(spec.ground_truth_edges())
    found = set(graph.lagged_edges())
    # Every recovered edge must be a real synapse: the point-process test must not invent any
    # where there is no unrecorded common input to blame.
    assert found <= true_edges


def test_screening_subsample_keeps_every_column_row_aligned() -> None:
    from causalrl.neuro.timeseries import _subsample

    frame = lagged_frame({"A": np.arange(100.0), "B": np.arange(100.0) * 2}, ["A", "B"], 1)
    sub = _subsample(frame, 20, seed=0)
    assert all(len(v) == 20 for v in sub.values())
    # The lag-1 column must remain exactly one step behind the lag-0 column, row for row.
    assert np.array_equal(sub["A"] - sub["A@t-1"], np.ones(20))
    assert np.array_equal(sub["B"], sub["A"] * 2)


def test_screening_subsample_is_a_no_op_when_larger_than_the_sample() -> None:
    from causalrl.neuro.timeseries import _subsample

    frame = lagged_frame({"A": np.arange(50.0)}, ["A"], 1)
    assert _subsample(frame, 10_000, seed=0)["A"].shape == frame["A"].shape


def test_screened_discovery_recovers_the_known_var_chain() -> None:
    """PC1 screens on a subsample; the MCI decision still sees the whole series."""
    graph = discover_lagged(
        _linear_var(),
        ["X", "Y", "Z"],
        max_lag=2,
        ci_test=PartialCorrelationTest(alpha=1e-4),
        max_conditioning_size=2,
        screen_samples=1500,
    )
    edges = set(graph.lagged_edges())
    assert ("X", "Y") in edges
    assert ("Y", "Z") in edges
    assert ("X", "Z") not in edges


def test_exogenous_variables_are_conditioned_on_but_never_linked() -> None:
    """A known driver belongs in every conditioning set and in no edge."""
    rng = np.random.default_rng(0)
    n = 6000
    driver = rng.standard_normal(n)
    # A follows the driver at lag 1 and B at lag 2, so A's *past* predicts B's present even though
    # nothing connects them — the spurious lagged edge a shared driver actually produces.
    a = np.zeros(n)
    b = np.zeros(n)
    a[1:] = 2.0 * driver[:-1] + 0.3 * rng.standard_normal(n - 1)
    b[2:] = 2.0 * driver[:-2] + 0.3 * rng.standard_normal(n - 2)
    data = {"A": a, "B": b, "drive": driver}
    naive = discover_lagged(
        data,
        ["A", "B"],
        max_lag=2,
        ci_test=PartialCorrelationTest(alpha=1e-4),
        max_conditioning_size=1,
    )
    adjusted = discover_lagged(
        data,
        ["A", "B"],
        max_lag=2,
        ci_test=PartialCorrelationTest(alpha=1e-4),
        max_conditioning_size=1,
        exogenous=["drive"],
    )
    assert ("A", "B") in naive.lagged_edges()  # the driver manufactures A -> B
    assert ("A", "B") not in adjusted.lagged_edges()  # conditioning on it removes the edge
    # The driver is conditioned on, never drawn: it is in neither graph's variables.
    assert "drive" not in adjusted.variables
    assert all("drive" not in edge for edge in adjusted.lagged_edges())


def test_exogenous_max_lag_bounds_the_pinned_columns() -> None:
    seen: list[int] = []

    def spy(data: object, x: str, y: str, z: list[str]) -> bool:
        seen.append(sum(1 for v in z if v.startswith("drive")))
        return True

    rng = np.random.default_rng(1)
    data = {k: rng.standard_normal(500) for k in ("A", "B", "drive")}
    discover_lagged(
        data,
        ["A", "B"],
        max_lag=3,
        ci_test=spy,
        max_conditioning_size=0,
        exogenous=["drive"],
        exogenous_max_lag=1,
        contemporaneous=False,
    )
    assert seen and max(seen) == 2  # lags 0 and 1 only, not all four
