"""The multi-scale recording container and the Neo/dataset bridges."""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pytest

from causalrl.neuro.io import (
    DATASETS,
    DatasetUnavailableError,
    from_neo_block,
    from_spike_trains,
    load_dataset,
)
from causalrl.neuro.recording import MultiScaleRecording, RecordingError, bin_spike_times


def test_bin_spike_times_counts_into_the_right_bins() -> None:
    counts = bin_spike_times([[0.0, 0.004, 0.011], [0.006]], bin_size=0.005, t_start=0.0,
                             t_stop=0.015)
    assert counts.shape == (3, 2)
    assert counts[:, 0].tolist() == [2, 0, 1]
    assert counts[:, 1].tolist() == [0, 1, 0]


def test_bin_spike_times_rejects_non_positive_bins() -> None:
    with pytest.raises(RecordingError, match="bin_size must be positive"):
        bin_spike_times([[0.0]], bin_size=0.0)


def _recording(n_bins: int = 20) -> MultiScaleRecording:
    rng = np.random.default_rng(0)
    return MultiScaleRecording(
        spikes=rng.integers(0, 2, size=(n_bins, 3)),
        unit_names=("a", "b", "c"),
        bin_size=0.005,
        meso=rng.standard_normal((n_bins, 2)),
        meso_names=("rate:A", "lfp:A"),
        unit_area={"a": "A", "b": "A", "c": "B"},
    )


def test_recording_reports_areas_units_and_rates() -> None:
    rec = _recording()
    assert rec.areas == ("A", "B")
    assert rec.units_in("A") == ("a", "b")
    assert set(rec.firing_rates()) == {"a", "b", "c"}
    assert rec.duration == pytest.approx(20 * 0.005)


def test_recording_rejects_mismatched_shapes() -> None:
    with pytest.raises(RecordingError, match="unit names"):
        MultiScaleRecording(spikes=np.zeros((4, 2), dtype=np.int64), unit_names=("a",),
                            bin_size=0.005)


def test_recording_rejects_unknown_area_units() -> None:
    with pytest.raises(RecordingError, match="unknown units"):
        MultiScaleRecording(
            spikes=np.zeros((4, 1), dtype=np.int64),
            unit_names=("a",),
            bin_size=0.005,
            unit_area={"zzz": "A"},
        )


def test_columns_expose_both_scales_and_slicing_keeps_them_aligned() -> None:
    rec = _recording()
    cols = rec.columns()
    assert set(cols) == {"a", "b", "c", "rate:A", "lfp:A"}
    sub = rec.slice_bins(5, 15)
    assert sub.n_bins == 10
    assert sub.meso is not None and sub.meso.shape == (10, 2)


def test_fingerprint_is_stable_and_content_sensitive() -> None:
    rec = _recording()
    assert rec.fingerprint() == _recording().fingerprint()
    other = _recording(n_bins=21)
    assert rec.fingerprint() != other.fingerprint()


def test_population_rate_requires_a_populated_area() -> None:
    with pytest.raises(RecordingError, match="no units in area"):
        _recording().population_rate("nope")


# --- Neo bridge (duck-typed stand-ins; Neo itself is never imported) -------------------------


class _SpikeTrain:
    def __init__(self, times: list[float], name: str) -> None:
        self.times = np.asarray(times)
        self.t_start = 0.0
        self.t_stop = 1.0
        self.name = name
        self.annotations: dict[str, object] = {}


class _AnalogSignal:
    def __init__(self, data: np.ndarray, rate: float, name: str) -> None:
        self.magnitude = data
        self.sampling_rate = rate
        self.name = name


class _Segment:
    def __init__(self) -> None:
        self.spiketrains = [_SpikeTrain([0.01, 0.2, 0.5], "u1"), _SpikeTrain([0.05, 0.3], "u2")]
        self.analogsignals = [_AnalogSignal(np.arange(1000.0).reshape(-1, 1), 1000.0, "lfp")]


class _Block:
    def __init__(self) -> None:
        self.segments = [_Segment()]
        self.name = "test-block"


def test_from_neo_block_puts_both_scales_on_one_time_base() -> None:
    rec = from_neo_block(_Block(), bin_size=0.01, area_of={"u1": "M1", "u2": "M1"})
    assert rec.n_units == 2
    assert rec.n_bins == 100
    assert rec.meso is not None and rec.meso.shape == (100, 1)
    assert rec.meso_names == ("lfp",)
    assert rec.areas == ("M1",)
    # Block averaging: bin k covers samples [10k, 10k+10), whose mean is 10k + 4.5.
    assert rec.meso[0, 0] == pytest.approx(4.5)
    assert rec.meso[1, 0] == pytest.approx(14.5)


def test_from_neo_block_accepts_a_callable_area_map() -> None:
    rec = from_neo_block(_Block(), bin_size=0.01, area_of=lambda train: train.name.upper())
    assert set(rec.unit_area.values()) == {"U1", "U2"}


def test_from_neo_block_rejects_an_empty_block() -> None:
    class Empty:
        segments: ClassVar[list[object]] = []

    with pytest.raises(RecordingError, match="no segments"):
        from_neo_block(Empty())


def test_from_spike_trains_checks_name_count() -> None:
    with pytest.raises(RecordingError, match="unit names"):
        from_spike_trains([[0.1], [0.2]], ["only-one"], bin_size=0.01)


def test_dataset_registry_names_the_reach_to_grasp_doi() -> None:
    spec = DATASETS["multielectrode_grasp"]
    assert spec.doi == "10.12751/g-node.f83565"
    assert "gin.g-node.org" in spec.url


def test_load_dataset_refuses_to_download_and_says_where_to_get_it(tmp_path) -> None:
    with pytest.raises(DatasetUnavailableError) as excinfo:
        load_dataset("multielectrode_grasp", tmp_path)
    message = str(excinfo.value)
    assert "10.12751/g-node.f83565" in message
    assert "never fetches datasets implicitly" in message


def test_load_dataset_rejects_unknown_names(tmp_path) -> None:
    with pytest.raises(KeyError, match="unknown dataset"):
        load_dataset("not-a-dataset", tmp_path)
