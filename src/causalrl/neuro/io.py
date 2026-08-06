# pyright: reportUnknownMemberType=false, reportMissingTypeStubs=false
"""Bridges to the electrophysiology stack: Neo objects in, :class:`MultiScaleRecording` out.

The analysis ecosystem this module targets — Neo, Elephant, NIX, and the Jülich INM-6/IAS-6
toolchain built on them — is where real multi-electrode data lives. Nothing here imports Neo or
Elephant: every adapter is duck-typed against the documented object surface, exactly as
:mod:`causalrl.interop.sbi_numpyro` treats NumPyro. A real ``neo.Block`` and a lightweight
stand-in with the same attributes both work, so the bridge is testable without pulling a heavy
dependency into the project.

:data:`DATASETS` records the public multi-electrode datasets this pipeline is designed for, with
their DOIs. :func:`load_dataset` reads a **local** copy; it never downloads. These datasets are
large, access-controlled in some cases, and versioned by DOI — silently fetching them would be the
wrong default, so the loader tells you exactly what to obtain and from where.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from causalrl.neuro.recording import MultiScaleRecording, RecordingError, bin_spike_times

__all__ = [
    "ALLEN_QUALITY",
    "DATASETS",
    "DatasetSpec",
    "DatasetUnavailableError",
    "from_neo_block",
    "from_nwb_ecephys",
    "from_spike_trains",
    "load_dataset",
]

FloatArray = NDArray[np.float64]


class DatasetUnavailableError(RecordingError):
    """A named dataset is not present locally; the message says what to obtain and from where."""


@dataclass(frozen=True)
class DatasetSpec:
    """A public multi-electrode dataset this pipeline is built for."""

    name: str
    description: str
    doi: str
    url: str
    format: str
    notes: str = ""

    def instructions(self, root: Path) -> str:
        return (
            f"dataset {self.name!r} not found under {root}.\n"
            f"  {self.description}\n"
            f"  DOI:    {self.doi}\n"
            f"  Source: {self.url}\n"
            f"  Format: {self.format}\n"
            + (f"  Note:   {self.notes}\n" if self.notes else "")
            + "  Download it separately and pass the directory as `root`; causalrl never fetches "
            "datasets implicitly."
        )


DATASETS: dict[str, DatasetSpec] = {
    "multielectrode_grasp": DatasetSpec(
        name="multielectrode_grasp",
        description=(
            "Massively parallel motor-cortex recordings (Utah array, monkey reach-to-grasp) — "
            "simultaneous spikes and LFP from ~100 electrodes in M1/PMd. The canonical dataset "
            "for micro/meso comparisons of concerted cortical activity."
        ),
        doi="10.12751/g-node.f83565",
        url="https://gin.g-node.org/INT/multielectrode_grasp",
        format="Neo / NIX (.nix) plus ODML metadata; load via neo.io.NixIO",
        notes=(
            "Brochier, Zehl, Hao, Duret, Sprenger, Denker, Grün & Riehle, Sci. Data 4:170055 "
            "(2018). Two sessions, i140703-001 and l101210-001."
        ),
    ),
    "reach_to_grasp_lfp": DatasetSpec(
        name="reach_to_grasp_lfp",
        description=(
            "The LFP-only derivative of the reach-to-grasp recordings, for mesoscopic-scale work "
            "without the sorted-spike layer."
        ),
        doi="10.12751/g-node.f83565",
        url="https://gin.g-node.org/INT/multielectrode_grasp",
        format="Neo / NIX (.nix)",
    ),
    "allen_neuropixels": DatasetSpec(
        name="allen_neuropixels",
        description=(
            "Allen Brain Observatory Neuropixels — simultaneous spiking across many cortical and "
            "subcortical areas, for multi-area abstraction work."
        ),
        doi="10.1038/s41586-020-03171-x",
        url="https://portal.brain-map.org/explore/circuits/visual-coding-neuropixels",
        format="NWB 2 (.nwb); load via pynwb or the AllenSDK",
    ),
}


def _require_h5py() -> Any:
    """Import h5py, or say how to get it. NWB files are HDF5; nothing else here needs it."""
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - depends on the optional dependency
        raise DatasetUnavailableError(
            "reading NWB requires h5py (NWB files are HDF5): pip install 'causalrl[neuro]', "
            "or pip install h5py"
        ) from exc
    return h5py


def _seconds(value: Any) -> FloatArray:
    """Coerce a quantity-like or plain array to a float array of seconds.

    Neo carries ``quantities`` arrays; ``rescale('s')`` is the documented conversion. Anything
    without it is assumed to already be in seconds, which is what a plain NumPy stand-in provides.
    """
    rescale = getattr(value, "rescale", None)
    if callable(rescale):
        # A non-time quantity has no seconds representation; fall through to its raw magnitude.
        with contextlib.suppress(Exception):
            value = rescale("s")
    magnitude = getattr(value, "magnitude", value)
    return np.asarray(magnitude, dtype=np.float64)


def _scalar_seconds(value: Any, default: float) -> float:
    if value is None:
        return default
    arr = _seconds(value).reshape(-1)
    return float(arr[0]) if arr.size else default


def _train_name(train: Any, index: int) -> str:
    """Best available identifier for a spike train: name, then common annotations, then index."""
    name = getattr(train, "name", None)
    if name:
        return str(name)
    annotations: Mapping[str, Any] = getattr(train, "annotations", None) or {}
    for key in ("unit_id", "channel_id", "id", "nix_name"):
        if key in annotations:
            return f"unit_{annotations[key]}"
    return f"unit_{index}"


def from_spike_trains(
    spike_times: Sequence[Sequence[float] | FloatArray],
    unit_names: Sequence[str],
    *,
    bin_size: float,
    t_start: float = 0.0,
    t_stop: float | None = None,
    unit_area: Mapping[str, str] | None = None,
    meso: FloatArray | None = None,
    meso_names: Sequence[str] = (),
) -> MultiScaleRecording:
    """Build a recording from raw spike-time arrays (seconds), optionally with meso signals."""
    if len(spike_times) != len(unit_names):
        raise RecordingError(f"{len(spike_times)} spike trains but {len(unit_names)} unit names")
    counts = bin_spike_times(spike_times, bin_size=bin_size, t_start=t_start, t_stop=t_stop)
    return MultiScaleRecording(
        spikes=counts,
        unit_names=tuple(unit_names),
        bin_size=bin_size,
        meso=meso,
        meso_names=tuple(meso_names),
        unit_area=dict(unit_area or {}),
    )


def from_neo_block(
    block: Any,
    *,
    bin_size: float = 0.005,
    segment: int = 0,
    area_of: Callable[[Any], str] | Mapping[str, str] | None = None,
    include_analog: bool = True,
    analog_downsample: int | None = None,
) -> MultiScaleRecording:
    """Convert a Neo ``Block`` into a :class:`MultiScaleRecording`.

    Spike trains become the micro scale (binned at ``bin_size``); analog signals — LFP, in
    practice — become the meso scale, resampled onto the same bin grid by block averaging so both
    scales share one time base. That shared time base is what makes an abstraction map between them
    well defined, so it is enforced here rather than left to the caller.

    ``area_of`` assigns an area label to each spike train: either a callable taking the train, or a
    mapping from unit name to area. Without it, units carry no area and area-level routines will
    report that there are no areas rather than inventing one.
    """
    segments = getattr(block, "segments", None)
    if not segments:
        raise RecordingError("neo block has no segments")
    if not 0 <= segment < len(segments):
        raise RecordingError(f"segment {segment} out of range (block has {len(segments)})")
    seg = segments[segment]

    trains = list(getattr(seg, "spiketrains", []) or [])
    if not trains:
        raise RecordingError("segment carries no spike trains")
    names = [_train_name(t, i) for i, t in enumerate(trains)]
    if len(set(names)) != len(names):
        names = [f"{n}#{i}" for i, n in enumerate(names)]

    t_start = _scalar_seconds(getattr(trains[0], "t_start", None), 0.0)
    t_stop = _scalar_seconds(getattr(trains[0], "t_stop", None), 0.0)
    times = [_seconds(getattr(t, "times", t)) for t in trains]
    if t_stop <= t_start:
        t_stop = max((float(a.max()) for a in times if a.size), default=t_start) + bin_size
    counts = bin_spike_times(times, bin_size=bin_size, t_start=t_start, t_stop=t_stop)
    n_bins = counts.shape[0]

    unit_area: dict[str, str] = {}
    if callable(area_of):
        unit_area = {n: str(area_of(t)) for n, t in zip(names, trains, strict=True)}
    elif isinstance(area_of, Mapping):
        unit_area = {n: str(area_of[n]) for n in names if n in area_of}

    meso: FloatArray | None = None
    meso_names: list[str] = []
    if include_analog:
        columns: list[FloatArray] = []
        for k, signal in enumerate(getattr(seg, "analogsignals", []) or []):
            data = np.asarray(getattr(signal, "magnitude", signal), dtype=np.float64)
            if data.ndim == 1:
                data = data[:, np.newaxis]
            rate = getattr(signal, "sampling_rate", None)
            rate_hz = float(np.asarray(getattr(rate, "magnitude", rate or 0.0)).reshape(-1)[0])
            if rate_hz <= 0.0:
                raise RecordingError(f"analog signal {k} has no usable sampling_rate")
            resampled = _block_average(data, rate_hz=rate_hz, bin_size=bin_size, n_bins=n_bins)
            if analog_downsample and analog_downsample > 1:
                resampled = resampled[:, ::analog_downsample]
            base = getattr(signal, "name", None) or f"analog{k}"
            for c in range(resampled.shape[1]):
                columns.append(resampled[:, c])
                meso_names.append(f"{base}:{c}" if resampled.shape[1] > 1 else str(base))
        if columns:
            meso = np.column_stack(columns)

    return MultiScaleRecording(
        spikes=counts,
        unit_names=tuple(names),
        bin_size=bin_size,
        meso=meso,
        meso_names=tuple(meso_names),
        unit_area=unit_area,
        metadata={
            "source": "neo",
            "block_name": str(getattr(block, "name", "") or ""),
            "segment": segment,
            "t_start": t_start,
            "t_stop": t_stop,
        },
    )


def _block_average(data: FloatArray, *, rate_hz: float, bin_size: float, n_bins: int) -> FloatArray:
    """Resample a continuous signal onto the spike bin grid by averaging within each bin.

    Block averaging is a low-pass followed by decimation, which is what keeps an LFP comparable to
    a spike count in the same bin instead of aliasing high-frequency content into it.
    """
    samples_per_bin = rate_hz * bin_size
    if samples_per_bin < 1.0:
        # Signal is slower than the bin grid: hold the nearest sample instead of averaging.
        idx = np.clip(
            (np.arange(n_bins) * bin_size * rate_hz).astype(np.int64), 0, data.shape[0] - 1
        )
        return data[idx]
    edges = (np.arange(n_bins + 1) * samples_per_bin).astype(np.int64)
    edges = np.clip(edges, 0, data.shape[0])
    out = np.zeros((n_bins, data.shape[1]), dtype=np.float64)
    cumulative = np.concatenate([np.zeros((1, data.shape[1])), np.cumsum(data, axis=0)], axis=0)
    widths = np.maximum(edges[1:] - edges[:-1], 1)
    out = (cumulative[edges[1:]] - cumulative[edges[:-1]]) / widths[:, np.newaxis]
    return out


def load_dataset(
    name: str,
    root: str | Path,
    *,
    bin_size: float = 0.005,
    segment: int = 0,
    reader: Callable[[Path], Any] | None = None,
    **kwargs: Any,
) -> MultiScaleRecording:
    """Load a registered dataset from a **local** copy under ``root``.

    ``reader`` maps a file path to a Neo ``Block``; the default tries ``neo.io.NixIO`` for NIX
    files and ``neo.io.NWBIO`` for NWB, both imported lazily so neither is a hard dependency.
    Raises :class:`DatasetUnavailableError` — naming the DOI and source — when nothing is found,
    rather than downloading anything.
    """
    if name not in DATASETS:
        raise KeyError(f"unknown dataset {name!r}; known: {sorted(DATASETS)}")
    spec = DATASETS[name]
    base = Path(root)
    candidates = sorted(base.glob("**/*.nix")) + sorted(base.glob("**/*.nwb"))
    if not candidates:
        raise DatasetUnavailableError(spec.instructions(base))
    path = candidates[0]
    block = (reader or _default_reader)(path)
    return from_neo_block(block, bin_size=bin_size, segment=segment, **kwargs)


def _default_reader(path: Path) -> Any:
    """Read a Neo block from a NIX or NWB file, importing Neo only when actually called."""
    suffix = path.suffix.lower()
    try:
        if suffix == ".nix":
            from neo.io import NixIO  # type: ignore[import-not-found]

            return cast("Any", NixIO(str(path), mode="ro").read_block())
        if suffix == ".nwb":
            from neo.io import NWBIO  # type: ignore[import-not-found]

            return cast("Any", NWBIO(str(path), mode="r").read_block())
    except ImportError as exc:  # pragma: no cover - depends on the optional dependency
        raise DatasetUnavailableError(
            f"reading {path.name} needs Neo: pip install neo (and h5py/pynwb for NWB). "
            f"Alternatively pass your own `reader` callable returning a neo.Block."
        ) from exc
    raise DatasetUnavailableError(f"no default Neo reader for {path.suffix!r} files")


# --------------------------------------------------------------------------------------------
# NWB ecephys: spikes and LFP straight out of an NWB file, via h5py.
# --------------------------------------------------------------------------------------------

#: Allen Institute's published quality thresholds for including a sorted unit.
ALLEN_QUALITY = {
    "presence_ratio": 0.9,  # minimum: unit present through the session
    "isi_violations": 0.5,  # maximum: refractory-period violations
    "amplitude_cutoff": 0.1,  # maximum: fraction of spikes lost below threshold
    "firing_rate": 0.5,  # minimum Hz: below this nothing is estimable per bin
}


def _ragged(dataset: Any, index: Any, i: int) -> FloatArray:
    """Row ``i`` of an NWB ragged column: ``index`` holds cumulative END offsets."""
    start = 0 if i == 0 else int(index[i - 1])
    return np.asarray(dataset[start : int(index[i])], dtype=np.float64)


def from_nwb_ecephys(
    session_path: str | Path,
    *,
    bin_size: float = 0.005,
    t_start: float | None = None,
    t_stop: float | None = None,
    areas: Sequence[str] | None = None,
    quality: Mapping[str, float] | None = None,
    max_units_per_area: int | None = None,
    lfp_path: str | Path | None = None,
    lfp_max_channels: int = 8,
) -> MultiScaleRecording:
    """Read an NWB ecephys session (spikes, and optionally LFP) into a recording.

    Targets the standard NWB ecephys layout — a ``units`` table with ragged ``spike_times`` and a
    ``general/extracellular_ephys/electrodes`` table carrying a ``location`` per channel — which is
    what the Allen Institute, IBL and most Neuropixels pipelines publish. Read with ``h5py`` only:
    no pynwb, no Neo, no AllenSDK, consistent with the rest of this module.

    Each unit's **brain area** comes from the ``location`` of its ``peak_channel_id``; that becomes
    the ``unit_area`` map, and hence the ``tau`` of the micro→meso abstraction. ``areas`` restricts
    which are kept.

    ``quality`` filters the units table by column thresholds. Pass :data:`ALLEN_QUALITY` (the
    default when ``quality`` is ``None``) for the published Allen criteria, or ``{}`` to keep every
    unit. Sorted-unit quality is not cosmetic here: a unit with refractory violations is partly
    another neuron's spikes, which is a *measurement-induced* dependence that no amount of causal
    machinery downstream can undo.

    ``lfp_path`` points at the matching probe file; its channels are block-averaged onto the same
    bin grid and attached as the mesoscopic scale.
    """
    h5py = _require_h5py()

    path = Path(session_path)
    if not path.exists():
        raise DatasetUnavailableError(f"NWB file not found: {path}")
    thresholds = ALLEN_QUALITY if quality is None else dict(quality)

    # h5py ships no type stubs and its __getitem__ returns Group | Dataset | Datatype, so the
    # handle stays Any (via _require_h5py) rather than being cast at every access site below.
    with h5py.File(path, "r") as f:
        if "units" not in f:
            raise RecordingError(f"{path.name} has no units table (not a sorted-spike NWB file)")
        units = f["units"]
        n_units = len(units["id"][()])

        keep = np.ones(n_units, dtype=bool)
        if "quality" in units:
            keep &= units["quality"][()] == b"good"
        for column, bound in thresholds.items():
            if column not in units:
                continue
            values = np.asarray(units[column][()], dtype=np.float64)
            # presence_ratio and firing_rate are floors; the rest are ceilings.
            floor = column in ("presence_ratio", "firing_rate")
            keep &= values >= bound if floor else values <= bound

        area = np.full(n_units, "?", dtype=object)
        electrodes = f.get("general/extracellular_ephys/electrodes")
        if electrodes is not None and "peak_channel_id" in units:
            location = {
                int(i): loc.decode() if isinstance(loc, bytes) else str(loc)
                for i, loc in zip(electrodes["id"][()], electrodes["location"][()], strict=True)
            }
            peak = units["peak_channel_id"][()]
            area = np.array([location.get(int(p), "?") for p in peak], dtype=object)
        if areas is not None:
            keep &= np.isin(area, list(areas))

        selected = np.flatnonzero(keep)
        if max_units_per_area is not None:
            counted: dict[str, int] = {}
            trimmed = []
            for i in selected:
                a = str(area[i])
                if counted.get(a, 0) >= max_units_per_area:
                    continue
                counted[a] = counted.get(a, 0) + 1
                trimmed.append(i)
            selected = np.array(trimmed, dtype=np.int64)
        if selected.size == 0:
            raise RecordingError(
                f"no units survived the filter (areas={areas}, quality={thresholds}); "
                f"{n_units} units in the file"
            )

        spike_times, spike_index = units["spike_times"], units["spike_times_index"][()]
        trains = [_ragged(spike_times, spike_index, int(i)) for i in selected]
        lo = float(t_start) if t_start is not None else float(min(t[0] for t in trains if t.size))
        hi = float(t_stop) if t_stop is not None else float(max(t[-1] for t in trains if t.size))
        if hi <= lo:
            raise RecordingError(f"empty time window [{lo}, {hi})")
        trains = [t[(t >= lo) & (t < hi)] for t in trains]

        names = [f"{area[i]}_{int(units['id'][()][i])}" for i in selected]
        unit_area = {n: str(area[i]) for n, i in zip(names, selected, strict=True)}
        counts = bin_spike_times(trains, bin_size=bin_size, t_start=lo, t_stop=hi)

    meso, meso_names = None, ()
    if lfp_path is not None:
        meso, meso_names = _read_nwb_lfp(
            Path(lfp_path),
            t_start=lo,
            t_stop=hi,
            bin_size=bin_size,
            n_bins=counts.shape[0],
            max_channels=lfp_max_channels,
        )

    return MultiScaleRecording(
        spikes=counts,
        unit_names=tuple(names),
        bin_size=bin_size,
        meso=meso,
        meso_names=meso_names,
        unit_area=unit_area,
        metadata={
            "source": "nwb-ecephys",
            "file": path.name,
            "t_start": lo,
            "t_stop": hi,
            "n_units_in_file": n_units,
            "quality": dict(thresholds),
        },
    )


def _read_nwb_lfp(
    path: Path,
    *,
    t_start: float,
    t_stop: float,
    bin_size: float,
    n_bins: int,
    max_channels: int,
) -> tuple[FloatArray, tuple[str, ...]]:
    """Read the LFP window from an NWB probe file and block-average it onto the spike bin grid.

    Only the requested time window is read off disk — a full probe's LFP is gigabytes, and the
    timestamps are sorted, so a binary search bounds the slice.
    """
    h5py = _require_h5py()

    with h5py.File(path, "r") as f:  # Any for the same reason as above
        acquisition = f.get("acquisition")
        if acquisition is None:
            raise RecordingError(f"{path.name} has no acquisition group")
        series: Any = None
        for key in acquisition:
            checked = acquisition[key]
            if not isinstance(checked, h5py.Group):
                continue
            group: Any = checked  # guarded above; widen again for the unstubbed member access
            for inner in group:
                candidate = group[inner]
                if isinstance(candidate, h5py.Group) and "timestamps" in candidate:
                    series = candidate
                    break
            if series is not None:
                break
        if series is None:
            raise RecordingError(f"{path.name} has no LFP series with timestamps")

        timestamps: Any = series["timestamps"]
        lo = int(np.searchsorted(timestamps[()], t_start, side="left"))
        hi = int(np.searchsorted(timestamps[()], t_stop, side="right"))
        if hi <= lo:
            raise RecordingError("LFP covers no part of the requested window")
        data: Any = series["data"]
        step = max(1, data.shape[1] // max_channels)
        channels = list(range(0, data.shape[1], step))[:max_channels]
        window = np.asarray(data[lo:hi, :], dtype=np.float64)[:, channels]
        times = np.asarray(timestamps[lo:hi], dtype=np.float64)

    # Average every LFP sample into the spike bin it falls in; empty bins interpolate.
    index = np.clip(((times - t_start) / bin_size).astype(np.int64), 0, n_bins - 1)
    out = np.zeros((n_bins, len(channels)), dtype=np.float64)
    hits = np.zeros(n_bins, dtype=np.int64)
    np.add.at(out, index, window)
    np.add.at(hits, index, 1)
    filled = hits > 0
    out[filled] /= hits[filled][:, np.newaxis]
    if not filled.all():
        rows = np.flatnonzero(filled)
        for c in range(out.shape[1]):
            out[:, c] = np.interp(np.arange(n_bins), rows, out[rows, c])
    return out, tuple(f"lfp:{c}" for c in channels)
