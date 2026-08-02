# pyright: reportUnknownMemberType=false
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
    "DATASETS",
    "DatasetSpec",
    "DatasetUnavailableError",
    "from_neo_block",
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
        raise RecordingError(
            f"{len(spike_times)} spike trains but {len(unit_names)} unit names"
        )
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


def _block_average(
    data: FloatArray, *, rate_hz: float, bin_size: float, n_bins: int
) -> FloatArray:
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
