"""Stimulus tables from NWB, as regressors a causal analysis can condition on.

A visual stimulus drives every responsive neuron in a recorded population at once. That makes it a
textbook confounder for functional connectivity: two neurons that both respond to the same grating
are dependent whether or not anything connects them, and no amount of conditioning on *other
neurons* removes it. It is also the rare confounder that is exactly known — the experimenter chose
it and the file records it — which is what makes it useful as ground truth.

:func:`stimulus_regressors` turns an NWB interval table into per-bin columns aligned to a
:class:`~causalrl.neuro.recording.MultiScaleRecording`, ready to hand to ``discover_lagged``'s
``exogenous`` argument. Orientation enters as ``cos 2θ`` and ``sin 2θ`` scaled by contrast, because
grating orientation is π-periodic and a raw angle in degrees would make 0° and 179° look maximally
different when they are one degree apart.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from causalrl.neuro.io import require_h5py
from causalrl.neuro.recording import RecordingError

__all__ = [
    "EpochTable",
    "contiguous_blocks",
    "read_epochs",
    "stimulus_regressors",
]

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class EpochTable:
    """One NWB interval table: presentation windows plus whatever parameters it carries."""

    name: str
    start: FloatArray
    stop: FloatArray
    parameters: Mapping[str, FloatArray]

    def __len__(self) -> int:
        return int(self.start.shape[0])

    @property
    def total_seconds(self) -> float:
        return float(np.sum(self.stop - self.start))

    def within(self, t_start: float, t_stop: float) -> EpochTable:
        """The presentations overlapping ``[t_start, t_stop)``."""
        keep = (self.stop > t_start) & (self.start < t_stop)
        return EpochTable(
            self.name,
            self.start[keep],
            self.stop[keep],
            {k: v[keep] for k, v in self.parameters.items() if v.shape[0] == keep.shape[0]},
        )

    def mask(self, *, n_bins: int, bin_size: float, t_start: float) -> NDArray[np.bool_]:
        """Per-bin boolean: is this bin inside one of the presentations?"""
        out = np.zeros(n_bins, dtype=bool)
        for a, b in zip(self.start, self.stop, strict=True):
            lo = int(np.floor((a - t_start) / bin_size))
            hi = int(np.ceil((b - t_start) / bin_size))
            if hi <= 0 or lo >= n_bins:
                continue
            out[max(lo, 0) : min(hi, n_bins)] = True
        return out

    def contiguous_blocks(self, gap: float = 10.0) -> list[tuple[float, float, int]]:
        """``(start, stop, n_presentations)`` for runs of presentations separated by < ``gap``.

        A stimulus block is not one long presentation but many short ones with gaps; a contiguous
        *block* is the analysable unit, because slicing out individual trials and concatenating
        them would splice unrelated moments together and corrupt every lagged relationship at the
        seams.
        """
        if len(self) == 0:
            return []
        blocks: list[tuple[float, float, int]] = []
        begin, count = float(self.start[0]), 1
        for i in range(1, len(self)):
            if self.start[i] - self.stop[i - 1] > gap:
                blocks.append((begin, float(self.stop[i - 1]), count))
                begin, count = float(self.start[i]), 1
            else:
                count += 1
        blocks.append((begin, float(self.stop[-1]), count))
        return blocks


def read_epochs(session_path: str | Path) -> dict[str, EpochTable]:
    """Read every interval table in an NWB file that carries presentation windows."""
    h5py = require_h5py()
    path = Path(session_path)
    tables: dict[str, EpochTable] = {}
    with h5py.File(path, "r") as f:
        intervals = f.get("intervals")
        if intervals is None:
            raise RecordingError(f"{path.name} has no intervals group")
        for name in intervals:
            group: Any = intervals[name]
            if "start_time" not in group or "stop_time" not in group:
                continue
            start = np.asarray(group["start_time"][()], dtype=np.float64)
            stop = np.asarray(group["stop_time"][()], dtype=np.float64)
            params: dict[str, FloatArray] = {}
            for key in group:
                if key in ("start_time", "stop_time"):
                    continue
                try:
                    values = group[key][()]
                except Exception:
                    continue
                numeric = np.asarray(values)
                if numeric.dtype.kind in "fiu" and numeric.shape == start.shape:
                    params[key] = numeric.astype(np.float64)
            tables[str(name)] = EpochTable(str(name), start, stop, params)
    return tables


def contiguous_blocks(
    tables: Mapping[str, EpochTable], name: str, *, gap: float = 10.0, min_seconds: float = 60.0
) -> list[tuple[float, float, int]]:
    """Contiguous blocks of ``name`` lasting at least ``min_seconds``."""
    if name not in tables:
        raise KeyError(f"unknown interval table {name!r}; have {sorted(tables)}")
    return [b for b in tables[name].contiguous_blocks(gap) if b[1] - b[0] >= min_seconds]


def stimulus_regressors(
    table: EpochTable,
    *,
    n_bins: int,
    bin_size: float,
    t_start: float,
    include: Sequence[str] = ("on", "contrast", "orientation", "phase"),
) -> dict[str, FloatArray]:
    """Per-bin stimulus columns aligned to a recording's bin grid.

    - ``stim_on`` — 1 while a presentation is on screen, 0 on the blank between them.
    - ``stim_contrast`` — the presentation's contrast, 0 when nothing is on screen.
    - ``stim_ori_cos`` / ``stim_ori_sin`` — ``contrast * cos 2θ`` and ``contrast * sin 2θ``.
      Doubling the angle respects the π-periodicity of orientation; multiplying by contrast means
      the column is zero when there is nothing to be oriented, rather than reporting the
      orientation of an absent grating.
    - ``stim_phase_cos`` / ``stim_phase_sin`` — the **within-trial drift phase** at the stimulus's
      own temporal frequency. A drifting grating is not a step: at 2 Hz it modulates its drive
      every 500 ms, and neurons follow that modulation. The columns above are constant within a
      trial and therefore describe onset, offset and identity but none of the ongoing drive, which
      is the component actually shared between simultaneously recorded neurons. Requires a
      ``temporal_frequency`` parameter in the table.

    Returned as plain columns so they can be passed straight to ``discover_lagged(exogenous=...)``.
    """
    on = table.mask(n_bins=n_bins, bin_size=bin_size, t_start=t_start).astype(np.float64)
    out: dict[str, FloatArray] = {}
    if "on" in include:
        out["stim_on"] = on

    def _per_bin(values: FloatArray) -> FloatArray:
        column = np.zeros(n_bins, dtype=np.float64)
        for a, b, v in zip(table.start, table.stop, values, strict=True):
            lo = int(np.floor((a - t_start) / bin_size))
            hi = int(np.ceil((b - t_start) / bin_size))
            if hi <= 0 or lo >= n_bins or not np.isfinite(v):
                continue
            column[max(lo, 0) : min(hi, n_bins)] = float(v)
        return column

    contrast = (
        _per_bin(table.parameters["contrast"])
        if "contrast" in table.parameters
        else on.copy()  # no contrast column: presence is the only amplitude available
    )
    if "contrast" in include:
        out["stim_contrast"] = contrast
    if "orientation" in include and "orientation" in table.parameters:
        theta = np.deg2rad(_per_bin(table.parameters["orientation"]))
        out["stim_ori_cos"] = contrast * np.cos(2.0 * theta)
        out["stim_ori_sin"] = contrast * np.sin(2.0 * theta)
    if "phase" in include and "temporal_frequency" in table.parameters:
        cos = np.zeros(n_bins, dtype=np.float64)
        sin = np.zeros(n_bins, dtype=np.float64)
        for a, b, freq, amp in zip(
            table.start,
            table.stop,
            table.parameters["temporal_frequency"],
            table.parameters.get("contrast", np.ones_like(table.start)),
            strict=True,
        ):
            lo = int(np.floor((a - t_start) / bin_size))
            hi = int(np.ceil((b - t_start) / bin_size))
            if hi <= 0 or lo >= n_bins or not np.isfinite(freq) or freq <= 0:
                continue
            lo, hi = max(lo, 0), min(hi, n_bins)
            # Phase measured from this trial's own onset: the drift restarts each presentation.
            elapsed = (np.arange(lo, hi) * bin_size + t_start) - a
            angle = 2.0 * np.pi * float(freq) * elapsed
            cos[lo:hi] = float(amp) * np.cos(angle)
            sin[lo:hi] = float(amp) * np.sin(angle)
        out["stim_phase_cos"] = cos
        out["stim_phase_sin"] = sin

    if not out:
        raise RecordingError(f"no regressors built from {table.name!r} (include={list(include)})")
    return out
