"""Multi-scale electrophysiology container: microscopic spikes + mesoscopic population signals.

One container carries both scales of a recording so every downstream routine (discovery,
abstraction, certification) sees the same object:

- **micro** — binned spike counts, ``(n_bins, n_units)``, one column per sorted unit;
- **meso** — continuous population signals, ``(n_bins, n_channels)``, one column per mesoscopic
  observable (population rate, LFP proxy, per-area aggregate).

Both scales share one time base (``bin_size`` seconds), which is what makes a micro→meso
abstraction map well defined at all. Trials are optional; when present they mark epoch boundaries
so a routine can respect trial structure rather than treating a session as one long stationary
stretch.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from causalrl.exceptions import CausalRLError

__all__ = ["MultiScaleRecording", "RecordingError", "bin_spike_times"]

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


class RecordingError(CausalRLError):
    """A recording is malformed (mismatched shapes, unknown unit/channel, empty time base)."""


def bin_spike_times(
    spike_times: Sequence[Sequence[float] | NDArray[np.float64]],
    *,
    bin_size: float,
    t_start: float = 0.0,
    t_stop: float | None = None,
) -> IntArray:
    """Bin a list of per-unit spike-time arrays (seconds) into ``(n_bins, n_units)`` counts.

    The standard first step for point-process data: everything downstream operates on counts in
    bins of ``bin_size``. Bin width is the analysis's temporal resolution — dependencies faster
    than one bin appear as *contemporaneous* (instantaneous) links, which is exactly why the
    contemporaneous slice of a lagged graph must be read as "common input or sub-bin interaction",
    never as a directed synaptic effect.
    """
    if bin_size <= 0.0:
        raise RecordingError("bin_size must be positive")
    arrays = [np.asarray(st, dtype=np.float64).reshape(-1) for st in spike_times]
    if t_stop is None:
        t_stop = max((float(a.max()) for a in arrays if a.size), default=t_start) + bin_size
    n_bins = int(np.ceil((t_stop - t_start) / bin_size))
    if n_bins <= 0:
        raise RecordingError("empty time base: t_stop must exceed t_start")
    counts = np.zeros((n_bins, len(arrays)), dtype=np.int64)
    for j, times in enumerate(arrays):
        inside = times[(times >= t_start) & (times < t_start + n_bins * bin_size)]
        idx = ((inside - t_start) / bin_size).astype(np.int64)
        np.add.at(counts[:, j], idx, 1)
    return counts


@dataclass(frozen=True)
class MultiScaleRecording:
    """Binned spikes (micro) and population signals (meso) on a shared time base."""

    spikes: IntArray  # (n_bins, n_units) spike counts
    unit_names: tuple[str, ...]
    bin_size: float  # seconds
    meso: FloatArray | None = None  # (n_bins, n_channels) population signals
    meso_names: tuple[str, ...] = ()
    unit_area: Mapping[str, str] = field(default_factory=lambda: {})  # unit -> area
    trial_starts: tuple[int, ...] = ()  # bin indices where a trial begins
    metadata: Mapping[str, object] = field(default_factory=lambda: {})

    def __post_init__(self) -> None:
        if self.spikes.ndim != 2:
            raise RecordingError("spikes must be a (n_bins, n_units) array")
        if self.spikes.shape[1] != len(self.unit_names):
            raise RecordingError(
                f"spikes has {self.spikes.shape[1]} columns but {len(self.unit_names)} unit names"
            )
        if self.bin_size <= 0.0:
            raise RecordingError("bin_size must be positive")
        if len(set(self.unit_names)) != len(self.unit_names):
            raise RecordingError("unit_names must be unique")
        if self.meso is not None:
            if self.meso.ndim != 2:
                raise RecordingError("meso must be a (n_bins, n_channels) array")
            if self.meso.shape[0] != self.spikes.shape[0]:
                raise RecordingError("meso and spikes must share the same number of bins")
            if self.meso.shape[1] != len(self.meso_names):
                raise RecordingError(
                    f"meso has {self.meso.shape[1]} columns but "
                    f"{len(self.meso_names)} channel names"
                )
            if len(set(self.meso_names)) != len(self.meso_names):
                raise RecordingError("meso_names must be unique")
        unknown = set(self.unit_area) - set(self.unit_names)
        if unknown:
            raise RecordingError(f"unit_area references unknown units: {sorted(unknown)}")

    @property
    def n_bins(self) -> int:
        return int(self.spikes.shape[0])

    @property
    def n_units(self) -> int:
        return len(self.unit_names)

    @property
    def duration(self) -> float:
        """Recording duration in seconds."""
        return self.n_bins * self.bin_size

    @property
    def areas(self) -> tuple[str, ...]:
        """Sorted distinct area labels."""
        return tuple(sorted(set(self.unit_area.values())))

    def units_in(self, area: str) -> tuple[str, ...]:
        """Units assigned to ``area``, in recording order."""
        return tuple(u for u in self.unit_names if self.unit_area.get(u) == area)

    def firing_rates(self) -> dict[str, float]:
        """Mean firing rate per unit in spikes/second."""
        mean = self.spikes.mean(axis=0) / self.bin_size
        return dict(zip(self.unit_names, (float(v) for v in mean), strict=True))

    def micro_columns(self) -> dict[str, FloatArray]:
        """Micro scale as a name→column mapping (the input shape every CI test takes)."""
        return {
            name: self.spikes[:, j].astype(np.float64) for j, name in enumerate(self.unit_names)
        }

    def meso_columns(self) -> dict[str, FloatArray]:
        """Meso scale as a name→column mapping (empty when no population signals are attached)."""
        if self.meso is None:
            return {}
        return {name: self.meso[:, j].astype(np.float64) for j, name in enumerate(self.meso_names)}

    def columns(self) -> dict[str, FloatArray]:
        """Both scales in one mapping. Unit and channel names must not collide."""
        micro = self.micro_columns()
        meso = self.meso_columns()
        clash = set(micro) & set(meso)
        if clash:
            raise RecordingError(f"unit and meso channel names collide: {sorted(clash)}")
        return {**micro, **meso}

    def population_rate(self, area: str | None = None) -> FloatArray:
        """Mean spike count per bin across units (of ``area``, or all units)."""
        names = self.units_in(area) if area is not None else self.unit_names
        if not names:
            raise RecordingError(f"no units in area {area!r}")
        idx = [self.unit_names.index(n) for n in names]
        return self.spikes[:, idx].mean(axis=1).astype(np.float64)

    def slice_bins(self, start: int, stop: int) -> MultiScaleRecording:
        """A sub-recording over ``[start, stop)`` bins, keeping both scales aligned."""
        if not 0 <= start < stop <= self.n_bins:
            raise RecordingError(f"invalid bin slice [{start}, {stop}) for {self.n_bins} bins")
        return MultiScaleRecording(
            spikes=self.spikes[start:stop],
            unit_names=self.unit_names,
            bin_size=self.bin_size,
            meso=None if self.meso is None else self.meso[start:stop],
            meso_names=self.meso_names,
            unit_area=dict(self.unit_area),
            trial_starts=tuple(t - start for t in self.trial_starts if start <= t < stop),
            metadata=dict(self.metadata),
        )

    def fingerprint(self) -> str:
        """A stable content hash, for :class:`~causalrl.certify.Provenance.data_fingerprint`."""
        h = hashlib.sha256()
        h.update(np.ascontiguousarray(self.spikes).tobytes())
        h.update(repr(self.unit_names).encode())
        h.update(f"{self.bin_size:.12g}".encode())
        if self.meso is not None:
            h.update(np.ascontiguousarray(np.round(self.meso, 9)).tobytes())
            h.update(repr(self.meso_names).encode())
        return h.hexdigest()[:16]
