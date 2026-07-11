"""Streaming column-join over a columnar log (plan §9; invariant I5).

The Phase-3 estimators consume the long/tidy :class:`~causalrl.data.trajectory.TrajectoryLog` one
row-batch at a time. A single *decision* ``(entity_id, episode_id, t)`` spans several rows (one per
named quantity), so pulling a weight and a reward cell for one decision means joining rows by key.
:class:`KeyJoiner` does that join with a carry-over buffer: a decision's completed record is emitted
the moment its last required cell arrives and then dropped, so for a key-sorted log
(:meth:`TrajectoryLog.sorted_by_key`) the buffer holds only the one straddling decision — O(1)
memory regardless of log size. :func:`iter_log_batches` streams an in-memory log or an on-disk
Parquet log without materialising it.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import numpy as np
from numpy.typing import NDArray

from causalrl.data.trajectory import TrajectoryLog

__all__ = ["KeyJoiner", "LogSource", "iter_log_batches"]

LogSource = TrajectoryLog | str | os.PathLike[str]
FloatArray = NDArray[np.float64]


def iter_log_batches(source: LogSource, batch_size: int) -> Iterator[TrajectoryLog]:
    """Yield the log in row batches: :meth:`TrajectoryLog.scan` in memory, else streamed Parquet."""
    if isinstance(source, TrajectoryLog):
        yield from source.scan(batch_size)
    else:
        yield from TrajectoryLog.iter_parquet_batches(source, batch_size)


class KeyJoiner:
    """Join a log's named value cells into per-decision records via a carry-over buffer.

    Construct with the tuple of value ``names`` a complete decision must supply; feed successive
    row-batches to :meth:`drain`, which returns the aligned value arrays for every decision that
    became complete in (or by) that batch. Decisions still missing a required cell stay buffered;
    :attr:`dropped` reports how many never completed once the stream ends (a diagnostic).
    """

    def __init__(self, names: tuple[str, ...]) -> None:
        if not names:
            raise ValueError("KeyJoiner needs at least one value name")
        self._names = names
        self._buffer: dict[tuple[int, int, int], dict[str, float]] = {}

    def drain(self, log: TrajectoryLog) -> dict[str, FloatArray]:
        """Return arrays (one per requested name) for decisions completed in this batch."""
        out: dict[str, list[float]] = {nm: [] for nm in self._names}
        names = log.column("name")
        vals = log.column("value")
        eid = log.column("entity_id")
        ep = log.column("episode_id")
        tt = log.column("t")
        wanted = set(self._names)
        for i in range(len(log)):
            nm = str(names[i])
            if nm not in wanted:
                continue
            key = (int(eid[i]), int(ep[i]), int(tt[i]))
            rec = self._buffer.get(key)
            if rec is None:
                rec = {}
                self._buffer[key] = rec
            rec[nm] = float(vals[i])
            if all(name in rec for name in self._names):
                for name in self._names:
                    out[name].append(rec[name])
                del self._buffer[key]
        return {nm: np.asarray(out[nm], dtype=np.float64) for nm in self._names}

    @property
    def dropped(self) -> int:
        """Number of decisions still missing a required cell (incomplete) after the stream ends."""
        return len(self._buffer)
