"""Incrementally build a :class:`~causalrl.data.trajectory.TrajectoryLog` while a producer runs.

:class:`~causalrl.data.trajectory.TrajectoryLog` is build-once: ``__init__`` and ``from_rows`` both
want a complete set of rows up front, and :meth:`~causalrl.data.trajectory.TrajectoryLog.scan` only
slices a log that already exists -- neither helps a caller that does not have a complete log yet.
That describes almost every *live* producer of trajectory data: an online controller, a running
experiment, a robot control loop firing at some sampling rate. Today each of those has to
accumulate its own Python list of row dicts by hand and construct the log only once the run stops,
which means :func:`~causalrl.ope.ipw.stream_policy_value` -- built for exactly this streaming shape
of producer, one row-batch at a time -- cannot actually be fed by one until there is nothing left
to stream.

:class:`TrajectoryLogBuilder` is the accumulator that closes that gap. Rows land through
:meth:`~TrajectoryLogBuilder.push`, or through :meth:`~TrajectoryLogBuilder.episode`, a context
manager that fixes ``episode_id`` for every push inside it so a caller stepping through one
episode's rows in order never has to repeat it. Both land in plain Python lists, one per column,
appended to directly rather than converted to a NumPy array on every call -- ``list.append`` is
amortised O(1), so an N-row session costs O(N) rather than the O(N^2) that reconverting everything
pushed so far on every single push would cost. :meth:`~TrajectoryLogBuilder.freeze` hands back a
genuine :class:`~causalrl.data.trajectory.TrajectoryLog` by running the exact column conversion
:meth:`~causalrl.data.trajectory.TrajectoryLog.from_rows` runs internally, just spread across many
calls instead of one -- which is also why a log built incrementally and one built from the same
rows in a single ``from_rows`` call come out indistinguishable: same dtypes, same fingerprint.
"""

from __future__ import annotations

from collections.abc import Generator, Mapping
from contextlib import contextmanager
from typing import Any

from causalrl.data.trajectory import TrajectoryLog

__all__ = ["EpisodeWriter", "TrajectoryLogBuilder"]


class EpisodeWriter:
    """A push handle bound to one ``episode_id``, valid only inside its ``with`` block.

    Returned by :meth:`TrajectoryLogBuilder.episode`; there is no reason to construct one
    directly. Binding ``episode_id`` at handle-creation time, rather than tracking a "current
    episode" on the builder itself, is what makes concurrent or nested ``with`` blocks safe: two
    handles never share mutable state, so each one always stamps its own fixed id no matter how
    their blocks are interleaved. The handle deliberately stops working once its block exits --
    :meth:`push` starts raising -- so a reference that escapes the block cannot go on silently
    tagging rows with an episode id the surrounding code no longer owns.
    """

    def __init__(self, builder: TrajectoryLogBuilder, episode_id: int) -> None:
        self._builder = builder
        self._episode_id = episode_id
        self._closed = False

    @property
    def episode_id(self) -> int:
        """The episode id every :meth:`push` through this handle stamps its row with."""
        return self._episode_id

    def push(
        self,
        *,
        kind: str,
        name: str,
        value: Any,
        t: int,
        entity_id: int = 0,
        regime: str = "observed",
        observed: bool = True,
    ) -> None:
        """Append one row at this handle's episode id. See :meth:`TrajectoryLogBuilder.push`."""
        if self._closed:
            raise RuntimeError(
                f"push() called on the EpisodeWriter for episode_id={self._episode_id} after its "
                "`with builder.episode(...)` block already exited. A handle is only valid for the "
                "lifetime of that block, precisely so a reference that escapes it cannot silently "
                "keep tagging rows with an episode id the surrounding code no longer owns -- open "
                "a new `with builder.episode(...)` block to push more rows."
            )
        self._builder.push(
            episode_id=self._episode_id,
            kind=kind,
            name=name,
            value=value,
            t=t,
            entity_id=entity_id,
            regime=regime,
            observed=observed,
        )

    def close(self) -> None:
        """Stop accepting pushes. Called automatically when the owning ``with`` block exits.

        Idempotent and safe to call directly, though there is normally no reason to:
        :meth:`TrajectoryLogBuilder.episode` calls it from a ``finally``, so it runs whether the
        block exits normally or via an exception.
        """
        self._closed = True


class TrajectoryLogBuilder:
    """Accumulate rows for a :class:`TrajectoryLog` across many calls instead of just one.

    The counterpart to :meth:`TrajectoryLog.from_rows` for a producer that does not have all its
    rows up front: push them in as they happen and call :meth:`freeze` whenever a
    :class:`TrajectoryLog` is needed -- as many times as needed, including more than once per
    builder. :meth:`push` is the primitive; :meth:`episode` is a context manager built on top of it
    that fixes ``episode_id`` for every push inside it.
    """

    def __init__(self, metadata: Mapping[str, Any] | None = None) -> None:
        """Start an empty builder. ``metadata`` is attached to every log :meth:`freeze` returns."""
        self._cols: dict[str, list[Any]] = {
            "entity_id": [],
            "episode_id": [],
            "t": [],
            "kind": [],
            "name": [],
            "value": [],
            "regime": [],
            "observed": [],
        }
        self._metadata: dict[str, Any] = dict(metadata or {})

    def __len__(self) -> int:
        """Rows pushed so far, across every episode, regardless of whether frozen yet."""
        return len(self._cols["entity_id"])

    @property
    def metadata(self) -> dict[str, Any]:
        """A copy of the log-level metadata :meth:`freeze` will attach to the next snapshot."""
        return dict(self._metadata)

    def push(
        self,
        *,
        episode_id: int,
        kind: str,
        name: str,
        value: Any,
        t: int,
        entity_id: int = 0,
        regime: str = "observed",
        observed: bool = True,
    ) -> None:
        """Append one row.

        Prefer :meth:`episode` when stepping through one episode's rows in order -- it saves
        repeating ``episode_id`` at every call. Call this directly when a caller already tracks
        its own episode id per row (replaying an existing sequence of row dicts into a builder,
        for instance) and a ``with`` block would only add ceremony.

        ``entity_id`` defaults to 0, the single-trajectory case this class exists for (one
        controller, one robot, one running experiment); ``regime``/``observed`` default exactly as
        :meth:`TrajectoryLog.from_rows` defaults them, so the two stay interchangeable. Values are
        stored as given -- push does no coercion or validation of its own -- and only converted to
        the schema's NumPy dtypes inside :meth:`freeze`, the same place ``from_rows`` does its own
        conversion, so a malformed value is reported the same way it would be for any other caller
        of :class:`TrajectoryLog`, not through a second, builder-specific error path.
        """
        self._cols["entity_id"].append(entity_id)
        self._cols["episode_id"].append(episode_id)
        self._cols["t"].append(t)
        self._cols["kind"].append(kind)
        self._cols["name"].append(name)
        self._cols["value"].append(value)
        self._cols["regime"].append(regime)
        self._cols["observed"].append(observed)

    @contextmanager
    def episode(self, episode_id: int) -> Generator[EpisodeWriter]:
        """Open a push scope that stamps every row inside it with ``episode_id``.

        Yields an :class:`EpisodeWriter` bound to ``episode_id``; call :meth:`~EpisodeWriter.push`
        on it instead of repeating the id at every call. The handle stops accepting pushes the
        moment this block exits, including when the body raises: the ``finally`` that closes it
        runs either way, so an exception partway through an episode still leaves the builder
        holding exactly the rows pushed before the failure, ready for more pushes or a
        :meth:`freeze` -- the exception itself propagates to the caller unchanged, nothing here
        swallows it.
        """
        writer = EpisodeWriter(self, episode_id)
        try:
            yield writer
        finally:
            writer.close()

    def freeze(self) -> TrajectoryLog:
        """Materialise everything pushed so far into a genuine :class:`TrajectoryLog`.

        Repeatable, not a one-shot terminal operation: call it as often as you like, including
        after pushing more rows following an earlier call. That is the deliberate choice between
        the two honest options for what "appending after freeze" should mean -- refuse it
        outright, or allow it and let freeze be called again -- because a live producer's entire
        reason for existing is that it keeps running after someone has inspected a snapshot, and
        the streaming certificate this class exists to feed
        (:func:`~causalrl.ope.ipw.stream_policy_value`) is exactly the kind of thing wanted
        mid-run, not only once at the very end. Refusing further pushes after the first freeze
        would force a caller to either stop the producer just to get a certificate, or maintain a
        fresh builder per snapshot and stitch the results back together -- both defeat the point
        of having an incremental builder at all.

        Safe to call repeatedly precisely because it is not free of a copy: :class:`TrajectoryLog`
        converts every column list into a new NumPy array in its constructor, so a log returned by
        an earlier ``freeze()`` shares no storage with this builder and cannot be changed by rows
        pushed after it was returned -- each snapshot is a real, independent log, not a view.
        """
        return TrajectoryLog(self._cols, self._metadata)
