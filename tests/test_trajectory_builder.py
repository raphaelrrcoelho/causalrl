"""TrajectoryLogBuilder: pushes accumulate a TrajectoryLog incrementally instead of all at once."""

from __future__ import annotations

from typing import Any

import pytest

from causalrl.certify.certificate import Kind
from causalrl.data.builder import EpisodeWriter, TrajectoryLogBuilder
from causalrl.data.trajectory import TrajectoryLog
from causalrl.ope.ipw import stream_policy_value

_COLUMNS = ("entity_id", "episode_id", "t", "kind", "name", "value", "regime", "observed")


def _one_shot_rows() -> list[dict[str, Any]]:
    return [
        {"entity_id": 0, "episode_id": 7, "t": 0, "kind": "obs", "name": "altitude", "value": 1.0},
        {"entity_id": 0, "episode_id": 7, "t": 1, "kind": "obs", "name": "altitude", "value": 1.5},
        {"entity_id": 0, "episode_id": 7, "t": 2, "kind": "obs", "name": "altitude", "value": 2.0},
        {"entity_id": 0, "episode_id": 9, "t": 0, "kind": "obs", "name": "altitude", "value": 0.5},
    ]


def test_incrementally_built_log_matches_one_shot_log_from_the_same_rows() -> None:
    """Acceptance criterion: a builder session and a one-shot from_rows() call are the same log."""
    builder = TrajectoryLogBuilder()
    with builder.episode(episode_id=7) as ep:
        ep.push(kind="obs", name="altitude", value=1.0, t=0)
        ep.push(kind="obs", name="altitude", value=1.5, t=1)
        ep.push(kind="obs", name="altitude", value=2.0, t=2)
    with builder.episode(episode_id=9) as ep:
        ep.push(kind="obs", name="altitude", value=0.5, t=0)

    incremental = builder.freeze()
    one_shot = TrajectoryLog.from_rows(_one_shot_rows())

    assert len(incremental) == len(one_shot)
    assert incremental.fingerprint() == one_shot.fingerprint()
    for column in _COLUMNS:
        assert incremental.column(column).tolist() == one_shot.column(column).tolist()


def test_episode_that_raises_mid_body_leaves_builder_consistent_and_still_usable() -> None:
    builder = TrajectoryLogBuilder()

    with pytest.raises(ValueError, match="simulated failure"), builder.episode(episode_id=1) as ep:
        ep.push(kind="obs", name="x", value=1.0, t=0)
        raise ValueError("simulated failure mid-episode")

    # Exactly the one row pushed before the raise survived -- no partial or corrupted row.
    assert len(builder) == 1

    # The builder itself is unaffected by the raise: more episodes, more pushes, work normally.
    with builder.episode(episode_id=2) as ep:
        ep.push(kind="obs", name="x", value=2.0, t=0)
    assert len(builder) == 2

    log = builder.freeze()
    assert len(log) == 2
    assert log.column("episode_id").tolist() == [1, 2]


def test_episode_handle_rejects_pushes_after_its_with_block_exits_normally() -> None:
    builder = TrajectoryLogBuilder()
    with builder.episode(episode_id=3) as ep:
        ep.push(kind="obs", name="x", value=1.0, t=0)

    with pytest.raises(RuntimeError, match="already exited"):
        ep.push(kind="obs", name="x", value=2.0, t=1)
    assert len(builder) == 1  # the rejected push did not land


def test_episode_handle_rejects_pushes_after_its_with_block_exits_via_exception() -> None:
    builder = TrajectoryLogBuilder()
    escaped_handle: EpisodeWriter | None = None

    with pytest.raises(ValueError, match="boom"), builder.episode(episode_id=4) as ep:
        escaped_handle = ep
        raise ValueError("boom")

    assert escaped_handle is not None
    with pytest.raises(RuntimeError, match="already exited"):
        escaped_handle.push(kind="obs", name="x", value=1.0, t=0)
    assert len(builder) == 0  # the raise happened before any push


def test_frozen_log_feeds_stream_policy_value() -> None:
    """The coupling this class exists for: freeze() output is a valid stream_policy_value source."""
    builder = TrajectoryLogBuilder()
    rewards = [1.0, 2.0, 3.0, 4.0, 5.0]
    with builder.episode(episode_id=0) as ep:
        for t, r in enumerate(rewards):
            ep.push(kind="weight", name="weight", value=1.0, t=t)
            ep.push(kind="reward", name="reward", value=r, t=t)

    cert = stream_policy_value(builder.freeze())

    assert cert.kind is Kind.IDENTIFIED
    assert cert.hedge is None
    assert cert.value == pytest.approx(sum(rewards) / len(rewards))
    assert cert.ci is not None


def test_freeze_is_a_repeatable_snapshot_that_later_pushes_cannot_retroactively_change() -> None:
    builder = TrajectoryLogBuilder()
    with builder.episode(episode_id=0) as ep:
        ep.push(kind="obs", name="x", value=1.0, t=0)
    first = builder.freeze()

    with builder.episode(episode_id=0) as ep:
        ep.push(kind="obs", name="x", value=2.0, t=1)
    second = builder.freeze()

    # The snapshot taken before the second push is untouched by it.
    assert len(first) == 1
    assert first.column("value").tolist() == [1.0]

    # A fresh freeze() sees everything pushed so far, old and new.
    assert len(second) == 2
    assert second.column("value").tolist() == [1.0, 2.0]


def test_push_without_a_context_manager_takes_an_explicit_episode_id() -> None:
    builder = TrajectoryLogBuilder()
    builder.push(episode_id=5, kind="obs", name="x", value=1.0, t=0)
    builder.push(episode_id=5, kind="obs", name="x", value=2.0, t=1)

    log = builder.freeze()
    assert log.column("episode_id").tolist() == [5, 5]
    assert log.column("entity_id").tolist() == [0, 0]  # entity_id defaults to 0


def test_metadata_passed_at_construction_survives_freeze() -> None:
    builder = TrajectoryLogBuilder(metadata={"n_states": 3, "n_actions": 2})
    builder.push(episode_id=0, kind="obs", name="x", value=1, t=0)

    log = builder.freeze()
    assert log.metadata == {"n_states": 3, "n_actions": 2}
