"""Phase 0: columnar TrajectoryLog + ConfoundedTrajectoryDataset bridge (§5.4; acceptance #4).

The NumPy core and the bridge run everywhere; Arrow/Parquet IO is gated on a real pyarrow install
(the ``[data]`` extra in CI) via ``importorskip`` on a compiled submodule, so the empty local
namespace stub skips rather than falsely running.
"""

import pytest

from causalrl.data.dataset import ConfoundedTrajectoryDataset, Transition
from causalrl.data.trajectory import TrajectoryLog


def _rows() -> list[dict[str, object]]:
    return [
        {"entity_id": 0, "episode_id": 0, "t": 0, "kind": "obs", "name": "state", "value": 1},
        {"entity_id": 0, "episode_id": 0, "t": 0, "kind": "action", "name": "action", "value": 0},
        {"entity_id": 0, "episode_id": 0, "t": 0, "kind": "reward", "name": "reward", "value": 2.5},
    ]


def _dataset() -> tuple[ConfoundedTrajectoryDataset, list[Transition]]:
    transitions = [
        Transition(0, 1, 0.0, 1, False),
        Transition(1, 0, 2.0, 2, True),
        Transition(0, 1, 0.0, 1, False),
        Transition(1, 1, 1.0, 0, True),
    ]
    return ConfoundedTrajectoryDataset(transitions, n_states=3, n_actions=2), transitions


def test_from_rows_len_and_columns() -> None:
    log = TrajectoryLog.from_rows(_rows())
    assert len(log) == 3
    assert list(log.column("name")) == ["state", "action", "reward"]
    assert list(log.column("value")) == [1, 0, 2.5]
    assert all(bool(o) for o in log.column("observed"))
    assert list(log.column("regime")) == ["observed", "observed", "observed"]


def test_missing_column_raises() -> None:
    with pytest.raises(ValueError, match="missing columns"):
        TrajectoryLog({"entity_id": [0]})


def test_bridge_is_lossless() -> None:
    ds, transitions = _dataset()
    log = TrajectoryLog.from_confounded_dataset(ds)
    assert len(log) == 5 * len(transitions)  # state/action/reward/next_state/done per transition
    back = log.to_confounded_dataset()
    assert back.transitions == transitions
    assert (back.n_states, back.n_actions) == (3, 2)


def test_bridge_preserves_derived_statistics() -> None:
    ds, _ = _dataset()
    back = TrajectoryLog.from_confounded_dataset(ds).to_confounded_dataset()
    for s in range(ds.n_states):
        for a in range(ds.n_actions):
            assert back.behavior_propensity(s, a) == ds.behavior_propensity(s, a)
            assert back.mean_reward(s, a) == ds.mean_reward(s, a)


def test_bridge_episode_and_time_indexing() -> None:
    ds, _ = _dataset()
    log = TrajectoryLog.from_confounded_dataset(ds)
    groups = sorted(
        {(int(e), int(t)) for e, t in zip(log.column("episode_id"), log.column("t"), strict=True)}
    )
    assert groups == [(0, 0), (0, 1), (1, 0), (1, 1)]


def test_fingerprint_deterministic_and_sensitive() -> None:
    log = TrajectoryLog.from_rows(_rows())
    assert log.fingerprint() == TrajectoryLog.from_rows(_rows()).fingerprint()
    rows2 = _rows()
    rows2[0]["value"] = 99
    assert log.fingerprint() != TrajectoryLog.from_rows(rows2).fingerprint()


def test_scan_covers_all_rows_and_keeps_metadata() -> None:
    ds, _ = _dataset()
    log = TrajectoryLog.from_confounded_dataset(ds)
    batches = list(log.scan(batch_size=3))
    assert sum(len(b) for b in batches) == len(log)
    assert all(b.metadata.get("n_states") == 3 for b in batches)


def test_values_by_name() -> None:
    ds, _ = _dataset()
    log = TrajectoryLog.from_confounded_dataset(ds)
    assert list(log.values_by_name("reward")) == [0.0, 2.0, 0.0, 1.0]


def test_pivot_dense() -> None:
    index, table = TrajectoryLog.from_rows(_rows()).pivot()
    assert index == [(0, 0, 0)]
    assert table["state"].tolist() == [1]
    assert table["reward"].tolist() == [2.5]


def test_parquet_roundtrip(tmp_path: object) -> None:
    pytest.importorskip("pyarrow._parquet")  # compiled ext; skips the local namespace stub
    ds, _ = _dataset()
    log = TrajectoryLog.from_confounded_dataset(ds)
    path = tmp_path / "log.parquet"  # type: ignore[operator]
    log.to_parquet(path)
    reloaded = TrajectoryLog.from_parquet(path)
    assert reloaded.to_confounded_dataset().transitions == ds.transitions
    assert reloaded.fingerprint() == log.fingerprint()


def test_arrow_roundtrip_preserves_value_types() -> None:
    pytest.importorskip("pyarrow.lib")  # compiled ext; skips the local namespace stub
    ds, _ = _dataset()
    log = TrajectoryLog.from_confounded_dataset(ds)
    reloaded = TrajectoryLog.from_arrow(log.to_arrow())
    assert reloaded.fingerprint() == log.fingerprint()
    # int/float/bool distinctions survive the union encoding
    assert reloaded.to_confounded_dataset().transitions == ds.transitions
