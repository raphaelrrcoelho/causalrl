"""Columnar ``TrajectoryLog`` (plan §5.4; invariant I5).

The canonical long/tidy trajectory schema every estimator consumes. The in-memory core is pure
NumPy and always importable; Arrow/Parquet IO is lazy behind the optional ``[data]`` extra. A
lossless two-way bridge with the shipped :class:`~eqcert.data.dataset.ConfoundedTrajectoryDataset`
keeps the d3rlpy path working unchanged.

Schema columns: ``entity_id``, ``episode_id``, ``t`` (int64); ``kind`` and ``name`` (str);
``value`` (float | int | bool | list[float] union); ``regime`` (str); ``observed`` (bool). Log-level
metadata (e.g. ``n_states``/``n_actions``) rides alongside.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from eqcert.data.dataset import ConfoundedTrajectoryDataset, Transition

_COLUMNS = ("entity_id", "episode_id", "t", "kind", "name", "value", "regime", "observed")
_MISSING = "TrajectoryLog Arrow/Parquet IO requires pyarrow; install the 'eqcert[data]' extra"


def _as_float_list(v: Any) -> list[float]:
    """Iterate ``v`` (list/tuple/ndarray) as floats. The ``Any`` parameter discards the caller's
    narrowed element type, keeping the comprehension free of pyright ``Unknown``."""
    return [float(x) for x in v]


def _coerce_value(v: Any) -> Any:
    """Normalise a value cell to a Python primitive of the schema union."""
    if isinstance(v, bool):  # before int: bool is an int subclass
        return bool(v)
    if isinstance(v, int):
        return int(v)
    if isinstance(v, float):
        return float(v)
    if isinstance(v, np.generic):  # numpy scalar (bool_/integer/floating) -> Python scalar
        return _to_python(v)
    if isinstance(v, (list, tuple, np.ndarray)):
        return _as_float_list(v)
    return v


def _to_python(o: Any) -> Any:
    """Convert a numpy scalar to a Python scalar. The ``Any`` param discards narrowing."""
    return o.item()


def _json_default(o: Any) -> Any:
    if isinstance(o, np.generic):
        return _to_python(o)
    raise TypeError(f"not JSON-serializable: {type(o).__name__}")


def _pyarrow() -> Any:
    try:
        import pyarrow as pa  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(_MISSING) from exc
    if not hasattr(pa, "table"):  # empty namespace stub masquerading as pyarrow
        raise ImportError(_MISSING)
    return pa


def _parquet() -> tuple[Any, Any]:
    pa = _pyarrow()
    try:
        import pyarrow.parquet as pq  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError(_MISSING) from exc
    return pa, pq


class TrajectoryLog:
    """A columnar, long-format trajectory log (§5.4)."""

    def __init__(
        self, columns: Mapping[str, Any], metadata: Mapping[str, Any] | None = None
    ) -> None:
        missing = [c for c in _COLUMNS if c not in columns]
        if missing:
            raise ValueError(f"TrajectoryLog missing columns: {missing}")
        n = len(columns["entity_id"])
        for c in _COLUMNS:
            if len(columns[c]) != n:
                raise ValueError(f"column {c!r} has length {len(columns[c])}, expected {n}")
        self._entity_id: NDArray[Any] = np.asarray(columns["entity_id"], dtype=np.int64)
        self._episode_id: NDArray[Any] = np.asarray(columns["episode_id"], dtype=np.int64)
        self._t: NDArray[Any] = np.asarray(columns["t"], dtype=np.int64)
        self._kind: NDArray[Any] = np.asarray([str(x) for x in columns["kind"]], dtype=object)
        self._name: NDArray[Any] = np.asarray([str(x) for x in columns["name"]], dtype=object)
        self._value: NDArray[Any] = np.empty(n, dtype=object)
        for i, v in enumerate(columns["value"]):
            self._value[i] = _coerce_value(v)
        self._regime: NDArray[Any] = np.asarray([str(x) for x in columns["regime"]], dtype=object)
        self._observed: NDArray[Any] = np.asarray(columns["observed"], dtype=bool)
        self._metadata: dict[str, Any] = dict(metadata or {})

    def __len__(self) -> int:
        return int(self._entity_id.shape[0])

    @property
    def metadata(self) -> dict[str, Any]:
        return dict(self._metadata)

    def column(self, name: str) -> NDArray[Any]:
        arrays: dict[str, NDArray[Any]] = {
            "entity_id": self._entity_id,
            "episode_id": self._episode_id,
            "t": self._t,
            "kind": self._kind,
            "name": self._name,
            "value": self._value,
            "regime": self._regime,
            "observed": self._observed,
        }
        if name not in arrays:
            raise KeyError(name)
        return arrays[name]

    @classmethod
    def from_rows(
        cls, rows: Sequence[Mapping[str, Any]], metadata: Mapping[str, Any] | None = None
    ) -> TrajectoryLog:
        cols: dict[str, list[Any]] = {c: [] for c in _COLUMNS}
        for r in rows:
            cols["entity_id"].append(r["entity_id"])
            cols["episode_id"].append(r["episode_id"])
            cols["t"].append(r["t"])
            cols["kind"].append(r["kind"])
            cols["name"].append(r["name"])
            cols["value"].append(r["value"])
            cols["regime"].append(r.get("regime", "observed"))
            cols["observed"].append(r.get("observed", True))
        return cls(cols, metadata)

    def values_by_name(self, name: str) -> NDArray[Any]:
        """All ``value`` cells whose ``name`` equals ``name``, in row order."""
        return self._value[self._name == name]

    def scan(self, batch_size: int) -> Iterator[TrajectoryLog]:
        """Iterate the log in row batches (streaming access; metadata carried on each batch)."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        n = len(self)
        for start in range(0, n, batch_size):
            yield self._slice(start, min(start + batch_size, n))

    def _slice(self, start: int, stop: int) -> TrajectoryLog:
        cols = {c: self.column(c)[start:stop] for c in _COLUMNS}
        return TrajectoryLog(cols, self._metadata)

    def pivot(self) -> tuple[list[tuple[int, int, int]], dict[str, NDArray[Any]]]:
        """Long -> wide: sorted ``(entity_id, episode_id, t)`` keys and one array per variable name.

        Cells missing a value for a key are ``None`` (dense-case helper; last write wins).
        """
        keys = sorted(
            {
                (int(self._entity_id[i]), int(self._episode_id[i]), int(self._t[i]))
                for i in range(len(self))
            }
        )
        key_index = {k: j for j, k in enumerate(keys)}
        names = sorted({str(x) for x in self._name})
        table: dict[str, NDArray[Any]] = {
            nm: np.full(len(keys), None, dtype=object) for nm in names
        }
        for i in range(len(self)):
            k = (int(self._entity_id[i]), int(self._episode_id[i]), int(self._t[i]))
            table[str(self._name[i])][key_index[k]] = self._value[i]
        return keys, table

    def fingerprint(self) -> str:
        """A stable content hash over the columns and metadata (I8 data-fingerprint)."""
        payload = {
            "entity_id": self._entity_id.tolist(),
            "episode_id": self._episode_id.tolist(),
            "t": self._t.tolist(),
            "kind": [str(x) for x in self._kind],
            "name": [str(x) for x in self._name],
            "value": [self._value[i] for i in range(len(self))],
            "regime": [str(x) for x in self._regime],
            "observed": self._observed.tolist(),
            "metadata": self._metadata,
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=_json_default)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    # --- ConfoundedTrajectoryDataset bridge (§5.4; acceptance #4) ---

    @classmethod
    def from_confounded_dataset(cls, ds: ConfoundedTrajectoryDataset) -> TrajectoryLog:
        """Lossless bridge in: one ``(entity_id=0, episode_id, t)`` row-group per transition."""
        rows: list[dict[str, Any]] = []
        episode = 0
        t = 0
        for tr in ds.transitions:
            base: dict[str, Any] = {
                "entity_id": 0,
                "episode_id": episode,
                "t": t,
                "regime": "observed",
                "observed": True,
            }
            rows.append({**base, "kind": "obs", "name": "state", "value": int(tr.state)})
            rows.append({**base, "kind": "action", "name": "action", "value": int(tr.action)})
            rows.append({**base, "kind": "reward", "name": "reward", "value": float(tr.reward)})
            rows.append({**base, "kind": "obs", "name": "next_state", "value": int(tr.next_state)})
            rows.append({**base, "kind": "done", "name": "done", "value": bool(tr.done)})
            if tr.done:
                episode += 1
                t = 0
            else:
                t += 1
        meta: dict[str, Any] = {
            "n_states": ds.n_states,
            "n_actions": ds.n_actions,
            "source": "ConfoundedTrajectoryDataset",
        }
        return cls.from_rows(rows, meta)

    def to_confounded_dataset(self) -> ConfoundedTrajectoryDataset:
        """Lossless bridge out. Requires ``n_states``/``n_actions`` in metadata."""
        if "n_states" not in self._metadata or "n_actions" not in self._metadata:
            raise ValueError("log lacks n_states/n_actions metadata; not a dataset bridge")
        groups: dict[tuple[int, int, int], dict[str, Any]] = {}
        for i in range(len(self)):
            k = (int(self._entity_id[i]), int(self._episode_id[i]), int(self._t[i]))
            groups.setdefault(k, {})[str(self._name[i])] = self._value[i]
        transitions = [
            Transition(
                state=int(groups[k]["state"]),
                action=int(groups[k]["action"]),
                reward=float(groups[k]["reward"]),
                next_state=int(groups[k]["next_state"]),
                done=bool(groups[k]["done"]),
            )
            for k in sorted(groups)
        ]
        return ConfoundedTrajectoryDataset(
            transitions,
            n_states=int(self._metadata["n_states"]),
            n_actions=int(self._metadata["n_actions"]),
        )

    # --- Arrow/Parquet IO (lazy; optional [data] extra). value union -> typed sub-columns,
    #     because Parquet has no union type. ---

    def to_arrow(self) -> Any:
        pa = _pyarrow()
        vtype: list[str] = []
        vint: list[int | None] = []
        vfloat: list[float | None] = []
        vbool: list[bool | None] = []
        vlist: list[list[float] | None] = []
        for i in range(len(self)):
            v = self._value[i]
            tag, iv, fv, bv, lv = _encode_value(v)
            vtype.append(tag)
            vint.append(iv)
            vfloat.append(fv)
            vbool.append(bv)
            vlist.append(lv)
        table = pa.table(
            {
                "entity_id": self._entity_id.tolist(),
                "episode_id": self._episode_id.tolist(),
                "t": self._t.tolist(),
                "kind": [str(x) for x in self._kind],
                "name": [str(x) for x in self._name],
                "regime": [str(x) for x in self._regime],
                "observed": self._observed.tolist(),
                "value_type": vtype,
                "value_int": pa.array(vint, type=pa.int64()),
                "value_float": pa.array(vfloat, type=pa.float64()),
                "value_bool": pa.array(vbool, type=pa.bool_()),
                "value_list": pa.array(vlist, type=pa.list_(pa.float64())),
            }
        )
        meta = {b"eqcert_metadata": json.dumps(self._metadata).encode("utf-8")}
        return table.replace_schema_metadata(meta)

    @classmethod
    def from_arrow(cls, table: Any) -> TrajectoryLog:
        d: dict[Any, Any] = {name: table.column(name).to_pylist() for name in table.column_names}
        values: list[Any] = [
            _decode_value(
                d["value_type"][i],
                d["value_int"][i],
                d["value_float"][i],
                d["value_bool"][i],
                d["value_list"][i],
            )
            for i in range(table.num_rows)
        ]
        cols: dict[str, Sequence[Any]] = {
            "entity_id": d["entity_id"],
            "episode_id": d["episode_id"],
            "t": d["t"],
            "kind": d["kind"],
            "name": d["name"],
            "value": values,
            "regime": d["regime"],
            "observed": d["observed"],
        }
        schema_meta: Any = table.schema.metadata
        raw = schema_meta.get(b"eqcert_metadata") if schema_meta else None
        metadata: dict[str, Any] = json.loads(raw) if raw else {}
        return cls(cols, metadata)

    def to_parquet(self, path: str | os.PathLike[str]) -> None:
        _, pq = _parquet()
        pq.write_table(self.to_arrow(), str(path))

    @classmethod
    def from_parquet(cls, path: str | os.PathLike[str]) -> TrajectoryLog:
        _, pq = _parquet()
        return cls.from_arrow(pq.read_table(str(path)))

    def sorted_by_key(self) -> TrajectoryLog:
        """Return a copy with rows sorted by ``(entity_id, episode_id, t)`` (stable within a key).

        Key-contiguous ordering lets the streaming estimators join a decision's cells with an
        O(1) carry-over buffer instead of holding the whole log (plan §9).
        """
        n = len(self)
        order = np.lexsort((np.arange(n), self._t, self._episode_id, self._entity_id))
        cols = {c: self.column(c)[order] for c in _COLUMNS}
        return TrajectoryLog(cols, self._metadata)

    @classmethod
    def iter_parquet_batches(
        cls, path: str | os.PathLike[str], batch_size: int
    ) -> Iterator[TrajectoryLog]:
        """Stream a Parquet log in row batches without materialising it (plan §9; ``[data]`` extra).

        Yields one :class:`TrajectoryLog` per Arrow record batch; log-level metadata is not carried
        on the per-batch logs (the streaming estimators consume named value columns, not metadata).
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        pa, pq = _parquet()
        parquet_file = pq.ParquetFile(str(path))
        for record_batch in parquet_file.iter_batches(batch_size=batch_size):
            yield cls.from_arrow(pa.Table.from_batches([record_batch]))


def _encode_value(
    v: Any,
) -> tuple[str, int | None, float | None, bool | None, list[float] | None]:
    if isinstance(v, bool):
        return "bool", None, None, bool(v), None
    if isinstance(v, int):
        return "int", int(v), None, None, None
    if isinstance(v, float):
        return "float", None, float(v), None, None
    if isinstance(v, list):
        return "floatlist", None, None, None, _as_float_list(v)
    raise TypeError(f"unsupported value type for Arrow: {type(v).__name__}")


def _decode_value(tag: Any, iv: Any, fv: Any, bv: Any, lv: Any) -> Any:
    if tag == "bool":
        return bool(bv)
    if tag == "int":
        return int(iv)
    if tag == "float":
        return float(fv)
    if tag == "floatlist":
        return [float(x) for x in lv]
    raise ValueError(f"unknown value_type {tag!r}")
