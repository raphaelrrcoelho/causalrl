"""Phase 4 §10: generic columnar-simulator adapter — the emit-a-TrajectoryLog contract (numpy)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from causalrl.certify.certificate import Certificate
from causalrl.data.trajectory import TrajectoryLog
from causalrl.estimate.streaming import stream_policy_value
from causalrl.exceptions import CausalInterfaceUnavailableError
from causalrl.interop.columnar_sim import (
    ColumnarSimulator,
    check_conformance,
    simulator_from_callables,
)


def _sample_fn(n: int, seed: int | None) -> list[dict[str, Any]]:
    """A toy simulator emitting an off-policy log: one importance weight + reward per entity."""
    rng = np.random.default_rng(seed)
    w = rng.uniform(0.5, 2.0, size=n)
    y = rng.standard_normal(n) + 1.0
    rows: list[dict[str, Any]] = []
    for i in range(n):
        base = {"entity_id": i, "episode_id": 0, "t": 0}
        rows.append({**base, "kind": "w", "name": "weight", "value": float(w[i])})
        rows.append({**base, "kind": "r", "name": "reward", "value": float(y[i])})
    return rows


def _do_fn(interventions: dict[str, Any], n: int, seed: int | None) -> list[dict[str, Any]]:
    return _sample_fn(n, seed)  # a trivial do() that ignores the intervention (toy)


def test_sample_emits_trajectory_log() -> None:
    sim = simulator_from_callables(_sample_fn, metadata={"source": "toy"})
    log = sim.sample(8, seed=0)
    assert isinstance(log, TrajectoryLog)
    assert len(log) == 16  # two rows per entity
    assert log.metadata["source"] == "toy"


def test_do_without_do_fn_raises() -> None:
    sim = ColumnarSimulator(_sample_fn)
    with pytest.raises(CausalInterfaceUnavailableError, match="do_fn"):
        sim.do({"a": 1}, 4, seed=0)


def test_do_with_do_fn_returns_log() -> None:
    sim = ColumnarSimulator(_sample_fn, _do_fn)
    log = sim.do({"a": 1}, 4, seed=0)
    assert isinstance(log, TrajectoryLog) and len(log) == 8


def test_noise_ledger_passthrough() -> None:
    assert ColumnarSimulator(_sample_fn).noise_ledger() is None
    sentinel: Any = object()
    assert ColumnarSimulator(_sample_fn, ledger=sentinel).noise_ledger() is sentinel


def test_check_conformance_reports_diagnostics() -> None:
    obs_only = simulator_from_callables(_sample_fn)
    report = check_conformance(obs_only, n=5)
    assert report["n_rows"] == 10
    assert report["names"] == ["reward", "weight"]
    assert report["supports_do"] is False
    assert report["has_noise_ledger"] is False

    full = simulator_from_callables(_sample_fn, _do_fn, ledger=object())
    report2 = check_conformance(full, n=5)
    assert report2["supports_do"] is True
    assert report2["has_noise_ledger"] is True


def test_check_conformance_rejects_non_log() -> None:
    class _Bad:
        def sample(self, n, *, seed=None, regime=None):
            return "not a log"

        def do(self, interventions, n, *, seed=None, regime=None):  # pragma: no cover
            raise CausalInterfaceUnavailableError("no")

        def noise_ledger(self):  # pragma: no cover
            return None

    with pytest.raises(TypeError, match="TrajectoryLog"):
        check_conformance(_Bad())


def test_simulator_feeds_streaming_certificate() -> None:
    """A conforming simulator's log flows straight into a streaming certificate kernel."""
    sim = simulator_from_callables(_sample_fn)
    cert = stream_policy_value(sim.sample(2_000, seed=1), weight="weight", reward="reward")
    assert isinstance(cert, Certificate)
    assert cert.value is not None  # positivity holds -> an identified IS value, not a hedge
