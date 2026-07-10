"""Phase 0: CausalEnvProtocol / NoiseLedger contracts + SCMCausalEnv conformance (§5.1, §5.5).

`SCMCausalEnv` is duck-typed and torch-free (it works with the real torch-backed
`StructuralCausalModel` in CI); here a numpy fake exercises the TrajectoryLog emission locally.
"""

import numpy as np
import pytest

from causalrl.data.trajectory import TrajectoryLog
from causalrl.protocols import (
    CausalEnvProtocol,
    DictNoiseLedger,
    NoiseLedger,
    SCMCausalEnv,
)


class _FakeSCM:
    """Minimal see/do-shaped SCM over numpy columns (no torch)."""

    def __init__(self, columns: dict[str, np.ndarray]) -> None:
        self._columns = columns

    def see(self, n: int, *, seed: int | None = None) -> dict[str, np.ndarray]:
        return {k: v[:n] for k, v in self._columns.items()}

    def do(self, interventions: dict[str, float]) -> "_FakeSCM":
        cols = dict(self._columns)
        n = len(next(iter(self._columns.values())))
        for k, val in interventions.items():
            cols[k] = np.full(n, float(val))
        return _FakeSCM(cols)


def _fake() -> _FakeSCM:
    return _FakeSCM({"X": np.array([0.0, 1.0, 2.0]), "Y": np.array([1.0, 3.0, 5.0])})


def test_sample_emits_trajectorylog() -> None:
    log = SCMCausalEnv(_fake()).sample(3)
    assert isinstance(log, TrajectoryLog)
    assert len(log) == 6  # 2 variables x 3 units
    assert set(log.column("name")) == {"X", "Y"}
    assert list(log.column("entity_id")) == [0, 1, 2, 0, 1, 2]
    assert list(log.values_by_name("Y")) == [1.0, 3.0, 5.0]


def test_do_pins_intervened_variable() -> None:
    log = SCMCausalEnv(_fake()).do({"X": 9.0}, 3)
    assert list(log.values_by_name("X")) == [9.0, 9.0, 9.0]


def test_conforms_to_protocol() -> None:
    env = SCMCausalEnv(_fake())
    assert isinstance(env, CausalEnvProtocol)
    assert env.noise_ledger() is None  # white-box ledger arrives in Phase 1


def test_dict_noise_ledger_draws() -> None:
    ledger = DictNoiseLedger({(0, 0): {"U": [0.5]}})
    assert isinstance(ledger, NoiseLedger)
    assert ledger.draws(0, 0) == {"U": [0.5]}
    assert ledger.draws(1, 1) is None


def test_dict_noise_ledger_posterior_is_phase1() -> None:
    ledger = DictNoiseLedger({})
    evidence = TrajectoryLog.from_rows(
        [{"entity_id": 0, "episode_id": 0, "t": 0, "kind": "obs", "name": "X", "value": 1.0}]
    )
    with pytest.raises(NotImplementedError):
        ledger.posterior(evidence)
