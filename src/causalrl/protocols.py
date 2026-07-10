"""Structural protocols for causal environments and noise (plan §5.1, §5.5).

Torch-free Protocol definitions — the contracts estimators and bounds drive — plus a duck-typed
conformance for the shipped ``StructuralCausalModel``. ``SCMCausalEnv`` never imports torch: it
calls the model's ``see``/``do`` and packs the result into a ``TrajectoryLog``, so it works with
the real torch-backed SCM and any ``see``/``do``-shaped object. Named ``CausalEnvProtocol`` avoids
colliding with the shipped Gymnasium base :class:`causalrl.envs.base.CausalEnv`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

from causalrl.data.trajectory import TrajectoryLog
from causalrl.regime import Regime


@runtime_checkable
class NoisePosterior(Protocol):
    """A posterior over exogenous noise (the Phase-1 abduction output)."""

    def sample(self, n: int, *, seed: int | None = None) -> Mapping[str, Any]: ...


@runtime_checkable
class NoiseLedger(Protocol):
    """Explicit noise handling for Level-3 queries (§5.5, I6).

    Either registered white-box draws (``draws``) or a posterior obtained by abduction
    (``posterior``, Phase 1). No counterfactual API may silently assume access to noise.
    """

    def draws(self, episode_id: int, entity_id: int) -> Mapping[str, Any] | None: ...

    def posterior(self, evidence: TrajectoryLog) -> NoisePosterior: ...


@runtime_checkable
class CausalEnvProtocol(Protocol):
    """The contract a simulator implements so every estimator/bound can drive it (§5.1)."""

    def sample(
        self, n: int, *, seed: int | None = None, regime: Regime | None = None
    ) -> TrajectoryLog: ...

    def do(
        self,
        interventions: Mapping[str, Any],
        n: int,
        *,
        seed: int | None = None,
        regime: Regime | None = None,
    ) -> TrajectoryLog: ...

    def noise_ledger(self) -> NoiseLedger | None: ...


def _to_floats(column: Any) -> list[float]:
    """Coerce a sample column (torch Tensor / numpy array / sequence) to a list of floats."""
    if hasattr(column, "tolist"):
        return [float(x) for x in column.tolist()]
    return [float(x) for x in column]


def _samples_to_log(samples: Mapping[str, Any], n: int, regime_label: str) -> TrajectoryLog:
    rows: list[dict[str, Any]] = []
    for name, column in samples.items():
        values = _to_floats(column)
        for i in range(n):
            rows.append(
                {
                    "entity_id": i,
                    "episode_id": 0,
                    "t": 0,
                    "kind": "obs",
                    "name": name,
                    "value": values[i],
                    "regime": regime_label,
                    "observed": True,
                }
            )
    return TrajectoryLog.from_rows(rows)


class SCMCausalEnv:
    """Conform a ``StructuralCausalModel`` (or any ``see``/``do`` model) to `CausalEnvProtocol`.

    Duck-typed and torch-free: ``sample`` calls ``scm.see(n, seed=...)``; ``do`` calls
    ``scm.do(interventions).see(n, ...)``, packing each into a TrajectoryLog (one entity
    per unit, single step). The white-box ``noise_ledger`` (wrapping ``ExogenousPosterior``)
    arrives in Phase 1; it returns ``None`` here.
    """

    def __init__(self, scm: Any, *, regime_label: str = "observed") -> None:
        self._scm = scm
        self._regime_label = regime_label

    def sample(
        self, n: int, *, seed: int | None = None, regime: Regime | None = None
    ) -> TrajectoryLog:
        return _samples_to_log(self._scm.see(n, seed=seed), n, self._regime_label)

    def do(
        self,
        interventions: Mapping[str, Any],
        n: int,
        *,
        seed: int | None = None,
        regime: Regime | None = None,
    ) -> TrajectoryLog:
        mutilated = self._scm.do(interventions)
        return _samples_to_log(mutilated.see(n, seed=seed), n, self._regime_label)

    def noise_ledger(self) -> NoiseLedger | None:
        return None


class DictNoiseLedger:
    """A white-box `NoiseLedger` backed by registered exogenous draws (§5.5).

    The black-box ``posterior`` path (amortized abduction) arrives in Phase 1; it raises here.
    """

    def __init__(self, draws: Mapping[tuple[int, int], Mapping[str, Any]]) -> None:
        self._draws = dict(draws)

    def draws(self, episode_id: int, entity_id: int) -> Mapping[str, Any] | None:
        return self._draws.get((episode_id, entity_id))

    def posterior(self, evidence: TrajectoryLog) -> NoisePosterior:
        raise NotImplementedError("black-box noise posterior arrives in Phase 1 (abduction)")
