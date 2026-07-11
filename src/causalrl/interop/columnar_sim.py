"""Generic columnar-simulator adapter (plan §10; invariant I5).

The single contract any simulator needs to support causalrl: **emit the columnar ``TrajectoryLog``
schema**. A simulator that can produce rows (``entity_id`` / ``episode_id`` / ``t`` / ``kind`` /
``name`` / ``value`` / ...) then plugs into every estimator, bound, and streaming certificate — no
bespoke integration, and causalrl never depends on the simulator.

:class:`ColumnarSimulator` is the reference implementation: wrap a row-emitting ``sample`` callable
(and, optionally, an interventional ``do`` callable and a :class:`NoiseLedger`) into an object that
conforms to :class:`~causalrl.protocols.CausalEnvProtocol`. :func:`check_conformance` validates that
a would-be simulator actually emits a well-formed log. Public simulators are supported this way as
worked examples, never as dependencies.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any

from causalrl.data.trajectory import TrajectoryLog
from causalrl.exceptions import CausalInterfaceUnavailableError
from causalrl.protocols import CausalEnvProtocol, NoiseLedger
from causalrl.regime import Regime

__all__ = ["ColumnarSimulator", "check_conformance", "simulator_from_callables"]

RowBatch = Iterable[Mapping[str, Any]]
SampleFn = Callable[[int, int | None], RowBatch]
DoFn = Callable[[Mapping[str, Any], int, int | None], RowBatch]


class ColumnarSimulator:
    """Reference :class:`~causalrl.protocols.CausalEnvProtocol` adapter over row-emitting callables.

    ``sample_fn(n, seed)`` returns an iterable of row mappings (the ``TrajectoryLog`` schema); the
    adapter wraps them into a :class:`TrajectoryLog`. ``do_fn(interventions, n, seed)`` is optional:
    without it, :meth:`do` raises :class:`CausalInterfaceUnavailableError` (honest: the simulator
    exposes only observational sampling). ``ledger`` optionally supplies a white-box
    :class:`NoiseLedger` for counterfactuals.
    """

    def __init__(
        self,
        sample_fn: SampleFn,
        do_fn: DoFn | None = None,
        *,
        metadata: Mapping[str, Any] | None = None,
        ledger: NoiseLedger | None = None,
    ) -> None:
        self._sample_fn = sample_fn
        self._do_fn = do_fn
        self._metadata: dict[str, Any] = dict(metadata or {})
        self._ledger = ledger

    def sample(
        self, n: int, *, seed: int | None = None, regime: Regime | None = None
    ) -> TrajectoryLog:
        return TrajectoryLog.from_rows(list(self._sample_fn(n, seed)), self._metadata)

    def do(
        self,
        interventions: Mapping[str, Any],
        n: int,
        *,
        seed: int | None = None,
        regime: Regime | None = None,
    ) -> TrajectoryLog:
        if self._do_fn is None:
            raise CausalInterfaceUnavailableError(
                "this simulator implements only observational sample(); provide a do_fn for do()"
            )
        return TrajectoryLog.from_rows(list(self._do_fn(interventions, n, seed)), self._metadata)

    def noise_ledger(self) -> NoiseLedger | None:
        return self._ledger


def simulator_from_callables(
    sample_fn: SampleFn,
    do_fn: DoFn | None = None,
    *,
    metadata: Mapping[str, Any] | None = None,
    ledger: NoiseLedger | None = None,
) -> ColumnarSimulator:
    """Build a :class:`ColumnarSimulator` from a sample callable (and optional do/ledger)."""
    return ColumnarSimulator(sample_fn, do_fn, metadata=metadata, ledger=ledger)


def check_conformance(sim: CausalEnvProtocol, *, n: int = 16, seed: int = 0) -> dict[str, Any]:
    """Validate that ``sim`` emits a well-formed ``TrajectoryLog``; return diagnostics or raise.

    Calls ``sim.sample(n)`` and checks it returns a :class:`TrajectoryLog`, then probes ``do`` (a
    do-nothing intervention) to record whether interventions are supported. Returns the observed row
    count, the distinct value names, and whether ``do`` / a noise ledger are available — a quick
    conformance report for a new simulator integration.
    """
    log = sim.sample(n, seed=seed)
    # Runtime guard: the protocol types this as TrajectoryLog; a non-conforming stand-in may not.
    if not isinstance(log, TrajectoryLog):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise TypeError(f"sample() must return a TrajectoryLog, got {type(log).__name__}")
    names = sorted({str(x) for x in log.column("name")})
    supports_do = True
    try:
        sim.do({}, 1, seed=seed)
    except CausalInterfaceUnavailableError:
        supports_do = False
    return {
        "n_rows": len(log),
        "names": names,
        "supports_do": supports_do,
        "has_noise_ledger": sim.noise_ledger() is not None,
    }
