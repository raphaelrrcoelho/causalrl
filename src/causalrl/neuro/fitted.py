"""Micro→meso abstraction on *fitted* models — the path from recordings to a certificate.

:func:`causalrl.neuro.abstraction.certify_abstraction` compares a mesoscopic model against the
spiking simulator, which means it only runs where a simulator exists. On experimental data there is
no simulator: both scales have to be *learned*. This module supplies that path.

    discover_lagged  ->  unrolled_admg  ->  fit_scm(families=PoissonGLMFit)  ->  interventions

The lag-unrolled graph is acyclic by construction — every edge points forward in time — which is
what makes a recurrent cortical circuit fittable by :func:`causalrl.fit_scm` at all. Fitting it with
:class:`~causalrl.scm.fitters.PoissonGLMFit` keeps the point-process structure: log-linear in the
parents, so a node with a dozen lagged parents stays tractable.

Two fitted SCMs are compared, one per scale:

* **micro** — one node per unit per lag, spike counts;
* **macro** — one node per area per lag, the area's total spike count per bin.

``tau`` is summation over an area's units, and ``omega`` maps "clamp every unit of area *A* to
``c``" to "clamp area *A* to ``n_A * c``" — an exact pair, because the macro variable is by
definition the sum the micro intervention pins.

**What is compared is an L-step response, not an equilibrium.** A lag-unrolled model of depth ``L``
propagates an intervention exactly ``L`` bins; clamping the target at *every* lag makes that a
sustained perturbation, and the lag-0 response is what it produces after ``L * bin_size`` seconds.
That is a real interventional quantity and the same one at both scales, so the commutation test is
meaningful — but it is a finite-horizon claim, and the certificate says so. Choose ``max_lag`` and
``bin_size`` so ``L * bin_size`` covers the timescale of the interaction you care about.

Requires the ``[torch]`` extra (``fit_scm`` does), so this module is deliberately *not* re-exported
from :mod:`causalrl.neuro`, which is otherwise pure NumPy. Import it directly.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from causalrl.certify.certificate import (
    Assumption,
    Certificate,
    EstimandSpec,
    Hedge,
    Kind,
    Provenance,
    Witness,
)
from causalrl.identification.bounds import Interval
from causalrl.neuro.abstraction import (
    AbstractionReport,
    MicroIntervention,
    abstraction_error,
)
from causalrl.neuro.citests import PoissonGLMTest
from causalrl.neuro.recording import MultiScaleRecording, RecordingError
from causalrl.neuro.timeseries import LaggedGraph, discover_lagged, lag_name, lagged_frame

__all__ = [
    "FittedMacroSystem",
    "FittedMicroSystem",
    "FittedScales",
    "area_count_columns",
    "certify_fitted_abstraction",
    "fit_lagged_scm",
    "render_outcomes",
]

FloatArray = NDArray[np.float64]


def area_count_columns(recording: MultiScaleRecording) -> dict[str, FloatArray]:
    """The macro scale: each area's **total** spike count per bin.

    A count, so :class:`~causalrl.scm.fitters.PoissonGLMFit` applies at both scales, and a sum, so
    the abstraction map is exact rather than approximate — pinning every unit of an area pins this
    quantity by definition.
    """
    if not recording.areas:
        raise RecordingError("recording carries no area labels; cannot build a macro scale")
    out: dict[str, FloatArray] = {}
    for area in recording.areas:
        idx = [recording.unit_names.index(u) for u in recording.units_in(area)]
        out[area] = recording.spikes[:, idx].sum(axis=1).astype(np.float64)
    return out


def fit_lagged_scm(
    columns: Mapping[str, FloatArray],
    variables: Sequence[str],
    *,
    max_lag: int,
    alpha: float = 1e-3,
    max_conditioning_size: int = 2,
    contemporaneous: bool = True,
    holdout: float = 0.2,
    seed: int = 0,
) -> tuple[Any, LaggedGraph, Any]:
    """Discover a lagged graph over ``variables`` and fit it as an SCM of Poisson count nodes.

    Returns ``(scm, graph, fit_report)``. Raises
    :class:`~causalrl.exceptions.NotIdentifiableError` — from :func:`causalrl.fit_scm`, unchanged —
    when the contemporaneous slice contains a ``<->`` edge: under latent common input a mechanism is
    not recoverable by regression on observed parents, and that refusal is the correct behaviour on
    array data, not an obstacle to route around.
    """
    from causalrl.scm.fit import fit_scm
    from causalrl.scm.fitters import PoissonGLMFit

    graph = discover_lagged(
        columns,
        list(variables),
        max_lag=max_lag,
        ci_test=PoissonGLMTest(alpha=alpha),
        max_conditioning_size=max_conditioning_size,
        contemporaneous=contemporaneous,
    )
    frame = lagged_frame(columns, list(variables), max_lag)
    admg = graph.unrolled_admg()
    scm = fit_scm(
        frame,
        graph=admg,  # type: ignore[arg-type]
        families=dict.fromkeys(frame, PoissonGLMFit()),
        holdout=holdout,
        seed=seed,
    )
    return scm, graph, getattr(scm, "fit_report", None)


@dataclass(frozen=True)
class _LaggedSCM:
    """A fitted lag-unrolled SCM queried as an interventional system."""

    scm: Any
    variables: tuple[str, ...]
    max_lag: int
    bin_size: float
    n_samples: int = 8000
    seed: int = 0

    def _sustained(self, do: Mapping[str, float]) -> dict[str, float]:
        """Clamp each target at *every* lag — a sustained perturbation, not a one-bin pulse."""
        return {
            lag_name(v, lag): float(c) for v, c in do.items() for lag in range(self.max_lag + 1)
        }

    def rates(self, do: Mapping[str, float] | None = None) -> dict[str, float]:
        """Mean lag-0 rate per variable (spikes/s) under an optional sustained intervention."""
        model = self.scm.do(self._sustained(do)) if do else self.scm
        draw = model.see(self.n_samples, seed=self.seed)
        return {
            v: float(np.asarray(draw[v], dtype=np.float64).mean()) / self.bin_size
            for v in self.variables
        }


@dataclass(frozen=True)
class FittedMicroSystem:
    """Micro scale: a fitted per-unit SCM, summarised to per-area rates by ``tau``."""

    engine: _LaggedSCM
    unit_area: Mapping[str, str]

    def outcome(self, intervention: MicroIntervention) -> Mapping[str, float]:
        rates = self.engine.rates(intervention.targets or None)
        summed: dict[str, float] = {}
        for unit, rate in rates.items():
            area = self.unit_area[unit]
            summed[area] = summed.get(area, 0.0) + rate
        return summed


@dataclass(frozen=True)
class FittedMacroSystem:
    """Macro scale: a fitted per-area SCM of total spike counts."""

    engine: _LaggedSCM

    def equilibrium(self, *, do: Mapping[str, float] | None = None) -> dict[str, float]:
        return self.engine.rates(do)


@dataclass(frozen=True)
class FittedScales:
    """Both fitted scales plus the graphs they came from."""

    micro_scm: Any
    macro_scm: Any
    micro_graph: LaggedGraph
    macro_graph: LaggedGraph
    unit_area: Mapping[str, str]
    areas: tuple[str, ...]
    max_lag: int
    bin_size: float

    def horizon_seconds(self) -> float:
        """How far a sustained intervention propagates in these models."""
        return self.max_lag * self.bin_size


def _omega(scales: FittedScales, counts_per_area: Mapping[str, int]) -> Any:
    """``omega``: clamping every unit of an area to ``c`` pins the area's total to ``n_A * c``."""

    def lift(intervention: MicroIntervention) -> dict[str, float] | None:
        if not intervention.targets:
            return {}
        by_area: dict[str, set[str]] = {}
        values: dict[str, set[float]] = {}
        for unit, value in intervention.targets.items():
            area = scales.unit_area[unit]
            by_area.setdefault(area, set()).add(unit)
            values.setdefault(area, set()).add(float(value))
        macro: dict[str, float] = {}
        for area, units in by_area.items():
            everyone = {u for u, a in scales.unit_area.items() if a == area}
            if units != everyone or len(values[area]) != 1:
                return None  # partial or heterogeneous: no macro counterpart exists
            macro[area] = next(iter(values[area])) * counts_per_area[area]
        return macro

    return lift


def certify_fitted_abstraction(
    recording: MultiScaleRecording,
    *,
    max_lag: int = 4,
    alpha: float = 1e-3,
    max_conditioning_size: int = 2,
    tolerance: float = 2.0,
    severe_multiple: float = 5.0,
    n_samples: int = 8000,
    drive_count: float | None = None,
    seed: int = 0,
) -> tuple[Certificate, AbstractionReport, FittedScales]:
    """Fit both scales from a recording and certify whether the macro model licenses micro claims.

    Everything is learned from ``recording``: the micro SCM over units, the macro SCM over area
    totals, and the graphs behind both. The interventions are whole-area silencing (``0``) and
    driving (``drive_count`` spikes per bin per unit), plus a partial-area silencing that ``omega``
    cannot lift — kept in deliberately, because a population model's inability to answer a
    targeted perturbation is a finding, not an inconvenience.

    Returns ``(certificate, report, scales)``.
    """
    micro_cols = recording.micro_columns()
    macro_cols = area_count_columns(recording)
    units = list(recording.unit_names)
    areas = tuple(recording.areas)

    micro_scm, micro_graph, _ = fit_lagged_scm(
        micro_cols,
        units,
        max_lag=max_lag,
        alpha=alpha,
        max_conditioning_size=max_conditioning_size,
        seed=seed,
    )
    macro_scm, macro_graph, _ = fit_lagged_scm(
        macro_cols,
        list(areas),
        max_lag=max_lag,
        alpha=alpha,
        max_conditioning_size=max_conditioning_size,
        seed=seed,
    )

    scales = FittedScales(
        micro_scm=micro_scm,
        macro_scm=macro_scm,
        micro_graph=micro_graph,
        macro_graph=macro_graph,
        unit_area=dict(recording.unit_area),
        areas=areas,
        max_lag=max_lag,
        bin_size=recording.bin_size,
    )
    counts = {a: len(recording.units_in(a)) for a in areas}

    micro = FittedMicroSystem(
        engine=_LaggedSCM(micro_scm, tuple(units), max_lag, recording.bin_size, n_samples, seed),
        unit_area=dict(recording.unit_area),
    )
    macro = FittedMacroSystem(
        engine=_LaggedSCM(macro_scm, areas, max_lag, recording.bin_size, n_samples, seed)
    )

    # A sustained clamp pins a unit at one level across every lag, so the level that matters is
    # not the per-bin maximum (1 spike in a 5 ms bin is routine) but the sustained activity the
    # recording actually contains. Compare against the distribution of rolling means over the
    # model's own horizon; a fitted log-linear mechanism extrapolates exponentially beyond it, and
    # a disagreement produced there is arithmetic, not evidence.
    window = max_lag + 1
    kernel = np.ones(window) / window
    sustained = np.stack(
        [
            np.convolve(recording.spikes[:, j].astype(np.float64), kernel, mode="valid")
            for j in range(recording.n_units)
        ]
    )
    sustained_max = float(sustained.max())
    drive = float(np.quantile(sustained, 0.999)) if drive_count is None else float(drive_count)
    unsupported: list[str] = []
    if drive > sustained_max:
        unsupported.append(
            f"drive={drive:.3g} exceeds the largest sustained level observed "
            f"over {window} bins ({sustained_max:.3g} counts/bin)"
        )

    interventions = [MicroIntervention({}, "observational")]
    for area in areas:
        members = recording.units_in(area)
        interventions.append(MicroIntervention(dict.fromkeys(members, 0.0), f"silence({area})"))
        interventions.append(MicroIntervention(dict.fromkeys(members, drive), f"drive({area})"))
        if len(members) > 1:
            half = members[: max(1, len(members) // 2)]
            interventions.append(
                MicroIntervention(dict.fromkeys(half, 0.0), f"silence(half of {area})")
            )

    report = abstraction_error(
        micro,
        macro,
        interventions,
        omega=_omega(scales, counts),
        stability_margin=float("nan"),  # not defined for a fitted finite-horizon model
        macro_variables=areas,
    )
    certificate = _certify(report, scales, tolerance, severe_multiple, seed, unsupported, drive)
    return certificate, report, scales


def _certify(
    report: AbstractionReport,
    scales: FittedScales,
    tolerance: float,
    severe_multiple: float,
    seed: int,
    unsupported: Sequence[str],
    drive: float,
) -> Certificate:
    """Turn a measured commutation report on fitted models into a certificate."""
    commutes = report.max_error <= tolerance
    severe = report.max_error > severe_multiple * tolerance
    worst = report.worst()
    horizon = scales.horizon_seconds()

    assumptions = (
        Assumption(
            name="tau-omega-abstraction",
            params={
                "tau": "area total spike count",
                "omega": "clamp all units of an area <-> clamp the area total",
                "tolerance_hz": tolerance,
            },
            checkable=True,
            diagnostic={
                "max_error_hz": report.max_error,
                "mean_error_hz": report.mean_error,
                "n_liftable": len(report.liftable),
                "n_non_liftable": len(report.non_liftable),
            },
        ),
        Assumption(
            name="fitted-mechanisms",
            params={"family": "PoissonGLMFit", "max_lag": scales.max_lag},
            checkable=True,
            diagnostic={"provenance": "fitted"},
        ),
        Assumption(
            name="intervention-support",
            params={"drive_count": drive},
            checkable=True,
            diagnostic={
                "outside_observed_support": list(unsupported),
                "reading": (
                    "a fitted log-linear mechanism extrapolates exponentially; an intervention "
                    "beyond the counts the fit saw is arithmetic, not evidence"
                ),
            },
        ),
        Assumption(
            name="finite-horizon",
            params={"horizon_seconds": horizon},
            checkable=False,
            diagnostic={
                "reading": (
                    f"a lag-{scales.max_lag} model propagates a sustained intervention "
                    f"{horizon:.3g} s; this is an L-step response, not an equilibrium"
                )
            },
        ),
    )

    if unsupported:
        kind, witness, hedge = (
            Kind.EMPIRICAL,
            None,
            Hedge(
                reason=(
                    "an intervention lies outside the support the mechanisms were fitted on "
                    f"({'; '.join(unsupported)}): the models are extrapolating, so any "
                    "disagreement between them is uninformative"
                ),
                detail={"unsupported": list(unsupported), "max_error_hz": report.max_error},
            ),
        )
    elif severe:
        kind, witness, hedge = (
            Kind.EMPIRICAL,
            None,
            Hedge(
                reason=(
                    f"the fitted area-level model misses the fitted unit-level model's "
                    f"interventional response by {report.max_error:.3g} Hz, more than "
                    f"{severe_multiple:g}x the {tolerance:g} Hz tolerance: at this horizon it is "
                    "not describing the same system"
                ),
                detail={
                    "max_error_hz": report.max_error,
                    "worst_intervention": None if worst is None else worst.intervention,
                },
            ),
        )
    elif not commutes or report.non_liftable:
        reason = (
            f"{len(report.non_liftable)} of {len(report.outcomes)} interventions have no "
            "mesoscopic counterpart (omega undefined): a population model cannot answer a "
            "perturbation targeting part of an area"
            if report.non_liftable and commutes
            else (
                f"commutes only up to {report.max_error:.3g} Hz, above the {tolerance:g} Hz "
                "tolerance; treat area-level interventional predictions as approximate"
            )
        )
        kind, witness, hedge = (
            Kind.BOUNDED,
            None,
            Hedge(
                reason=reason,
                detail={
                    "max_error_hz": report.max_error,
                    "non_liftable": [o.intervention for o in report.non_liftable],
                },
            ),
        )
    else:
        kind, witness, hedge = (
            Kind.BOUNDED,  # never IDENTIFIED: both sides are fitted, not known
            Witness(
                kind="commuting-fitted-abstraction",
                detail={
                    "max_error_hz": report.max_error,
                    "horizon_seconds": horizon,
                    "reading": (
                        "the fitted area-level model reproduces the fitted unit-level model's "
                        "interventional response within tolerance at this horizon"
                    ),
                },
            ),
            Hedge(
                reason=(
                    "both scales are fitted from one recording, so this certifies agreement "
                    "between two learned models, not against ground truth; it is BOUNDED for that "
                    "reason even when the diagram commutes"
                ),
                detail={"provenance": "fitted", "horizon_seconds": horizon},
            ),
        )

    claim = (
        f"fitted area-level model reproduces the fitted unit-level interventional response "
        f"(max commutation error {report.max_error:.3g} Hz over {len(report.liftable)} liftable "
        f"interventions, {horizon:.3g} s horizon)"
    )
    return Certificate(
        claim=claim,
        estimand=EstimandSpec(query="do", target="area-rate", domains=scales.areas),
        kind=kind,
        value=Interval(0.0, report.max_error),
        alpha=None,
        assumptions=assumptions,
        method="tau-omega-commutation-on-fitted-scms",
        witness=witness,
        hedge=hedge,
        provenance=Provenance.create(seeds=(seed,)),
    )


def render_outcomes(report: AbstractionReport) -> str:
    """One line per intervention — micro prediction, macro prediction, and the gap."""
    lines: list[str] = []
    for row in report.outcomes:
        if not row.liftable:
            lines.append(f"  {row.intervention}: NOT LIFTABLE")
            continue
        micro = ", ".join(f"{k}={v:.1f}" for k, v in sorted(row.micro.items()))
        macro = ", ".join(f"{k}={v:.1f}" for k, v in sorted(row.macro.items()))
        lines.append(f"  {row.intervention}: micro[{micro}] macro[{macro}] err={row.error:.3g} Hz")
    return "\n".join(lines)
