"""Micro→meso causal abstraction: when does a mesoscopic model license micro-level claims?

A mean-field or population-level model of cortex is used as though intervening on it answered
questions about the underlying neurons. That is a *causal abstraction* claim, and it is not free:
it holds only when the diagram commutes —

    tau( P_micro^{do(i)} )  =  P_macro^{do(omega(i))}     for every micro intervention i

with ``tau`` mapping micro states to macro states (here: units to area firing rates) and ``omega``
mapping micro interventions to macro interventions. This is the exact-transformation condition of
Rubenstein et al. (*UAI* 2017); Beckers & Halpern (*UAI* 2019) relax it to an ``alpha``-approximate
abstraction with a bounded commutation error. This module measures that error against a simulator
where both sides can actually be computed, and returns a
:class:`~causalrl.certify.Certificate` recording which of three situations holds:

* **IDENTIFIED** — the diagram commutes within tolerance *and* the mesoscopic mean dynamics are
  stable at the equilibrium. The macro model's ``do()`` is then the causally correct object, and
  mesoscopic interventional claims transfer to the micro scale.
* **BOUNDED** — it commutes only up to a measured error ``alpha``; the macro prediction is usable
  with that error attached, not as a point claim.
* **EMPIRICAL (hedged)** — the mean dynamics are unstable or the error is large. In the
  oscillatory / synchronous regime the mesoscopic equilibrium is *not* what the micro system
  converges to, so intervening on the macro model answers a different question than intervening on
  the network. The hedge names the intervention that breaks commutation.

The stability side is the neuroscience instance of the equilibrium-vs-dynamics condition already
formalised for cyclic SCMs in this library (see ``docs/equilibrium_counterfactuals/THEORY.md``):
the margin is the spectral abscissa of the mesoscopic Jacobian, positive exactly when the
mean-field fixed point is the limit of the dynamics.

**Non-liftable interventions.** Silencing *some* units of an area has no mesoscopic counterpart:
``omega`` is undefined there, because the macro state does not resolve which units were clamped.
That is not a technicality to route around — it is the precise sense in which a population model
cannot answer a targeted-perturbation question, and :class:`AreaRateAbstraction` reports it rather
than inventing a macro intervention.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol, cast

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
from causalrl.neuro.recording import MultiScaleRecording
from causalrl.neuro.simulate import (
    CorticalNetworkSpec,
    MeanFieldAreaModel,
    SpikingCorticalSimulator,
)

__all__ = [
    "AbstractionReport",
    "AreaRateAbstraction",
    "InterventionOutcome",
    "MacroSystem",
    "MicroIntervention",
    "MicroSystem",
    "SimulatedMicroSystem",
    "abstraction_error",
    "certify_abstraction",
    "mean_field_stability_margin",
]

FloatArray = NDArray[np.float64]


def mean_field_stability_margin(jacobian: FloatArray) -> float:
    """``1 - max Re(eig(B))`` — positive iff the mesoscopic mean dynamics equilibrate.

    The mean dynamics associated with the fixed point ``r = B r + c`` are ``r' = (B - I) r + c``,
    stable exactly when every eigenvalue of ``B - I`` has negative real part. Identical in
    definition to :meth:`causalrl.experimental.cyclic.LinearCyclicSCM.stability_margin`, computed
    here without importing the experimental package.
    """
    if jacobian.size == 0:
        return 1.0
    return float(1.0 - np.max(np.linalg.eigvals(jacobian).real))


@dataclass(frozen=True)
class MicroIntervention:
    """A micro-scale perturbation: clamp each named unit to a spike probability per bin."""

    targets: Mapping[str, float]
    label: str = ""

    def name(self) -> str:
        if self.label:
            return self.label
        if not self.targets:
            return "observational"
        return ", ".join(f"do({u}={p:g})" for u, p in sorted(self.targets.items()))


class MicroSystem(Protocol):
    """A micro-scale system whose interventional macro-summary can be evaluated."""

    def outcome(self, intervention: MicroIntervention) -> Mapping[str, float]:
        """Macro-summarised outcome ``tau(P_micro^{do(i)})`` — one value per macro variable."""
        ...


class MacroSystem(Protocol):
    """A mesoscopic model that answers equilibrium interventional queries."""

    def equilibrium(self, *, do: Mapping[str, float] | None = None) -> dict[str, float]: ...


@dataclass(frozen=True)
class SimulatedMicroSystem:
    """Micro system backed by :class:`~causalrl.neuro.simulate.SpikingCorticalSimulator`.

    ``outcome`` runs the spiking network under the intervention and applies ``tau`` — by default
    the per-area mean firing rate in spikes/second, the macro variables of
    :class:`~causalrl.neuro.simulate.MeanFieldAreaModel`.
    """

    simulator: SpikingCorticalSimulator
    n_bins: int = 20000
    seed: int = 0
    burn_in: int = 500

    def recording(self, intervention: MicroIntervention) -> MultiScaleRecording:
        return self.simulator.simulate(
            self.n_bins,
            do=dict(intervention.targets) or None,
            seed=self.seed,
            burn_in=self.burn_in,
        )

    def outcome(self, intervention: MicroIntervention) -> Mapping[str, float]:
        rec = self.recording(intervention)
        return area_rates(rec)


def area_rates(recording: MultiScaleRecording) -> dict[str, float]:
    """``tau``: per-area mean firing rate in spikes/second — the canonical micro→meso map."""
    return {
        area: float(recording.population_rate(area).mean() / recording.bin_size)
        for area in recording.areas
    }


@dataclass(frozen=True)
class AreaRateAbstraction:
    """The area-rate abstraction ``(tau, omega)`` for a cortical microcircuit.

    ``tau`` is :func:`area_rates`. ``omega`` maps a micro intervention to a macro one **only when
    the intervention clamps every unit of exactly one area to the same value** — the condition
    under which the macro state determines the intervened system. Anything else (a subset of an
    area, or units spanning several areas at different values) is *not liftable*, and
    :meth:`omega` returns ``None`` instead of guessing.
    """

    spec: CorticalNetworkSpec

    def tau(self, recording: MultiScaleRecording) -> dict[str, float]:
        return area_rates(recording)

    def omega(self, intervention: MicroIntervention) -> dict[str, float] | None:
        if not intervention.targets:
            return {}
        by_area: dict[str, set[str]] = {}
        values: dict[str, set[float]] = {}
        for unit, p in intervention.targets.items():
            area = self.spec.unit_area[unit]
            by_area.setdefault(area, set()).add(unit)
            values.setdefault(area, set()).add(float(p))
        macro: dict[str, float] = {}
        for area, units in by_area.items():
            everyone = {u for u in self.spec.unit_names if self.spec.unit_area[u] == area}
            if units != everyone or len(values[area]) != 1:
                return None  # partial or heterogeneous: no mesoscopic counterpart exists
            macro[area] = next(iter(values[area])) / self.spec.bin_size
        return macro


@dataclass(frozen=True)
class InterventionOutcome:
    """One row of the commutation check: micro vs macro prediction under the same intervention."""

    intervention: str
    liftable: bool
    micro: Mapping[str, float] = field(default_factory=lambda: {})
    macro: Mapping[str, float] = field(default_factory=lambda: {})
    error: float = float("nan")  # max_a |micro_a - macro_a|, spikes/second

    def render(self) -> str:
        if not self.liftable:
            return f"{self.intervention}: NOT LIFTABLE (omega undefined)"
        return f"{self.intervention}: error={self.error:.3g} Hz"


@dataclass(frozen=True)
class AbstractionReport:
    """Commutation errors of a micro→meso abstraction over a set of interventions."""

    outcomes: tuple[InterventionOutcome, ...]
    stability_margin: float
    macro_variables: tuple[str, ...]
    n_equilibria: int = 1

    @property
    def liftable(self) -> tuple[InterventionOutcome, ...]:
        return tuple(o for o in self.outcomes if o.liftable)

    @property
    def non_liftable(self) -> tuple[InterventionOutcome, ...]:
        return tuple(o for o in self.outcomes if not o.liftable)

    @property
    def max_error(self) -> float:
        errs = [o.error for o in self.liftable]
        return float(max(errs)) if errs else 0.0

    @property
    def mean_error(self) -> float:
        errs = [o.error for o in self.liftable]
        return float(np.mean(errs)) if errs else 0.0

    def worst(self) -> InterventionOutcome | None:
        errs = self.liftable
        return max(errs, key=lambda o: o.error) if errs else None

    def render(self) -> str:
        head = (
            f"abstraction over {len(self.macro_variables)} macro variables; "
            f"stability margin={self.stability_margin:.3g}; "
            f"max error={self.max_error:.3g} Hz over {len(self.liftable)} liftable "
            f"({len(self.non_liftable)} not liftable)"
        )
        return "\n".join([head, *(f"  {o.render()}" for o in self.outcomes)])


def abstraction_error(
    micro: MicroSystem,
    macro: MacroSystem,
    interventions: Sequence[MicroIntervention],
    *,
    omega: object,
    stability_margin: float,
    macro_variables: Sequence[str],
    n_equilibria: int = 1,
) -> AbstractionReport:
    """Evaluate the commutation error of ``(tau, omega)`` over ``interventions``.

    ``omega`` is any callable mapping a :class:`MicroIntervention` to a macro intervention dict, or
    ``None`` when the intervention has no mesoscopic counterpart.
    """
    if not callable(omega):
        raise TypeError("omega must be callable")
    lift = cast("Callable[[MicroIntervention], Mapping[str, float] | None]", omega)
    rows: list[InterventionOutcome] = []
    for iv in interventions:
        macro_do = lift(iv)
        if macro_do is None:
            rows.append(InterventionOutcome(iv.name(), liftable=False))
            continue
        micro_out = dict(micro.outcome(iv))
        macro_out = macro.equilibrium(do=dict(macro_do) or None)
        err = max(
            (abs(micro_out.get(v, 0.0) - macro_out.get(v, 0.0)) for v in macro_variables),
            default=0.0,
        )
        rows.append(InterventionOutcome(iv.name(), True, micro_out, macro_out, float(err)))
    return AbstractionReport(
        tuple(rows), float(stability_margin), tuple(macro_variables), int(n_equilibria)
    )


def certify_abstraction(
    spec: CorticalNetworkSpec,
    *,
    interventions: Sequence[MicroIntervention] | None = None,
    tolerance: float = 1.0,  # spikes/second
    severe_multiple: float = 5.0,
    n_bins: int = 20000,
    seed: int = 0,
    burn_in: int = 500,
) -> tuple[Certificate, AbstractionReport]:
    """Certify the area-rate abstraction: does the mesoscopic model license micro claims?

    Runs the spiking network under each micro intervention, runs the mean-field area model under
    the lifted macro intervention, and compares. ``tolerance`` is the commutation error (in
    spikes/second) below which the abstraction counts as exact for the purpose at hand — an
    ``alpha``-abstraction with ``alpha = tolerance``.

    Returns ``(certificate, report)``; the report carries the per-intervention detail behind the
    certificate's verdict.
    """
    simulator = SpikingCorticalSimulator(spec, seed=seed)
    macro_model = MeanFieldAreaModel(spec)
    abstraction = AreaRateAbstraction(spec)
    margin = mean_field_stability_margin(macro_model.jacobian())
    if interventions is None:
        interventions = default_interventions(spec)

    micro = SimulatedMicroSystem(simulator, n_bins=n_bins, seed=seed, burn_in=burn_in)
    report = abstraction_error(
        micro,
        macro_model,
        interventions,
        omega=abstraction.omega,
        stability_margin=margin,
        macro_variables=macro_model.areas,
        n_equilibria=len(macro_model.equilibria()),
    )

    stable = margin > 0.0
    commutes = report.max_error <= tolerance
    severe = report.max_error > severe_multiple * tolerance
    unique = report.n_equilibria <= 1
    worst = report.worst()
    assumptions = (
        Assumption(
            name="tau-omega-abstraction",
            params={"tau": "area mean firing rate (Hz)", "tolerance_hz": tolerance},
            checkable=True,
            diagnostic={
                "max_error_hz": report.max_error,
                "mean_error_hz": report.mean_error,
                "n_liftable": len(report.liftable),
                "n_non_liftable": len(report.non_liftable),
            },
        ),
        Assumption(
            name="mean-field-stability",
            params={"margin": margin},
            checkable=True,
            diagnostic={"stable": stable, "n_equilibria": report.n_equilibria},
        ),
    )

    if severe or not stable:
        if not stable:
            reason = (
                f"mesoscopic mean dynamics are unstable at the equilibrium (margin={margin:.3g} "
                "<= 0): the mean-field fixed point is not what the spiking network converges to, "
                "so equilibrium do() answers a different question"
            )
        else:
            reason = (
                f"commutation fails by {report.max_error:.3g} Hz, more than {severe_multiple:g}x "
                f"the {tolerance:g} Hz tolerance: the mesoscopic model does not describe this "
                "network's interventional behaviour, and its stability margin does not reveal that"
            )
        kind, witness, hedge = (
            Kind.EMPIRICAL,
            None,
            Hedge(
                reason=reason,
                detail={
                    "stability_margin": margin,
                    "max_error_hz": report.max_error,
                    "n_equilibria": report.n_equilibria,
                    "worst_intervention": None if worst is None else worst.intervention,
                },
            ),
        )
    elif not unique:
        kind, witness, hedge = (
            Kind.BOUNDED,
            None,
            Hedge(
                reason=(
                    f"the mesoscopic model has {report.n_equilibria} equilibria: which one the "
                    "network occupies depends on its history and basin, which no plain macro SCM "
                    "represents (equilibrium-selection hedge)"
                ),
                detail={"n_equilibria": report.n_equilibria, "max_error_hz": report.max_error},
            ),
        )
    elif stable and commutes and not report.non_liftable:
        kind, witness, hedge = (
            Kind.IDENTIFIED,
            Witness(
                kind="commuting-stable-abstraction",
                detail={
                    "max_error_hz": report.max_error,
                    "stability_margin": margin,
                    "reading": (
                        "the mesoscopic equilibrium do() agrees with the spiking network's "
                        "interventional behaviour, so macro-level causal claims transfer"
                    ),
                },
            ),
            None,
        )
    elif commutes:
        kind, witness, hedge = (
            Kind.BOUNDED,
            None,
            Hedge(
                reason=(
                    f"{len(report.non_liftable)} of {len(report.outcomes)} interventions have no "
                    "mesoscopic counterpart (omega undefined): a population model cannot answer a "
                    "perturbation targeting part of an area"
                ),
                detail={"non_liftable": [o.intervention for o in report.non_liftable]},
            ),
        )
    else:
        kind, witness, hedge = (
            Kind.BOUNDED,
            None,
            Hedge(
                reason=(
                    f"abstraction commutes only up to {report.max_error:.3g} Hz, above the "
                    f"{tolerance:g} Hz tolerance; treat macro predictions as approximate"
                ),
                detail={
                    "max_error_hz": report.max_error,
                    "worst_intervention": None if worst is None else worst.intervention,
                },
            ),
        )

    claim = (
        f"area-rate mean-field model is a valid causal abstraction of the spiking network "
        f"(max commutation error {report.max_error:.3g} Hz over {len(report.liftable)} "
        f"liftable interventions)"
    )
    cert = Certificate(
        claim=claim,
        estimand=EstimandSpec(query="do", target="area-rate", domains=macro_model.areas),
        kind=kind,
        value=Interval(0.0, report.max_error),
        alpha=None,
        assumptions=assumptions,
        method="tau-omega-commutation",
        witness=witness,
        hedge=hedge,
        provenance=Provenance.create(seeds=(seed,)),
    )
    return cert, report


def default_interventions(spec: CorticalNetworkSpec) -> list[MicroIntervention]:
    """A standard battery: observational, whole-area silencing/driving, and a partial silencing.

    The partial silencing is included deliberately — it is the one that ``omega`` cannot lift, and
    seeing it reported is the point.
    """
    out = [MicroIntervention({}, "observational")]
    for area in spec.areas:
        units = [u for u in spec.unit_names if spec.unit_area[u] == area]
        out.append(MicroIntervention(dict.fromkeys(units, 0.0), f"silence({area})"))
        out.append(MicroIntervention(dict.fromkeys(units, min(0.25, 0.9)), f"drive({area})"))
        if len(units) > 1:
            half = units[: max(1, len(units) // 2)]
            out.append(MicroIntervention(dict.fromkeys(half, 0.0), f"silence(half of {area})"))
    return out
