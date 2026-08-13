"""A flight-data commons: transport what is shared, then ask what-if about your own flight.

Two ideas that only work together, aimed at the constraint that actually binds amateur and
collegiate rocketry: **a team gets one to three flights a year**, and every one of them is a
different vehicle.

**The commons (transportability).** Hundreds of flights happen across dozens of teams every season
and none of the data pools, because everyone's airframe differs. But not every mechanism differs.
Coast drag is the same physics on every vehicle; motor and mass mechanisms are not.
:func:`~causalrl.localize_mechanism_shift` tests each mechanism for invariance across teams and
returns the selection set of the ones that moved, which is exactly what
:func:`~causalrl.identify_transport` consumes. A team with four flights can then borrow the
mechanisms it shares with everyone else and estimate only the ones that are genuinely its own.

**The what-if (counterfactual abduction).** After a flight the debrief is an argument: *did we miss
apogee because the motor was hot, or because the brakes underperformed?* Nobody can answer it,
because you cannot re-fly that flight. :func:`~causalrl.counterfactual_expectation` can: abduct the
exogenous noise that this particular flight realised, then re-run the model under a different
deployment holding that noise fixed. Not a fresh simulation -- a counterfactual conditioned on what
actually happened that day. OpenRocket re-simulating from scratch throws exactly that away.

**Why they need each other.** A counterfactual is only as good as the model doing the abducting, and
four flights do not fit a model worth abducting with. The commons supplies the model; abduction
turns it into an answer about *your* flight. Neither half is useful alone.

**And why this is checkable here.** In the field a counterfactual can never be verified -- that is
the whole problem. In a simulator it can: the motor lot is ours to set, so the flight can be flown
again with the same lot and a different brake setting, and the counterfactual compared against what
truly would have happened. That validation is the point of running this in RocketPy at all.

    pip install "causalrl[rocketpy]"
    python examples/rocketpy_commons.py
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from causalrl import CausalGraph, fit_scm, localize_mechanism_shift

try:
    from rocketpy import Environment, Flight, SolidMotor
    from rocketpy import Rocket as RocketPyRocket
except ImportError as exc:  # pragma: no cover - the example is opt-in
    raise SystemExit(
        "This example needs RocketPy: pip install 'causalrl[rocketpy]'\n"
        "RocketPy is an optional extra -- causalrl itself never imports it."
    ) from exc

BURN_TIME = 3.0
BASE_THRUST = ((0.0, 0.0), (0.1, 1600.0), (2.6, 1500.0), (BURN_TIME, 0.0))
BRAKE_FLOOR = 700.0
CEILING = 0.75
SEED = 0


@dataclass(frozen=True)
class Team:
    """One team's vehicle and launch site. Airframes differ; the physics does not."""

    name: str
    mass: float
    radius: float
    elevation: float
    thrust_scale: float
    flights: int
    drag: float = 0.5  # airframe Cd; the other half of the ballistic coefficient


TEAMS = (
    Team("alpha", mass=14.0, radius=0.0635, elevation=1400.0, thrust_scale=1.00, flights=24),
    Team("bravo", mass=19.5, radius=0.0785, elevation=1400.0, thrust_scale=1.35, flights=24),
    # The team this example is about: a new airframe, a different site, and four flights.
    Team("charlie", mass=16.5, radius=0.0700, elevation=900.0, thrust_scale=1.15, flights=4),
)
NEWCOMER = TEAMS[-1]

# The falsification check. TEAMS above differ in mass and radius but land within ~10% of each
# other on ballistic coefficient A/m -- 2.88e-4, 3.16e-4, 2.97e-4 -- which is precisely the
# quantity coast physics depends on, so their apogee mechanisms really are near-invariant and an
# empty selection set is the *correct* answer. That makes it worthless as evidence that the test
# works: a test with no power returns the same empty set. DELTA has A/m = 1.00e-3, three times the
# others, so its mechanism genuinely differs and the test must say so or it is not measuring
# anything.
DELTA = Team("delta", mass=10.0, radius=0.1000, elevation=1400.0, thrust_scale=1.10, flights=24)


def build(team: Team, impulse: float, controller) -> RocketPyRocket:
    thrust = [(t, f * team.thrust_scale * impulse) for t, f in BASE_THRUST]
    motor = SolidMotor(
        thrust_source=thrust,
        dry_mass=1.6,
        dry_inertia=(0.12, 0.12, 0.002),
        nozzle_radius=0.033,
        grain_number=4,
        grain_density=1815,
        grain_outer_radius=0.033,
        grain_initial_inner_radius=0.015,
        grain_initial_height=0.12,
        grain_separation=0.005,
        grains_center_of_mass_position=0.0,
        center_of_dry_mass_position=0.0,
        nozzle_position=-0.25,
        burn_time=BURN_TIME,
        throat_radius=0.011,
        coordinate_system_orientation="nozzle_to_combustion_chamber",
    )
    rocket = RocketPyRocket(
        radius=team.radius,
        mass=team.mass,
        inertia=(6.3, 6.3, 0.034),
        power_off_drag=team.drag,
        power_on_drag=team.drag,
        center_of_mass_without_motor=0.0,
        coordinate_system_orientation="tail_to_nose",
    )
    rocket.add_motor(motor, position=-1.25)
    rocket.add_nose(length=0.55, kind="von karman", position=1.28)
    rocket.add_trapezoidal_fins(4, root_chord=0.12, tip_chord=0.06, span=0.11, position=-1.05)
    rocket.add_tail(
        top_radius=team.radius, bottom_radius=team.radius * 0.69, length=0.06, position=-1.19
    )
    rocket.add_air_brakes(
        drag_coefficient_curve=lambda deployment, mach: 1.2 * deployment,
        controller_function=controller,
        sampling_rate=10,
        clamp=True,
        reference_area=None,
        override_rocket_drag=False,
    )
    return rocket


_environments: dict[float, Environment] = {}


def environment_at(elevation: float) -> Environment:
    if elevation not in _environments:
        env = Environment(latitude=32.99, longitude=-106.97, elevation=elevation)
        env.set_atmospheric_model(type="standard_atmosphere")
        _environments[elevation] = env
    return _environments[elevation]


def fly(team: Team, impulse: float, deployment: float) -> tuple[float, float]:
    """Fly once at a held deployment; return (speed at the first brake decision, apogee AGL)."""
    seen: dict[str, float] = {}

    def controller(time_, sampling_rate, state, history, observed, interactive, *rest):
        brakes = interactive[0] if isinstance(interactive, (list, tuple)) else interactive
        altitude, speed = state[2] - team.elevation, state[5]
        active = altitude > BRAKE_FLOOR and speed > 0
        brakes.deployment_level = deployment if active else 0.0
        if active:
            seen.setdefault("speed", float(speed))
        return (time_, brakes.deployment_level)

    flight = Flight(
        rocket=build(team, impulse, controller),
        environment=environment_at(team.elevation),
        rail_length=5.2,
        inclination=85,
        heading=0,
        terminate_on_apogee=True,
    )
    return seen.get("speed", 0.0), float(flight.apogee) - team.elevation


@dataclass
class Campaign:
    """One team's logs. ``impulse`` is recorded because this is a simulator, not because a team
    would know it -- it is used only to *validate* counterfactuals, never to fit them."""

    speed: np.ndarray
    deployment: np.ndarray
    apogee: np.ndarray
    impulse: np.ndarray

    def columns(self) -> dict[str, np.ndarray]:
        return {"speed": self.speed, "deployment": self.deployment, "apogee": self.apogee}


def fly_campaign(team: Team, rng: np.random.Generator) -> Campaign:
    speeds, deployments, apogees, impulses = [], [], [], []
    for _ in range(team.flights):
        impulse = 1.0 + 0.05 * float(np.clip(rng.normal(), -2.0, 2.0))
        deployment = float(rng.uniform(0.0, CEILING))
        speed, apogee = fly(team, impulse, deployment)
        speeds.append(speed)
        deployments.append(deployment)
        apogees.append(apogee)
        impulses.append(impulse)
    return Campaign(np.array(speeds), np.array(deployments), np.array(apogees), np.array(impulses))


GRAPH = CausalGraph(
    directed_edges=[("speed", "apogee"), ("deployment", "apogee")],
    nodes=["speed", "deployment", "apogee"],
)


def binned(
    columns: Mapping[str, np.ndarray], edges: Mapping[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """Integer-code the columns: the invariance test is the discrete one.

    Binning is a real modelling choice, not a formality -- the test then asks whether the *binned*
    mechanism is invariant, which is a coarser question than the continuous one. Shared edges across
    teams are essential, or every mechanism looks shifted because the bins mean different things.
    """
    return {name: np.digitize(columns[name], edges[name]) for name in columns}


def shared_edges(campaigns: Sequence[Campaign], bins: int = 4) -> dict[str, np.ndarray]:
    pooled = {
        "speed": np.concatenate([c.speed for c in campaigns]),
        "deployment": np.concatenate([c.deployment for c in campaigns]),
        "apogee": np.concatenate([c.apogee for c in campaigns]),
    }
    return {
        name: np.quantile(values, np.linspace(0, 1, bins + 1)[1:-1])
        for name, values in pooled.items()
    }


ABDUCTION_ATOL = 10.0
ABDUCTION_DRAWS = 200_000


def counterfactual_apogee(scm, flight: tuple[float, float, float], deployment: float) -> float:
    """What this flight's apogee would have been under a different deployment.

    ``evidence`` is the flight as flown, so abduction recovers the exogenous draw this flight
    actually realised -- the hot or cold motor nobody measured -- and the intervention re-runs the
    model holding it fixed.

    **Why this calls ``scm.counterfactual`` rather than ``counterfactual_expectation``.** Abduction
    here is rejection sampling: draw exogenous noise, keep the draws whose factual evaluation
    matches the evidence within ``atol``. The default ``atol=1e-6`` asks three *continuous*
    quantities to match to a micrometre, which has probability zero and raises
    ``RealizabilityError`` -- correctly. ``counterfactual_expectation`` does not expose ``atol``;
    the method does.

    **``atol`` is not a free knob.** It is the conditioning width, and the estimate moves with it::

        atol    1.0 ->     9 matched, 1940.9 m
        atol   10.0 ->   780 matched, 1931.1 m
        atol   25.0 ->  2843 matched, 1896.1 m
        atol  100.0 -> 11109 matched, 1892.2 m

    Loosening it stops conditioning on *this* flight and starts conditioning on a neighbourhood of
    flights, which is a different query wearing the same name. 10 m is chosen because the estimate
    has substantially converged by there; the draw count, not the window, pays for the match rate.

    Rejection over three continuous dimensions also scales badly, and two of those dimensions buy
    nothing: ``speed`` and ``deployment`` are roots, so conditioning on them discards draws without
    informing the only noise worth abducting, which is the apogee mechanism's. For invertible
    additive-noise mechanisms that abduction is analytic -- ``U = apogee - f(speed, deployment)`` --
    and sampling for it is a generic implementation paying a specific cost.
    """
    speed, held, apogee = flight
    drawn = scm.counterfactual(
        {"speed": speed, "deployment": held, "apogee": apogee},
        {"deployment": deployment},
        ABDUCTION_DRAWS,
        seed=SEED,
        atol=ABDUCTION_ATOL,
    )
    return float(np.asarray(drawn["apogee"]).mean())


def main() -> None:
    rng = np.random.default_rng(SEED)
    started = time.perf_counter()

    print("1. Three teams fly. Different airframes, different sites, wildly different budgets.")
    campaigns = {}
    for team in TEAMS:
        campaigns[team.name] = fly_campaign(team, rng)
        print(
            f"   {team.name:8s} mass {team.mass:4.1f} kg  radius {team.radius:.4f} m  "
            f"site {team.elevation:.0f} m  {team.flights:2d} flights"
        )
    print(f"   ({time.perf_counter() - started:.0f}s)\n")

    print("2. Which mechanisms are shared, and which belong to one team?")
    edges = shared_edges(list(campaigns.values()))
    report = localize_mechanism_shift(
        {name: binned(c.columns(), edges) for name, c in campaigns.items()},
        graph=GRAPH,
        alpha=0.05,
    )
    selection = set(report.selection)
    for node in GRAPH.nodes:
        verdict = "SHIFTED (team-specific)" if node in selection else "invariant (transportable)"
        print(f"   {node:12s} {verdict}")
    print(f"   selection set = {sorted(selection)}  -> what identify_transport would route")
    ratio = {t.name: t.drag * np.pi * t.radius**2 / t.mass for t in TEAMS}
    print(
        "   ballistic coefficient Cd*A/m: "
        + ", ".join(f"{n} {v:.2e}" for n, v in ratio.items())
        + " -- within ~10%, so invariance is the correct answer here.\n"
    )

    print("2b. Falsification: a vehicle whose mechanism genuinely differs must be flagged.")
    delta_campaign = fly_campaign(DELTA, rng)
    with_delta = dict(campaigns)
    with_delta[DELTA.name] = delta_campaign
    delta_edges = shared_edges(list(with_delta.values()))
    delta_report = localize_mechanism_shift(
        {name: binned(c.columns(), delta_edges) for name, c in with_delta.items()},
        graph=GRAPH,
        alpha=0.05,
    )
    flagged = sorted(delta_report.selection)
    # Cd*A/m, matching `ratio` above. Dividing a bare A/m by a Cd*A/m ratio inflated the printed
    # multiple by 1/Cd and contradicted the label two lines up.
    delta_ratio = DELTA.drag * np.pi * DELTA.radius**2 / DELTA.mass
    print(f"   delta Cd*A/m = {delta_ratio:.2e} ({delta_ratio / ratio['alpha']:.1f}x alpha)")
    print(f"   selection set with delta included = {flagged}")
    print(
        "   "
        + (
            "apogee flagged: the test has power, so the empty set above was a real negative."
            if "apogee" in flagged
            else "apogee NOT flagged: the test lacks power here, so the empty set above proves "
            "nothing."
        )
        + "\n"
    )

    print(f"3. {NEWCOMER.name} has {NEWCOMER.flights} flights. Fit alone, or borrow the commons?")
    newcomer = campaigns[NEWCOMER.name]
    solo = fit_scm(newcomer.columns(), graph=GRAPH)
    pooled_columns = {
        key: np.concatenate([c.columns()[key] for c in campaigns.values()])
        for key in ("speed", "deployment", "apogee")
    }
    commons = fit_scm(pooled_columns, graph=GRAPH)
    print("   solo   : fitted on 4 flights from one airframe")
    print("   commons: fitted on all 52, sharing the mechanisms the test called invariant\n")

    print("4. The debrief question, and the answer nobody in the field can check.")
    probe_deployments = (0.0, 0.35, CEILING)
    print(f"   {'flight':>7s} {'held':>6s} {'asked':>6s} {'TRUE':>9s} {'solo':>9s} {'commons':>9s}")
    solo_errors, commons_errors = [], []
    for index in range(min(3, NEWCOMER.flights)):
        flight = (
            float(newcomer.speed[index]),
            float(newcomer.deployment[index]),
            float(newcomer.apogee[index]),
        )
        impulse = float(newcomer.impulse[index])
        for asked in probe_deployments:
            if abs(asked - flight[1]) < 0.05:
                continue
            # Ground truth: re-fly THIS flight -- same motor lot -- with the counterfactual brake.
            _, truth = fly(NEWCOMER, impulse, asked)
            solo_value = counterfactual_apogee(solo, flight, asked)
            commons_value = counterfactual_apogee(commons, flight, asked)
            solo_errors.append(abs(solo_value - truth))
            commons_errors.append(abs(commons_value - truth))
            print(
                f"   {index:7d} {flight[1]:6.2f} {asked:6.2f} {truth:8.1f}m "
                f"{solo_value:8.1f}m {commons_value:8.1f}m"
            )
    print(
        f"\n   mean |error| vs ground truth:  solo {np.mean(solo_errors):.1f} m   "
        f"commons {np.mean(commons_errors):.1f} m"
    )
    print(f"\nTotal wall clock {time.perf_counter() - started:.0f}s.")


if __name__ == "__main__":
    main()
