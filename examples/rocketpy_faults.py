"""Break things, then score. Every earlier comparison on this branch ran on a benign mission.

Five architectures have now been measured here and they tied or lost, and the common cause was the
test conditions rather than the designs: the physics was known, the feedback forgiving, the actuator
perfect, the sensors noiseless and the motor lot inside +-5%. **In that regime a good closed form is
unbeatable and every wrapper around it is dead weight**, which is exactly what got measured, five
times. ``rocketpy_crew.py`` said as much in its own docstring -- its safety layer collected nothing
because nothing ever broke -- and named the missing experiment. This is that experiment.

**The scoring rule is not ours.** Inventing a score and then optimising against it is how a
benchmark quietly becomes a mirror, so this uses the Spaceport America Cup / IREC apogee-accuracy
formula, which is worth 70% of flight performance in the actual competition::

    points = 350 * (1 - |target - actual| / (target * 0.30))     floored at zero

Only the *formula* is theirs. IREC targets 10,000 or 30,000 ft AGL and this vehicle reaches roughly
2,000 m, so the target altitude is scaled to what the airframe can fly; the +-30% tolerance band and
the 350-point scale are taken as published.

**The faults are the point.** Each is a *mechanism shift* -- the world stops matching the model --
which is the one thing a closed form cannot notice about itself:

===============  ================================================================================
brake-jam        the actuator freezes part-deployed. Feedback cannot fix a lever that does not
                 move, and the closed form keeps solving for a deployment that is not happening.
sensor-bias      the altimeter reads high. The analytic controller trusts its state estimate
                 absolutely, so a wrong altitude poisons the prediction with total confidence.
drogue-fail      the drogue never deploys, invalidating the descent model the recovery decision
                 depends on.
early-burnout    motor impulse far below anything the campaigns characterized.
wind-shear       wind well beyond the profile the drift model was fitted on.
===============  ================================================================================

The prediction being tested, recorded before running: the closed form wins clean and **degrades
hardest under brake-jam and sensor-bias**, because it has no mechanism for doubting itself; the crew
should win there or nowhere. If it wins nowhere, its safety layer is dead weight and this benchmark
will have earned that conclusion rather than assumed it.

    pip install "causalrl[rocketpy]"
    python examples/rocketpy_faults.py
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

try:
    from rocketpy import Environment, Flight
except ImportError as exc:  # pragma: no cover - the example is opt-in
    raise SystemExit(
        "This example needs RocketPy: pip install 'causalrl[rocketpy]'\n"
        "RocketPy is an optional extra -- causalrl itself never imports it."
    ) from exc

from examples.rocketpy_airbrakes import (
    BRAKE_FLOOR,
    ELEVATION,
    TESTED_CEILING,
    ApogeeModel,
    FlightRecord,
    build_rocket,
)
from examples.rocketpy_autopilot import (
    MAIN_WINDOW,
    MAX_IMPACT_SPEED,
    DescentModel,
    randomized_campaigns,
)
from examples.rocketpy_baselines import CD_BRAKE_TRUE, analytic_policy, calibrate_cd
from examples.rocketpy_crew import calibrate_impact_band

TARGET_APOGEE = 1850.0
IREC_TOLERANCE = 0.30  # the competition's +-30% band
IREC_POINTS = 350.0  # the competition's apogee-accuracy allocation
# Set from the campaign, not by eye: the fastest state any brake decision was ever taken at.
# The first version guessed 200 m/s when the true figure is 231.6, so the regime cap fired on
# flights that were in fact characterized.
CHARACTERIZED_SPEED = 231.6
N_LOTS = 8
LOT_SPREAD = 0.05
SEED = 0


def irec_points(apogee: float) -> float:
    """Spaceport America Cup apogee accuracy, scaled to this vehicle's target altitude."""
    error = abs(TARGET_APOGEE - apogee)
    return float(max(0.0, IREC_POINTS * (1.0 - error / (TARGET_APOGEE * IREC_TOLERANCE))))


@dataclass(frozen=True)
class Fault:
    """One way the world stops matching the model."""

    name: str
    impulse: float = 1.0
    jam_at: float | None = None  # deployment the brake freezes at, once commanded past it
    altimeter_bias: float = 0.0  # metres the controller's altitude reads high
    drogue: bool = True
    wind: tuple[float, float] = (8.0, 14.0)  # surface and altitude wind, m/s


FAULTS = (
    Fault("nominal"),
    Fault("brake-jam", jam_at=0.15),
    Fault("sensor-bias", altimeter_bias=150.0),
    Fault("drogue-fail", drogue=False),
    Fault("early-burnout", impulse=0.82),
    Fault("wind-shear", wind=(18.0, 30.0)),
)


def environment_for(fault: Fault) -> Environment:
    environment = Environment(latitude=32.99, longitude=-106.97, elevation=ELEVATION)
    surface, aloft = fault.wind
    environment.set_atmospheric_model(
        type="custom_atmosphere",
        wind_u=[(0, surface), (4000, aloft)],
        wind_v=[(0, 0.0), (4000, 0.0)],
    )
    return environment


@dataclass
class Outcome:
    apogee: float
    drift: float
    impact_speed: float

    @property
    def safe(self) -> bool:
        return self.impact_speed <= MAX_IMPACT_SPEED

    def points(self) -> float:
        """IREC apogee points, forfeited entirely when the airframe does not survive.

        A competition flight that destroys the vehicle does not bank its apogee accuracy, and
        averaging the two would let a good number pay for a broken rocket.
        """
        return irec_points(self.apogee) if self.safe else 0.0


# A pilot is two callables: what to deploy, and whether to release the main now.
BrakeFn = Callable[[float, float], float]
ReleaseFn = Callable[[float, float], bool]


def fly(fault: Fault, lot: float, brake: BrakeFn, release: ReleaseFn) -> Outcome:
    """One flight under one fault. The pilot only ever sees what its sensors report."""
    environment = environment_for(fault)
    apogee_seen = {"value": 0.0}
    jammed: dict[str, float | None] = {"level": None}

    def controller(time_, sampling_rate, state, history, observed, interactive, *rest):
        brakes = interactive[0] if isinstance(interactive, (list, tuple)) else interactive
        true_altitude = float(state[2]) - ELEVATION
        speed = float(state[5])
        if speed > 0:
            apogee_seen["value"] = max(apogee_seen["value"], true_altitude)
        if true_altitude <= BRAKE_FLOOR or speed <= 0:
            brakes.deployment_level = 0.0
            return (time_, 0.0)

        # The pilot reads the altimeter, not the truth.
        commanded = float(brake(true_altitude + fault.altimeter_bias, speed))
        if fault.jam_at is not None:
            # Once the brake is driven past the jam point it sticks there for the rest of the
            # flight: the commanded level keeps changing and the surface does not.
            if jammed["level"] is None and commanded > fault.jam_at:
                jammed["level"] = fault.jam_at
            if jammed["level"] is not None:
                commanded = float(jammed["level"])
        brakes.deployment_level = commanded
        return (time_, commanded)

    def main_trigger(pressure: float, height: float, state: Sequence[float]) -> bool:
        if state[5] >= 0 or not (MAIN_WINDOW[0] <= height <= MAIN_WINDOW[1]):
            return False
        return release(height + fault.altimeter_bias, apogee_seen["value"])

    rocket = build_rocket(fault.impulse, controller)
    if fault.drogue:
        rocket.add_parachute("drogue", cd_s=0.9, trigger="apogee", sampling_rate=20, lag=0.5)
    rocket.add_parachute("main", cd_s=8.0, trigger=main_trigger, sampling_rate=20, lag=0.8)

    flight = Flight(
        rocket=rocket, environment=environment, rail_length=5.2, inclination=85, heading=0
    )
    return Outcome(
        apogee=float(flight.apogee) - ELEVATION,
        drift=float(np.hypot(flight.x_impact, flight.y_impact)),
        impact_speed=abs(float(flight.impact_velocity)),
    )


def analytic_pilot(cd_brake: float, main_altitude: float = 400.0) -> tuple[BrakeFn, ReleaseFn]:
    """Closed form, fixed recovery. No mechanism for doubting either half."""
    policy = analytic_policy(cd_brake)
    return (
        lambda altitude, speed: policy(altitude, speed),
        lambda altitude, _apogee: altitude <= main_altitude,
    )


def learned_pilot(ascent: ApogeeModel, descent: DescentModel) -> tuple[BrakeFn, ReleaseFn]:
    """The response-surface autopilot: learned brake, learned recovery, point estimates."""

    def brake(altitude: float, speed: float) -> float:
        grid = np.linspace(0.0, TESTED_CEILING, 40)
        errors = [abs(ascent.apogee(altitude, speed, float(d)) - TARGET_APOGEE) for d in grid]
        return float(grid[int(np.argmin(errors))])

    def release(altitude: float, apogee: float) -> bool:
        grid = np.linspace(MAIN_WINDOW[0], MAIN_WINDOW[1], 48)
        safe = [h for h in grid if descent.impact_speed(apogee, float(h)) <= MAX_IMPACT_SPEED - 1.5]
        return altitude <= (float(min(safe)) if safe else float(MAIN_WINDOW[1]))

    return brake, release


def crew_pilot(
    cd_brake: float,
    descent: DescentModel,
    band: float,
    characterized: tuple[float, float],
) -> tuple[BrakeFn, ReleaseFn]:
    """Closed-form guidance, capped out of regime; recovery on a conformal upper bound."""
    policy = analytic_policy(cd_brake)

    def bound(apogee: float, main_altitude: float) -> float:
        width = band * (4.0 if not characterized[0] <= apogee <= characterized[1] else 1.0)
        return descent.impact_speed(apogee, main_altitude) + width

    def brake(altitude: float, speed: float) -> float:
        wanted = policy(altitude, speed)
        # Outside the characterized envelope the model is extrapolating: trim, do not trust.
        ceiling = 0.25 if speed > CHARACTERIZED_SPEED else TESTED_CEILING
        return float(np.clip(wanted, 0.0, ceiling))

    def release(altitude: float, apogee: float) -> bool:
        grid = np.linspace(MAIN_WINDOW[0], MAIN_WINDOW[1], 48)
        safe = [h for h in grid if bound(apogee, float(h)) <= MAX_IMPACT_SPEED]
        return altitude <= (float(min(safe)) if safe else float(MAIN_WINDOW[1]))

    return brake, release


def breaking_point(
    make_fault: Callable[[float], Fault],
    severities: Sequence[float],
    pilots: dict[str, tuple[BrakeFn, ReleaseFn]],
    lots: Sequence[float],
) -> dict[str, float]:
    """Largest severity at which every flight still lands safely.

    A single fault magnitude cannot separate robust from lucky: analytic survives a 150 m altimeter
    bias only because its hand-picked 400 m release constant happens to leave enough slack, which
    says nothing about whether the constant was chosen well. Sweeping until each architecture breaks
    replaces one arbitrary number with the quantity actually of interest -- how much reality has to
    diverge from the model before this design kills the vehicle.
    """
    survived = dict.fromkeys(pilots, 0.0)
    for severity in severities:
        fault = make_fault(severity)
        for name, (brake, release) in pilots.items():
            outcomes = [fly(fault, float(lot), brake, release) for lot in lots]
            if all(o.safe for o in outcomes):
                survived[name] = severity
    return survived


def main() -> None:
    rng = np.random.default_rng(SEED)
    started = time.perf_counter()

    print("Calibrating the three architectures on CLEAN flights only.")
    print("(A fault the pilots were trained on would not be a fault.)\n")
    cd_hat = calibrate_cd(rng, 6)
    from examples.rocketpy_autopilot import build_environment

    clean = build_environment()
    ascent_rows, descent_rows = randomized_campaigns(clean, rng)
    ascent = ApogeeModel.fit(
        [
            FlightRecord(
                decisions=[(row[0], row[1], row[2])],
                apogee=row[3],
                impulse=1.0,
                tick_times=[],
                truncated_ticks=0,
            )
            for row in ascent_rows
        ]
    )
    descent, band, characterized = calibrate_impact_band(descent_rows)
    print(
        f"   Cd_brake={cd_hat:.3f} (true {CD_BRAKE_TRUE}); conformal impact band +{band:.2f} m/s; "
        f"characterized apogee {characterized[0]:.0f}-{characterized[1]:.0f} m"
    )
    print(f"   ({time.perf_counter() - started:.0f}s)\n")

    pilots = {
        "analytic (closed form)": analytic_pilot(cd_hat),
        "learned autopilot": learned_pilot(ascent, descent),
        "crew (bounded + veto)": crew_pilot(cd_hat, descent, band, characterized),
    }

    lots = rng.normal(size=N_LOTS) * LOT_SPREAD
    print(
        f"IREC apogee points (max {IREC_POINTS:.0f}), mean over {N_LOTS} motor lots; "
        "unsafe landings score zero.\n"
    )
    header = f"   {'fault':16s}" + "".join(f"{name:>26s}" for name in pilots)
    print(header)
    totals = dict.fromkeys(pilots, 0.0)
    for fault in FAULTS:
        cells = []
        for name, (brake, release) in pilots.items():
            outcomes = [
                fly(fault, float(lot) + (fault.impulse - 1.0), brake, release) for lot in lots
            ]
            points = float(np.mean([o.points() for o in outcomes]))
            unsafe = sum(not o.safe for o in outcomes)
            totals[name] += points
            cells.append(f"{points:18.1f} ({unsafe}/{N_LOTS})")
        print(f"   {fault.name:16s}" + "".join(f"{c:>26s}" for c in cells))

    print(f"\n   {'TOTAL':16s}" + "".join(f"{totals[n]:>26.1f}" for n in pilots))

    print("\n\nBreaking points: how far can reality diverge before the vehicle is lost?")
    print("(IREC scores apogee only, so it cannot see a recovery decision. This can.)\n")
    biases = [0.0, 50.0, 100.0, 150.0, 200.0, 250.0, 300.0, 350.0, 400.0]
    survived = breaking_point(
        lambda b: Fault(f"bias-{b:.0f}", altimeter_bias=b), biases, pilots, lots
    )
    print(f"   {'architecture':26s} {'largest altimeter bias survived':>34s}")
    for name in pilots:
        value = survived[name]
        marker = f"{value:.0f} m" if value < biases[-1] else f">= {value:.0f} m"
        print(f"   {name:26s} {marker:>34s}")
    print(f"\nTotal wall clock {time.perf_counter() - started:.0f}s.")


if __name__ == "__main__":
    main()
