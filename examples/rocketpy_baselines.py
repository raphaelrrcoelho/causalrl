"""How the learned agent fares against the autopilots a real team would actually fly.

The honest baseline for an air-brake controller is not another learned agent. It is the
**drag-limited coast prediction** every real flight computer runs: integrating ``v dv/dh = -g -
k v^2`` from the current state to zero velocity gives a closed form for the remaining altitude,

    h_gain = ln(1 + k v^2 / g) / (2k),      k = rho * Cd * A / (2m)

and the controller picks the deployment whose predicted apogee equals the target. Checked against
RocketPy with the true drag coefficient it lands within 9-17 m over the whole envelope -- an order
of magnitude better than the learned agent in ``rocketpy_autopilot.py``, using **zero flights**.

That deserves to be said plainly: *for a system whose physics you know, write down the physics.* A
learned model that needs 120 flights to be worse than a closed form is not an advance, and an
example that hid the closed form to make its agent look good would be worthless.

**Where calibration matters, and where it does not.** ``k`` contains ``Cd``, and the brake's drag
coefficient is what nobody knows before flying a new geometry -- it is the thing a characterization
campaign exists to measure. The obvious expectation is that guessing it wrong ruins the flight. It
does not, and the reason is worth more than the expectation was: **a closed-loop controller is
almost indifferent to the parameter.** Flying with ``Cd`` wrong by 2x costs nothing measurable
(16.9 m against the oracle's 17.1 m), because the controller re-measures ``(altitude, speed)`` every
tick and re-solves -- a wrong ``Cd`` changes only the path taken to the target, never the target
converged on, since the true dynamics supply the correction the model got wrong.

Freeze the first command and the picture inverts. Open-loop, the 2x error costs **84.1 m** of median
apogee against the oracle's 14.9 m, and six flights of calibration recover it entirely (12.8 m). So
estimation earns its keep exactly where feedback cannot rescue it: in decisions committed once and
not revisited. That is the same regime the rest of this library addresses -- an offline policy
shipped against logs, where there is no next tick to correct on.

The comparison below therefore runs at *equal knowledge*, not equal formulas, and then repeats it
without feedback:

===============  ==========================================================================
analytic-true    the closed form with the true ``Cd``. An oracle: available only after you
                 have already characterized the brake, which is the whole problem.
analytic-guess   the same closed form with ``Cd`` guessed wrong by 2x -- the actual state of
                 knowledge before the first flight of a new brake.
learned-cd       the closed form with ``Cd`` *fitted* from flights. Known structure, unknown
                 parameter: one number to estimate instead of a whole response surface.
causal-search    ``rocketpy_autopilot``'s route -- a black-box response surface over
                 (altitude, speed, deployment) searched under a deadline.
===============  ==========================================================================

    pip install "causalrl[rocketpy]"
    python examples/rocketpy_baselines.py
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence

import numpy as np
from numpy.typing import NDArray

try:
    from rocketpy import Flight
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
    build_environment,
    build_rocket,
    searching_policy,
)

TARGET_APOGEE = 1850.0
G = 9.81
MASS = 15.6  # airframe plus motor dry mass, kg
AREA = float(np.pi * 0.0635**2)  # reference area, m^2
RHO = 1.0  # representative density over the coast band; a flight computer reads a table
CD_AIRFRAME = 0.5
CD_BRAKE_TRUE = 1.2  # what build_rocket actually applies: cd = 1.2 * deployment
CD_BRAKE_GUESS = 0.6  # a plausible pre-flight guess for a new geometry, wrong by 2x

LOT_SPREAD = 0.05
LOT_CLIP = 2.0
N_EVAL = 14
N_CALIBRATION = 6  # flights the learned-Cd controller is allowed
N_SURFACE = 60  # flights the black-box response surface is allowed
SEED = 0

_environment = None


def environment():
    global _environment
    if _environment is None:
        _environment = build_environment()
    return _environment


def coast_gain(speed: float, cd_total: float) -> float:
    """Altitude still to be gained coasting from ``speed`` against total drag ``cd_total``."""
    k = RHO * cd_total * AREA / (2.0 * MASS)
    return float(np.log1p(k * speed * speed / G) / (2.0 * k))


def predicted_apogee(altitude: float, speed: float, deployment: float, cd_brake: float) -> float:
    return altitude + coast_gain(speed, CD_AIRFRAME + cd_brake * deployment)


def analytic_policy(cd_brake: float) -> Callable[[float, float], float]:
    """Pick the deployment whose predicted apogee equals the target.

    Predicted apogee is strictly decreasing in deployment, so a bisection is exact to tolerance and
    needs no search budget worth reporting -- which is itself part of why the closed form is hard
    to beat when its parameters are right.
    """

    def policy(altitude: float, speed: float) -> float:
        if predicted_apogee(altitude, speed, 0.0, cd_brake) <= TARGET_APOGEE:
            return 0.0
        if predicted_apogee(altitude, speed, TESTED_CEILING, cd_brake) >= TARGET_APOGEE:
            return TESTED_CEILING
        low, high = 0.0, TESTED_CEILING
        for _ in range(24):
            middle = 0.5 * (low + high)
            if predicted_apogee(altitude, speed, middle, cd_brake) > TARGET_APOGEE:
                low = middle
            else:
                high = middle
        return 0.5 * (low + high)

    return policy


def open_loop(policy: Callable[[float, float], float]) -> Callable[[float, float], float]:
    """Commit to the deployment computed at the first decision point and hold it.

    The control that isolates *why* a miscalibrated closed form still flies well. A closed-loop
    controller re-measures ``(altitude, speed)`` every tick and re-solves, so a wrong ``Cd`` only
    changes the path it takes to the target, not the target it converges on -- the true dynamics
    supply the correction the model got wrong. Freezing the first command removes that feedback and
    lets the parameter error reach the apogee, which is the cost the closed-loop arm never pays.
    """
    committed: dict[str, float] = {}

    def frozen(altitude: float, speed: float) -> float:
        if "level" not in committed:
            committed["level"] = float(policy(altitude, speed))
        return committed["level"]

    return frozen


def fly(impulse: float, policy: Callable[[float, float], float]) -> tuple[float, float, float]:
    """Fly once under ``policy(altitude, speed) -> deployment``.

    Returns apogee and the first commanded state, which is what the calibration fit needs.
    """
    first: dict[str, float] = {}

    def controller(time_, sampling_rate, state, history, observed, interactive, *rest):
        brakes = interactive[0] if isinstance(interactive, (list, tuple)) else interactive
        altitude, speed = state[2] - ELEVATION, state[5]
        if altitude > BRAKE_FLOOR and speed > 0:
            level = float(policy(altitude, speed))
            first.setdefault("altitude", altitude)
            first.setdefault("speed", speed)
            first.setdefault("deployment", level)
        else:
            level = 0.0
        brakes.deployment_level = level
        return (time_, level)

    flight = Flight(
        rocket=build_rocket(impulse, controller),
        environment=environment(),
        rail_length=5.2,
        inclination=85,
        heading=0,
        terminate_on_apogee=True,
    )
    return (
        float(flight.apogee) - ELEVATION,
        first.get("speed", 0.0),
        first.get("deployment", 0.0),
    )


def calibrate_cd(rng: np.random.Generator, n_flights: int) -> float:
    """Fit the one unknown in the closed form: the brake's drag coefficient.

    Known structure, unknown parameter. Each flight inverts the coast formula for the total ``Cd``
    that would have produced the apogee observed, and the brake's contribution is read off by least
    squares against the deployment held. One number, so a handful of flights suffices -- which is
    the whole argument for pinning the mechanism you know instead of learning a surface.
    """
    deployments, totals = [], []
    for _ in range(n_flights):
        impulse = 1.0 + LOT_SPREAD * float(np.clip(rng.normal(), -LOT_CLIP, LOT_CLIP))
        level = float(rng.uniform(0.0, TESTED_CEILING))
        apogee, speed, held = fly(impulse, lambda _a, _v, level=level: level)
        altitude = BRAKE_FLOOR  # the band the brake acts over starts here
        gain = apogee - altitude
        # Invert h_gain = ln(1 + k v^2/g)/(2k) for k, then read Cd off k.
        candidates = np.linspace(1e-6, 0.02, 20000)
        predicted = np.log1p(candidates * speed * speed / G) / (2.0 * candidates)
        k_hat = float(candidates[int(np.argmin(np.abs(predicted - gain)))])
        totals.append(2.0 * MASS * k_hat / (RHO * AREA))
        deployments.append(held)
    design = np.column_stack([np.ones(len(deployments)), np.asarray(deployments)])
    slope = float(np.linalg.lstsq(design, np.asarray(totals), rcond=None)[0][1])
    return max(slope, 0.0)


def _recording_controller(
    level: float, decisions: list[tuple[float, float, float]]
) -> Callable[..., tuple[float, float]]:
    """Build a fixed-deployment controller bound to its own ``level`` and log.

    A factory rather than a closure over the loop variables, for two reasons that pull the same
    way. Late binding would apply the last flight's deployment to every record; and binding them as
    keyword-only defaults instead -- the other obvious fix -- pushes the signature to nine
    parameters, which RocketPy rejects outright, since ``_Controller`` accepts only 6, 7 or 8.
    """

    def controller(time_, sampling_rate, state, history, observed, interactive, *rest):
        brakes = interactive[0] if isinstance(interactive, (list, tuple)) else interactive
        altitude, speed = state[2] - ELEVATION, state[5]
        active = altitude > BRAKE_FLOOR and speed > 0
        brakes.deployment_level = level if active else 0.0
        if active:
            decisions.append((altitude, speed, level))
        return (time_, brakes.deployment_level)

    return controller


def surface_policy(rng: np.random.Generator, n_flights: int) -> Callable[[float, float], float]:
    """The autopilot's route: a black-box response surface, searched at decision time."""
    records: list[FlightRecord] = []
    for _ in range(n_flights):
        impulse = 1.0 + LOT_SPREAD * float(np.clip(rng.normal(), -LOT_CLIP, LOT_CLIP))
        level = float(rng.uniform(0.0, TESTED_CEILING))
        decisions: list[tuple[float, float, float]] = []

        flight = Flight(
            rocket=build_rocket(impulse, _recording_controller(level, decisions)),
            environment=environment(),
            rail_length=5.2,
            inclination=85,
            heading=0,
            terminate_on_apogee=True,
        )
        records.append(
            FlightRecord(
                decisions=decisions,
                apogee=float(flight.apogee) - ELEVATION,
                impulse=impulse,
                tick_times=[],
                truncated_ticks=0,
            )
        )
    import examples.rocketpy_airbrakes as airbrakes

    airbrakes.TARGET_APOGEE = TARGET_APOGEE
    searcher = searching_policy(ApogeeModel.fit(records), rounds=8, candidates=16)
    return lambda altitude, speed: searcher(altitude, speed, None)[0]


def evaluate(
    make_policy: Callable[[], Callable[[float, float], float]], lots: NDArray[np.float64]
) -> Sequence[float]:
    """Fly every lot under a *freshly built* policy.

    A factory rather than a policy because :func:`open_loop` carries per-flight state: reusing one
    instance would freeze the first lot's command and apply it to all the others, which would look
    like a catastrophic open-loop result caused entirely by the harness.
    """
    return [abs(fly(1.0 + LOT_SPREAD * float(u), make_policy())[0] - TARGET_APOGEE) for u in lots]


def main() -> None:
    rng = np.random.default_rng(SEED)
    started = time.perf_counter()
    lots = np.clip(rng.normal(size=N_EVAL), -LOT_CLIP, LOT_CLIP)

    print(
        f"Target {TARGET_APOGEE:.0f} m AGL, {N_EVAL} motor lots, same lots for every autopilot.\n"
    )

    print(f"Calibrating the closed form's one unknown from {N_CALIBRATION} flights...")
    cd_learned = calibrate_cd(rng, N_CALIBRATION)
    print(
        f"   fitted Cd_brake = {cd_learned:.3f}  (true {CD_BRAKE_TRUE}, "
        f"pre-flight guess {CD_BRAKE_GUESS})"
    )
    print(f"Fitting the black-box response surface from {N_SURFACE} flights...")
    searched = surface_policy(rng, N_SURFACE)
    print(f"   done ({time.perf_counter() - started:.0f}s)\n")

    autopilots: list[tuple[str, int, Callable[[], Callable[[float, float], float]]]] = [
        ("analytic-true (oracle)", 0, lambda: analytic_policy(CD_BRAKE_TRUE)),
        ("analytic-guess (Cd 2x off)", 0, lambda: analytic_policy(CD_BRAKE_GUESS)),
        ("learned-Cd (pinned physics)", N_CALIBRATION, lambda: analytic_policy(cd_learned)),
        ("causal-search (black box)", N_SURFACE, lambda: searched),
    ]
    print(f"   {'autopilot':30s} {'flights':>8s} {'mean |err|':>11s} {'median':>9s} {'worst':>9s}")
    for name, cost, make_policy in autopilots:
        errors = np.array(evaluate(make_policy, lots))
        print(
            f"   {name:30s} {cost:8d} {errors.mean():9.1f} m {np.median(errors):7.1f} m "
            f"{errors.max():7.1f} m"
        )
    print("\n   Same controllers open-loop -- first command frozen, no feedback:")
    for name, cd_brake in (
        ("analytic-true", CD_BRAKE_TRUE),
        ("analytic-guess (Cd 2x off)", CD_BRAKE_GUESS),
        ("learned-Cd", cd_learned),
    ):
        errors = np.array(evaluate(lambda cd=cd_brake: open_loop(analytic_policy(cd)), lots))
        print(
            f"   {name:30s} {'-':>8s} {errors.mean():9.1f} m {np.median(errors):7.1f} m "
            f"{errors.max():7.1f} m"
        )
    print(f"\nTotal wall clock {time.perf_counter() - started:.0f}s.")


if __name__ == "__main__":
    main()
