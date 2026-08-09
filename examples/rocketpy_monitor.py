"""Detect that a mechanism moved, instead of being robust to it. The one lever not yet pulled.

``rocketpy_faults.py`` ended on a specific failure: the crew's conformal band could not save it from
a biased altimeter, because **conformal coverage is over the distribution it was calibrated on** and
an input fault leaves that distribution entirely. The band answers "how wrong is my model here",
never "am I actually here". Every architecture on this branch shares that blind spot -- each one
*trusts its inputs and hedges its model*, and the fault that killed them was the other way round.

This tries the remaining option: notice the sensor moved.

**Why a constant bias was the wrong fault to have chosen.** A fixed offset is zeroed on the pad by
every real flight computer, and it is also *unobservable* from altitude and speed together -- it
shifts the reported altitude and the predicted apogee by exactly the same amount, so it cancels in
every comparison an onboard model can form. ``rocketpy_faults.py`` measured breaking points against
an offset that a launch-day procedure removes for free. The realistic and detectable fault is
**drift**: a pressure sensor whose error grows during the flight, which pad-zeroing cannot remove
because it is not there at pad-zero time.

**What makes drift detectable is redundancy plus a mechanism.** Vertical speed comes from a
different sensor than altitude. Physics ties them: over any interval the altitude *change* must
equal the integral of the speed. A drifting altimeter breaks that identity while a correct one
preserves it, so the residual between reported altitude change and integrated speed is a direct
estimate of the drift rate -- and it is a statement about *which* mechanism moved, not merely that
prediction error went up. Isolating the shifted mechanism is what makes the fault correctable
rather than only survivable.

**Honest attribution.** This is analytical redundancy -- classical fault detection and isolation,
decades old in aerospace, and not an invention of this branch or of causal inference. What the
causal framing contributes is the question it puts first: *which mechanism changed?* Every earlier
architecture here asked "how uncertain am I?" and answered it correctly while dying of a question it
never asked.

    pip install "causalrl[rocketpy]"
    python examples/rocketpy_monitor.py
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

from examples.rocketpy_airbrakes import BRAKE_FLOOR, ELEVATION, build_rocket
from examples.rocketpy_autopilot import MAIN_WINDOW, MAX_IMPACT_SPEED
from examples.rocketpy_baselines import CD_BRAKE_TRUE, analytic_policy, calibrate_cd
from examples.rocketpy_faults import IREC_POINTS, IREC_TOLERANCE, TARGET_APOGEE

MAIN_RELEASE = 400.0
DRIFT_RATES = (0.0, 4.0, 8.0, 12.0, 16.0, 20.0, 25.0)  # metres per second of altimeter error
N_LOTS = 6
LOT_SPREAD = 0.05
DETECTION_THRESHOLD = 12.0  # metres of unexplained altitude before the monitor acts
SEED = 0


def irec_points(apogee: float) -> float:
    error = abs(TARGET_APOGEE - apogee)
    return float(max(0.0, IREC_POINTS * (1.0 - error / (TARGET_APOGEE * IREC_TOLERANCE))))


@dataclass
class Outcome:
    apogee: float
    impact_speed: float
    drift_estimate: float
    true_drift: float

    @property
    def safe(self) -> bool:
        return self.impact_speed <= MAX_IMPACT_SPEED

    def points(self) -> float:
        return irec_points(self.apogee) if self.safe else 0.0


class AltimeterMonitor:
    """Estimates altimeter drift from the identity that ties altitude to speed.

    Over any interval the reported altitude change must equal the integral of vertical speed. A
    drifting altimeter breaks that identity at a constant rate; an honest one does not. The running
    residual divided by elapsed time is therefore a direct estimate of the drift rate, and
    subtracting it recovers the true altitude.

    Deliberately conservative about *acting*: the estimate is only applied once the unexplained
    altitude exceeds :data:`DETECTION_THRESHOLD`, because integration noise produces a small nonzero
    residual on every flight and correcting for a fault that is not there is its own failure mode.
    """

    def __init__(self) -> None:
        self.integrated: float | None = None
        self.reported_start: float | None = None
        self.elapsed = 0.0
        self.last_time: float | None = None
        self.estimate = 0.0

    def update(self, time_: float, reported_altitude: float, speed: float) -> float:
        """Feed one tick; return the drift-corrected altitude."""
        if self.last_time is None:
            self.last_time = time_
            self.integrated = reported_altitude
            self.reported_start = reported_altitude
            return reported_altitude

        step = time_ - self.last_time
        self.last_time = time_
        self.elapsed += step
        assert self.integrated is not None and self.reported_start is not None
        self.integrated += speed * step  # dead-reckoned from the *other* sensor

        unexplained = (reported_altitude - self.reported_start) - (
            self.integrated - self.reported_start
        )
        if self.elapsed > 0.5 and abs(unexplained) > DETECTION_THRESHOLD:
            self.estimate = unexplained / self.elapsed
        return reported_altitude - self.estimate * self.elapsed


BrakeFn = Callable[[float, float], float]


def fly(drift_rate: float, lot: float, cd_brake: float, *, monitored: bool) -> Outcome:
    """One flight with an altimeter drifting at ``drift_rate`` m/s."""
    environment = Environment(latitude=32.99, longitude=-106.97, elevation=ELEVATION)
    environment.set_atmospheric_model(
        type="custom_atmosphere",
        wind_u=[(0, 8.0), (4000, 14.0)],
        wind_v=[(0, 0.0), (4000, 0.0)],
    )
    policy = analytic_policy(cd_brake)
    monitor = AltimeterMonitor()

    def controller(time_, sampling_rate, state, history, observed, interactive, *rest):
        brakes = interactive[0] if isinstance(interactive, (list, tuple)) else interactive
        true_altitude = float(state[2]) - ELEVATION
        speed = float(state[5])
        reported = true_altitude + drift_rate * float(time_)

        altitude = monitor.update(float(time_), reported, speed) if monitored else reported
        if true_altitude <= BRAKE_FLOOR or speed <= 0:
            brakes.deployment_level = 0.0
            return (time_, 0.0)
        level = float(policy(altitude, speed))
        brakes.deployment_level = level
        return (time_, level)

    def main_trigger(pressure: float, height: float, state: Sequence[float]) -> bool:
        if state[5] >= 0 or not (MAIN_WINDOW[0] <= height <= MAIN_WINDOW[1]):
            return False
        # The recovery decision reads the same altimeter, corrected or not.
        offset = monitor.estimate * monitor.elapsed if monitored else 0.0
        return (height + drift_rate * 20.0 - offset) <= MAIN_RELEASE

    rocket = build_rocket(1.0 + lot, controller)
    rocket.add_parachute("drogue", cd_s=0.9, trigger="apogee", sampling_rate=20, lag=0.5)
    rocket.add_parachute("main", cd_s=8.0, trigger=main_trigger, sampling_rate=20, lag=0.8)
    flight = Flight(
        rocket=rocket, environment=environment, rail_length=5.2, inclination=85, heading=0
    )
    return Outcome(
        apogee=float(flight.apogee) - ELEVATION,
        impact_speed=abs(float(flight.impact_velocity)),
        drift_estimate=monitor.estimate,
        true_drift=drift_rate,
    )


def main() -> None:
    rng = np.random.default_rng(SEED)
    started = time.perf_counter()
    print("Calibrating the closed form on clean flights (no drift).")
    cd_hat = calibrate_cd(rng, 6)
    print(
        f"   Cd_brake={cd_hat:.3f} (true {CD_BRAKE_TRUE}) ({time.perf_counter() - started:.0f}s)\n"
    )

    lots = rng.normal(size=N_LOTS) * LOT_SPREAD
    print("Altimeter drift: the fault pad-zeroing cannot remove and differences can detect.")
    print(f"IREC apogee points (max {IREC_POINTS:.0f}); unsafe landings score zero.\n")
    print(f"   {'drift m/s':>10s} {'blind':>22s} {'monitored':>22s} {'rate est.':>12s}")
    for rate in DRIFT_RATES:
        blind = [fly(rate, float(lot), cd_hat, monitored=False) for lot in lots]
        watched = [fly(rate, float(lot), cd_hat, monitored=True) for lot in lots]
        blind_points = float(np.mean([o.points() for o in blind]))
        watched_points = float(np.mean([o.points() for o in watched]))
        blind_unsafe = sum(not o.safe for o in blind)
        watched_unsafe = sum(not o.safe for o in watched)
        estimate = float(np.mean([o.drift_estimate for o in watched]))
        print(
            f"   {rate:10.1f} {blind_points:14.1f} ({blind_unsafe}/{N_LOTS})"
            f" {watched_points:14.1f} ({watched_unsafe}/{N_LOTS}) {estimate:11.2f}"
        )
    print(f"\nTotal wall clock {time.perf_counter() - started:.0f}s.")


if __name__ == "__main__":
    main()
