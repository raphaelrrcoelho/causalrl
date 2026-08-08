"""One causal agent flies the whole mission: every in-flight decision the vehicle exposes.

``examples/rocketpy_airbrakes.py`` modulates one lever and asks what may be claimed about it. This
one hands the *whole flight* to a single
:class:`~causalrl.agents.interventional.InterventionalAgent` and lets it fly, from rail departure
to touchdown, with nothing scripted in between.

**What "the whole flight" means, precisely.** RocketPy simulates a passively stable vehicle: there
is no thrust-vector control, no fin actuation, no attitude command. The complete set of in-flight
authority it exposes is

* ``Rocket.add_air_brakes`` -- a ``controller_function`` setting ``deployment_level`` in ``[0, 1]``;
* ``Rocket.add_parachute`` -- a ``trigger(pressure, height, state) -> bool`` per recovery device.

:class:`RocketPilot` owns all of it. Nothing else decides anything. The claim is not that the agent
steers the rocket -- it cannot, and neither can anything else in RocketPy -- but that every choice
the vehicle admits is the agent's, taken live, under a wall-clock budget, from a model it learned
from flight logs.

**Why this needs an agent rather than a schedule.** The two levers are coupled through the mission
score and pull against each other:

* braking lowers apogee toward the target *and* shortens the descent, reducing wind drift;
* releasing the main low reduces drift further but raises the impact speed sharply -- 150 m gives
  17.8 m/s where 300 m gives 6.0 m/s;
* so the best main-release altitude depends on the apogee the brakes actually achieved, which
  depends on a motor lot nobody measured until the vehicle was already flying.

A fixed schedule has to commit to all of that before the rail. The agent does not.

**The phase structure is the type.** :class:`~causalrl.InterventionSpace` is built per decision, so
each phase of flight *is* a different space:

===================  ===========================================================
under thrust         ``{}`` -- nothing is manipulable; the brakes have no authority
coasting up          ``{"deployment": Continuous(0.0, ceiling)}``
descending on drogue ``{"release_main": Discrete((0.0, 1.0))}``
under main           ``{}`` -- every decision has been made
===================  ===========================================================

An empty space is not "do nothing", it is "there is nothing here to decide", and the agent returns
the empty intervention -- the observational regime.

    pip install "causalrl[rocketpy]"
    python examples/rocketpy_autopilot.py
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from causalrl import Continuous, Deadline, Discrete, Intervention, InterventionSpace
from causalrl.agents.anytime import AnytimeInterventionSearch
from causalrl.agents.interventional import InterventionalAgent

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
    RBFEncoder,
    build_rocket,
)

# --- Mission -----------------------------------------------------------------------------------
TARGET_APOGEE = 1850.0  # metres AGL
MAX_IMPACT_SPEED = 8.0  # m/s -- above this the airframe is damaged; a hard constraint, not a cost
MAIN_WINDOW = (120.0, 900.0)  # altitudes at which releasing the main is physically possible
SAMPLING_RATE = 10.0  # Hz; the pilot has 100 ms per decision
SEED = 0

# --- Campaigns ---------------------------------------------------------------------------------
N_ASCENT = 60
N_DESCENT = 60
LOT_SPREAD = 0.12  # motor impulse dispersion; the confounder of the sister example


def build_environment() -> Environment:
    """Standard atmosphere with a real headwind, so descent time costs you landing accuracy."""
    environment = Environment(latitude=32.99, longitude=-106.97, elevation=ELEVATION)
    environment.set_atmospheric_model(
        type="custom_atmosphere",
        wind_u=[(0, 8.0), (4000, 14.0)],
        wind_v=[(0, 0.0), (4000, 0.0)],
    )
    return environment


@dataclass
class Mission:
    """What one flight achieved, and what the pilot did to achieve it."""

    apogee: float
    drift: float
    impact_speed: float
    main_altitude: float | None
    # (altitude, vertical speed, deployment) at each brake decision, as they actually occurred.
    # Synthesising these from nominal values instead is what made the first ascent model useless:
    # a state column that never varies cannot teach a model what the state does.
    states: list[tuple[float, float, float]] = field(default_factory=list)
    tick_times: list[float] = field(default_factory=list)
    truncated: int = 0

    @property
    def safe(self) -> bool:
        return self.impact_speed <= MAX_IMPACT_SPEED

    def score(self) -> float:
        """Mission score in ``[0, 1]``: apogee accuracy and landing accuracy, zero if unsafe.

        Bounded, because the certificate layer's sensitivity bounds assume a bounded outcome. The
        safety term is a gate rather than a weight: a flight that breaks the airframe did not
        half-succeed, and averaging it against a good apogee would say otherwise.
        """
        if not self.safe:
            return 0.0
        apogee_term = max(0.0, 1.0 - abs(self.apogee - TARGET_APOGEE) / 400.0)
        drift_term = max(0.0, 1.0 - self.drift / 2000.0)
        return float(0.5 * apogee_term + 0.5 * drift_term)


class _Scaled:
    """Min-max scaled named features -- the inner encoder the RBF basis sits on."""

    def __init__(self, keys: Sequence[str], lows: NDArray[np.float64], highs: NDArray[np.float64]):
        self.keys = tuple(keys)
        self._lows = lows
        self._scale = np.maximum(highs - lows, 1e-9)
        self.dim = len(self.keys)

    def encode(self, observation: Mapping[str, Any]) -> NDArray[np.float64]:
        raw = np.array([float(observation[k]) for k in self.keys])
        return (raw - self._lows) / self._scale


def _fit_rbf(
    inputs: NDArray[np.float64],
    targets: NDArray[np.float64],
    keys: Sequence[str],
    *,
    centers: int = 5,
    ridge: float = 1e-4,
) -> tuple[RBFEncoder, NDArray[np.float64]]:
    """Ridge on Gaussian RBF features -- the shape both mission models need."""
    lows, highs = inputs.min(axis=0), inputs.max(axis=0)
    inner = _Scaled(keys, lows, highs)
    axis = np.linspace(0.0, 1.0, centers)
    grid = np.array(np.meshgrid(*[axis] * len(keys))).reshape(len(keys), -1).T
    encoder = RBFEncoder(inner, grid, bandwidth=0.35)
    design = np.array(
        [encoder.encode(dict(zip(keys, row, strict=True))) for row in inputs], dtype=np.float64
    )
    gram = design.T @ design + ridge * np.eye(design.shape[1])
    return encoder, np.linalg.solve(gram, design.T @ targets).astype(np.float64)


class AscentModel:
    """Apogee reached from holding a deployment level, given the current climb state."""

    def __init__(self, encoder: RBFEncoder, weights: NDArray[np.float64]) -> None:
        self._encoder, self._weights = encoder, weights

    @classmethod
    def fit(cls, missions: Sequence[tuple[float, float, float, float]]) -> AscentModel:
        data = np.array(missions, dtype=np.float64)
        encoder, weights = _fit_rbf(
            data[:, :3], data[:, 3], ("altitude", "vertical_speed", "deployment")
        )
        return cls(encoder, weights)

    def apogee(self, altitude: float, vertical_speed: float, deployment: float) -> float:
        return float(
            self._encoder.encode(
                {
                    "altitude": altitude,
                    "vertical_speed": vertical_speed,
                    "deployment": deployment,
                }
            )
            @ self._weights
        )


class DescentModel:
    """Drift and impact speed from releasing the main at a given altitude, given the apogee."""

    def __init__(
        self,
        encoder: RBFEncoder,
        drift_weights: NDArray[np.float64],
        speed_weights: NDArray[np.float64],
    ) -> None:
        self._encoder = encoder
        self._drift, self._speed = drift_weights, speed_weights

    @classmethod
    def fit(cls, missions: Sequence[tuple[float, float, float, float]]) -> DescentModel:
        data = np.array(missions, dtype=np.float64)
        keys = ("apogee", "main_altitude")
        encoder, drift_weights = _fit_rbf(data[:, :2], data[:, 2], keys)
        design = np.array(
            [encoder.encode({"apogee": r[0], "main_altitude": r[1]}) for r in data],
            dtype=np.float64,
        )
        gram = design.T @ design + 1e-4 * np.eye(design.shape[1])
        speed_weights = np.linalg.solve(gram, design.T @ data[:, 3]).astype(np.float64)
        return cls(encoder, drift_weights, speed_weights)

    def _features(self, apogee: float, main_altitude: float) -> NDArray[np.float64]:
        return self._encoder.encode({"apogee": apogee, "main_altitude": main_altitude})

    def drift(self, apogee: float, main_altitude: float) -> float:
        return float(self._features(apogee, main_altitude) @ self._drift)

    def impact_speed(self, apogee: float, main_altitude: float) -> float:
        return float(self._features(apogee, main_altitude) @ self._speed)


class RocketPilot(InterventionalAgent):
    """One agent, the whole flight.

    :meth:`act` is called at every decision point of every phase with the space that phase admits.
    The pilot never asks which phase it is in: the space it was handed *is* that information, which
    is the point of building the space per decision rather than once per mission.

    ``margin`` is the safety headroom the pilot keeps below :data:`MAX_IMPACT_SPEED`. It exists
    because the descent model is a fitted regression with error, and a constraint enforced at a
    model's point estimate is violated about half the time it binds.
    """

    def __init__(
        self,
        ascent: AscentModel,
        descent: DescentModel,
        *,
        margin: float = 1.5,
        rounds: int = 8,
        candidates: int = 16,
    ) -> None:
        self.ascent, self.descent = ascent, descent
        self.margin = margin
        self._rounds, self._candidates = rounds, candidates
        self.truncated = 0
        self.decisions = 0

    def _release_altitude(self, apogee: float) -> float:
        """Lowest main-release altitude whose predicted impact speed still clears the margin.

        Lower is better for drift and worse for impact speed, so the constrained optimum is the
        boundary. Scanning is honest here: the constraint is a cliff, and a gradient step across a
        cliff lands on the wrong side of it.
        """
        limit = MAX_IMPACT_SPEED - self.margin
        grid = np.linspace(MAIN_WINDOW[0], MAIN_WINDOW[1], 64)
        safe = [h for h in grid if self.descent.impact_speed(apogee, float(h)) <= limit]
        return float(min(safe)) if safe else float(MAIN_WINDOW[1])

    def act(
        self,
        observation: Mapping[str, Any],
        *,
        space: InterventionSpace,
        deadline: Deadline | None = None,
    ) -> Intervention:
        self.decisions += 1
        if not space.variables:
            return {}  # nothing to decide here -- the observational regime

        if "deployment" in space.variables:
            search = AnytimeInterventionSearch(
                self._ascent_value,
                rounds=self._rounds,
                candidates_per_round=self._candidates,
                seed=SEED,
            )
            chosen = search.act(observation, space=space, deadline=deadline)
            if not search.last_search.exhausted:
                self.truncated += 1
            return chosen

        # Descent: release the main once we have fallen to the altitude the model licenses.
        altitude = float(observation["altitude"])
        target = self._release_altitude(float(observation["apogee"]))
        return {"release_main": 1.0 if altitude <= target else 0.0}

    def update(
        self, observation: Mapping[str, Any], intervention: Intervention, reward: float
    ) -> None:
        """No-op: this pilot's models come from the randomized campaigns, not from in-flight reward.

        Spelled out rather than inherited because there is no reward *during* a flight to learn
        from -- apogee, drift and impact speed are known only once the mission is over, by which
        time every decision has been taken. Refitting happens between flights, not within one.
        """

    def _ascent_value(
        self, observation: Mapping[str, Any], intervention: Mapping[str, Any]
    ) -> float:
        predicted = self.ascent.apogee(
            float(observation["altitude"]),
            float(observation["vertical_speed"]),
            float(intervention["deployment"]),
        )
        return -abs(predicted - TARGET_APOGEE)


def fly_mission(
    environment: Environment,
    impulse: float,
    pilot: RocketPilot | None,
    *,
    fixed_deployment: float | None = None,
    fixed_main: float | None = None,
    budget: float | None = None,
) -> Mission:
    """Fly one mission. With ``pilot``, every decision is the agent's; otherwise both are scripted.

    The two levers reach RocketPy through different callbacks -- the brakes through the air-brake
    ``controller_function``, the main through the parachute ``trigger`` -- so the pilot is invoked
    from two places with two different spaces, exactly as the phase table in the module docstring
    describes.
    """
    record = Mission(apogee=0.0, drift=0.0, impact_speed=0.0, main_altitude=None)
    apogee_estimate = {"value": TARGET_APOGEE}

    ascent_space = InterventionSpace.create({"deployment": Continuous(0.0, TESTED_CEILING)})
    descent_space = InterventionSpace.create({"release_main": Discrete((0.0, 1.0))})
    empty = InterventionSpace.create({})

    def brake_controller(
        time_: float,
        sampling_rate: float,
        state: Sequence[float],
        state_history: Sequence[Sequence[float]],
        observed_variables: Sequence[Any],
        interactive_objects: Any,
        sensors: Any = None,
        environment_: Any = None,
    ) -> tuple[float, float]:
        air_brakes = (
            interactive_objects[0]
            if isinstance(interactive_objects, (list, tuple))
            else interactive_objects
        )
        altitude, vertical_speed = float(state[2]) - ELEVATION, float(state[5])
        if vertical_speed > 0:
            apogee_estimate["value"] = max(apogee_estimate["value"], altitude)

        under_thrust = altitude <= BRAKE_FLOOR or vertical_speed <= 0
        if pilot is None:
            level = 0.0 if under_thrust else float(fixed_deployment or 0.0)
        else:
            started = time.perf_counter()
            action = pilot.act(
                {"altitude": altitude, "vertical_speed": vertical_speed},
                space=empty if under_thrust else ascent_space,
                deadline=Deadline.after(budget) if budget is not None else None,
            )
            # Time only the decisions that searched. RocketPy calls this callback for the whole
            # flight, and an empty space returns in nanoseconds; averaging those in would report a
            # latency the searching path never achieves.
            if not under_thrust:
                record.tick_times.append(time.perf_counter() - started)
            level = float(action.get("deployment", 0.0))

        air_brakes.deployment_level = level
        if not under_thrust:
            record.states.append((altitude, vertical_speed, level))
        return (time_, level)

    def main_trigger(pressure: float, height: float, state: Sequence[float]) -> bool:
        if state[5] >= 0 or not (MAIN_WINDOW[0] <= height <= MAIN_WINDOW[1]):
            return False
        if pilot is None:
            release = height <= float(fixed_main or MAIN_WINDOW[0])
        else:
            action = pilot.act(
                {"altitude": height, "apogee": apogee_estimate["value"]},
                space=descent_space,
                deadline=Deadline.after(budget) if budget is not None else None,
            )
            release = bool(action.get("release_main", 0.0) >= 0.5)
        if release and record.main_altitude is None:
            record.main_altitude = height
        return release

    rocket = build_rocket(impulse, brake_controller)
    rocket.add_parachute("drogue", cd_s=0.9, trigger="apogee", sampling_rate=20, lag=0.5)
    rocket.add_parachute("main", cd_s=8.0, trigger=main_trigger, sampling_rate=20, lag=0.8)

    flight = Flight(
        rocket=rocket, environment=environment, rail_length=5.2, inclination=85, heading=0
    )
    record.apogee = float(flight.apogee) - ELEVATION
    record.drift = float(np.hypot(flight.x_impact, flight.y_impact))
    record.impact_speed = abs(float(flight.impact_velocity))
    if pilot is not None:
        record.truncated = pilot.truncated
    return record


def randomized_campaigns(
    environment: Environment, rng: np.random.Generator
) -> tuple[list[tuple[float, float, float, float]], list[tuple[float, float, float, float]]]:
    """Fly both experiments the pilot's two models need, with each lever randomized independently.

    Randomized rather than logged: a schedule chosen in reaction to the motor lot would put the lot
    on a back-door path into both models, which is the failure ``rocketpy_airbrakes.py`` is about.
    Here the levers are drawn independently of the lot, so each model is fit on its own experiment.
    """
    ascent: list[tuple[float, float, float, float]] = []
    descent: list[tuple[float, float, float, float]] = []

    for _ in range(N_ASCENT):
        impulse = 1.0 + LOT_SPREAD * float(rng.normal())
        level = float(rng.uniform(0.0, TESTED_CEILING))
        mission = fly_mission(
            environment, impulse, None, fixed_deployment=level, fixed_main=MAIN_WINDOW[0]
        )
        # One row per decision point actually taken, carrying the state as it really was. These are
        # the states the pilot will be asked about in flight, so they are the ones to learn from.
        ascent.extend(
            (altitude, speed, deployment, mission.apogee)
            for (altitude, speed, deployment) in mission.states
        )

    for _ in range(N_DESCENT):
        impulse = 1.0 + LOT_SPREAD * float(rng.normal())
        main_altitude = float(rng.uniform(*MAIN_WINDOW))
        level = float(rng.uniform(0.0, TESTED_CEILING))
        mission = fly_mission(
            environment, impulse, None, fixed_deployment=level, fixed_main=main_altitude
        )
        descent.append((mission.apogee, main_altitude, mission.drift, mission.impact_speed))
    return ascent, descent


def main() -> None:
    rng = np.random.default_rng(SEED)
    environment = build_environment()

    print("Mission: apogee 1850 m, land close, touch down under 8 m/s.")
    print("Every in-flight decision belongs to the agent; nothing below is scheduled.\n")

    print(f"1. Randomized campaigns: {N_ASCENT} ascent + {N_DESCENT} descent flights.")
    started = time.perf_counter()
    ascent_rows, descent_rows = randomized_campaigns(environment, rng)
    print(f"   flown in {time.perf_counter() - started:.1f}s")
    ascent_model = AscentModel.fit(ascent_rows)
    descent_model = DescentModel.fit(descent_rows)
    print(
        f"   ascent model believes deployment 0 -> {ascent_model.apogee(900, 150, 0.0):.0f} m, "
        f"{TESTED_CEILING} -> {ascent_model.apogee(900, 150, TESTED_CEILING):.0f} m"
    )
    probe = TARGET_APOGEE
    print(
        f"   descent model at apogee {probe:.0f} m: main@150 -> "
        f"{descent_model.impact_speed(probe, 150.0):.1f} m/s, "
        f"main@400 -> {descent_model.impact_speed(probe, 400.0):.1f} m/s\n"
    )

    print("2. Fly the mission autonomously, 100 ms per decision.")
    budget = 1.0 / SAMPLING_RATE
    pilot = RocketPilot(ascent_model, descent_model)
    flown = fly_mission(environment, 1.0, pilot, budget=budget)
    ticks = np.array(flown.tick_times) if flown.tick_times else np.array([0.0])
    print(
        f"   apogee      {flown.apogee:7.1f} m  (target {TARGET_APOGEE:.0f}, "
        f"error {flown.apogee - TARGET_APOGEE:+.1f})"
    )
    print(
        f"   main pulled {flown.main_altitude or float('nan'):7.1f} m  "
        f"(the pilot chose this in flight)"
    )
    print(f"   drift       {flown.drift:7.1f} m")
    print(
        f"   impact      {flown.impact_speed:7.2f} m/s  "
        f"({'SAFE' if flown.safe else 'UNSAFE'}, limit {MAX_IMPACT_SPEED})"
    )
    print(f"   score       {flown.score():7.3f}")
    print(
        f"   {pilot.decisions} decisions taken, of which {len(flown.tick_times)} ran a search; "
        f"worst {ticks.max() * 1e3:.2f} ms, median {np.median(ticks) * 1e3:.2f} ms "
        f"against a {budget * 1e3:.0f} ms budget\n"
    )

    print("3. The same mission against schedules fixed before the rail.")
    lots = rng.normal(size=12)
    arms: dict[str, Any] = {
        "no brake, main @120": {"fixed_deployment": 0.0, "fixed_main": 120.0},
        "no brake, main @400": {"fixed_deployment": 0.0, "fixed_main": 400.0},
        "full brake, main @400": {"fixed_deployment": TESTED_CEILING, "fixed_main": 400.0},
        "autopilot": None,
    }
    print(f"   {'policy':24s} {'score':>7s} {'apogee err':>11s} {'drift':>9s} {'unsafe':>7s}")
    for name, config in arms.items():
        scores, errors, drifts, unsafe = [], [], [], 0
        for lot in lots:
            impulse = 1.0 + LOT_SPREAD * float(lot)
            if config is None:
                agent = RocketPilot(ascent_model, descent_model)
                mission = fly_mission(environment, impulse, agent, budget=budget)
            else:
                mission = fly_mission(environment, impulse, None, **config)
            scores.append(mission.score())
            errors.append(abs(mission.apogee - TARGET_APOGEE))
            drifts.append(mission.drift)
            unsafe += int(not mission.safe)
        print(
            f"   {name:24s} {np.mean(scores):7.3f} {np.mean(errors):9.1f} m "
            f"{np.mean(drifts):7.1f} m {unsafe:5d}/{len(lots)}"
        )


if __name__ == "__main__":
    main()
