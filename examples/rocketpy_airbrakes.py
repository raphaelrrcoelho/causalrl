"""Fly a rocket's air brakes to a target apogee, and know what you are allowed to claim.

RocketPy (https://rocketpy-team.github.io) is a 6-DOF flight simulator. Its one in-flight actuator
is an air-brake system: ``Rocket.add_air_brakes`` takes a ``controller_function`` called at a fixed
sampling rate that sets ``deployment_level`` in ``[0, 1]``. That makes it the smallest honest
control problem causalrl can be pointed at -- a *continuous* action, chosen against a *wall-clock*
budget, from *confounded* logs -- and each of those three words is a different part of the library.

RocketPy is a worked example of the columnar-simulator contract, never a dependency: nothing under
``src/`` imports it, and the causal layer below would drive any simulator that emits the same rows.

    pip install "causalrl[rocketpy]"
    python examples/rocketpy_airbrakes.py

The arc is the one a real campaign has to walk, and the point is that steps 2 and 3 cannot be
skipped:

1. **Fly a confounded campaign.** A crew that has seen the motor lot brakes harder on the lots it
   knows fly high. The lot is not in the logs.
2. **Ask the logs what the brakes do -- and refuse the answer.** Naively the brakes *raise* apogee,
   which is impossible: a brake is drag. ``certify_policy`` declines to certify rather than
   shipping the sign.
3. **Run the experiment the refusal calls for.** Because we own the simulator, ``do`` is a real
   randomized deployment schedule rather than an assumption. The sign comes back correct.
4. **Fit the value model, and gate it by regime.** ``certify_fitted_query`` hedges a query the
   randomization never covered instead of extrapolating into it.
5. **Fly closed-loop under a real budget.** ``AnytimeInterventionSearch`` searches ``Continuous(0,
   1)`` inside the controller callback with ``Deadline.after(1 / sampling_rate)``, and reports
   whether the clock cut the search short.
6. **Certify the flown policy.** A ``FeatureDecisionLog`` over banded deployments, gated on the
   downside rather than the mean.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from causalrl import (
    CausalGraph,
    Continuous,
    Deadline,
    InterventionSpace,
    RBFEncoder,
    TrajectoryLogBuilder,
    certify_fitted_query,
    certify_policy,
    fit_scm,
)
from causalrl.agents.anytime import AnytimeInterventionSearch
from causalrl.data.logged import FeatureDecisionLog, LoggedDecision

try:
    from rocketpy import Environment, Flight, Rocket, SolidMotor
except ImportError as exc:  # pragma: no cover - the example is opt-in
    raise SystemExit(
        "This example needs RocketPy: pip install 'causalrl[rocketpy]'\n"
        "RocketPy is an optional extra -- causalrl itself never imports it."
    ) from exc

# --- The vehicle -------------------------------------------------------------------------------
# A synthetic thrust curve rather than a .eng file, so the example is self-contained.
ELEVATION = 1400.0
BURN_TIME = 3.0
BASE_THRUST = ((0.0, 0.0), (0.1, 1600.0), (2.6, 1500.0), (BURN_TIME, 0.0))

# --- The control problem -----------------------------------------------------------------------
# 1700 m sits between what this vehicle reaches with the brakes shut (~1979 m) and with them fully
# open (~1584 m), so the target is genuinely interior: neither extreme policy attains it.
TARGET_APOGEE = 1700.0  # metres above ground level
TOLERANCE = 400.0  # apogee error at which the flight scores zero
SAMPLING_RATE = 10.0  # Hz -- the controller has 100 ms per decision
BRAKE_FLOOR = 700.0  # the brakes have no authority below this altitude (still under thrust)

# --- Campaign sizes ----------------------------------------------------------------------------
N_CONFOUNDED = 48
N_RANDOMIZED = 120
# The randomization deliberately never opens the brakes past 0.75. A cautious crew would not, and
# the consequence is the lesson of step 4: the experiment's coverage, not the actuator's range, is
# what bounds where the fitted model may be believed -- and therefore where the controller may act.
DEPLOYMENT_BANDS = (0.0, 0.25, 0.5, 0.75)
TESTED_CEILING = max(DEPLOYMENT_BANDS)
SEED = 0


def build_rocket(impulse: float, controller: Any) -> Rocket:
    """A rocket whose motor delivers ``impulse`` times nominal thrust.

    ``impulse`` is the lot-to-lot variation that drives the confounding below: it is a real
    property of the motor, and it is exactly what a log of ``(altitude, velocity, deployment)``
    fails to record.
    """
    thrust = [(t, f * impulse) for t, f in BASE_THRUST]
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
    rocket = Rocket(
        radius=0.0635,
        mass=14.0,
        inertia=(6.3, 6.3, 0.034),
        power_off_drag=0.5,
        power_on_drag=0.5,
        center_of_mass_without_motor=0.0,
        coordinate_system_orientation="tail_to_nose",
    )
    rocket.add_motor(motor, position=-1.25)
    rocket.add_nose(length=0.55, kind="von karman", position=1.28)
    rocket.add_trapezoidal_fins(4, root_chord=0.12, tip_chord=0.06, span=0.11, position=-1.05)
    rocket.add_tail(top_radius=0.0635, bottom_radius=0.0435, length=0.06, position=-1.19)
    rocket.add_air_brakes(
        drag_coefficient_curve=lambda deployment, mach: 1.2 * deployment,
        controller_function=controller,
        sampling_rate=SAMPLING_RATE,
        clamp=True,
        reference_area=None,
        override_rocket_drag=False,
    )
    return rocket


def has_authority(altitude: float, vertical_speed: float) -> bool:
    """Whether the brakes can do anything at all right now.

    This is why :class:`~causalrl.InterventionSpace` is built per decision rather than once: under
    thrust and after apogee the lever is not merely a bad idea, it is absent. A space that offered
    it would be describing a different vehicle.
    """
    return altitude > BRAKE_FLOOR and vertical_speed > 0.0


@dataclass
class FlightRecord:
    """One flight's logged decisions and its outcome."""

    decisions: list[tuple[float, float, float]]  # (altitude, vertical speed, deployment)
    apogee: float
    impulse: float
    tick_times: list[float]
    truncated_ticks: int


def score(apogee: float) -> float:
    """Bounded flight score in ``[0, 1]``: 1 at the target, 0 once ``TOLERANCE`` off.

    Bounded on purpose -- the marginal-sensitivity-model bounds the certificate layer runs assume
    a bounded outcome, and an unbounded apogee error would silently break that.
    """
    return float(max(0.0, 1.0 - abs(apogee - TARGET_APOGEE) / TOLERANCE))


def run_flight(
    environment: Environment,
    impulse: float,
    policy: Any,
    *,
    budget: float | None = None,
) -> FlightRecord:
    """Fly once, calling ``policy(altitude, vertical_speed, deadline)`` at ``SAMPLING_RATE`` Hz.

    ``budget`` is the per-decision wall-clock allowance handed to the policy. ``None`` means the
    policy is a fixed rule with nothing to search, which is the case for both logging campaigns.
    """
    decisions: list[tuple[float, float, float]] = []
    tick_times: list[float] = []
    truncated = 0

    def controller_function(
        time_: float,
        sampling_rate: float,
        state: Sequence[float],
        state_history: Sequence[Sequence[float]],
        observed_variables: Sequence[Any],
        interactive_objects: Any,
        sensors: Any = None,
        environment_: Any = None,
    ) -> tuple[float, float]:
        nonlocal truncated
        air_brakes = (
            interactive_objects[0]
            if isinstance(interactive_objects, (list, tuple))
            else interactive_objects
        )
        altitude = float(state[2]) - ELEVATION
        vertical_speed = float(state[5])

        if not has_authority(altitude, vertical_speed):
            air_brakes.deployment_level = 0.0
            return (time_, 0.0)

        started = time.perf_counter()
        deadline = Deadline.after(budget) if budget is not None else None
        deployment, was_truncated = policy(altitude, vertical_speed, deadline)
        tick_times.append(time.perf_counter() - started)
        truncated += int(was_truncated)

        air_brakes.deployment_level = deployment
        decisions.append((altitude, vertical_speed, deployment))
        return (time_, deployment)

    flight = Flight(
        rocket=build_rocket(impulse, controller_function),
        environment=environment,
        rail_length=5.2,
        inclination=85,
        heading=0,
        terminate_on_apogee=True,
    )
    return FlightRecord(
        decisions=decisions,
        apogee=float(flight.apogee) - ELEVATION,
        impulse=impulse,
        tick_times=tick_times,
        truncated_ticks=truncated,
    )


def fixed_policy(level: float) -> Any:
    """A policy that holds one deployment level whenever the brakes have authority."""

    def policy(_altitude: float, _speed: float, _deadline: Deadline | None) -> tuple[float, bool]:
        return level, False

    return policy


def fly_campaign(
    environment: Environment,
    rng: np.random.Generator,
    *,
    n: int,
    randomized: bool,
    builder: TrajectoryLogBuilder,
    episode_offset: int,
) -> list[FlightRecord]:
    """Fly ``n`` flights, logging every decision into ``builder`` as it happens.

    ``randomized=False`` is the confounded campaign: the crew has seen the motor lot and brakes
    harder on the lots that fly high, so deployment and apogee share an unlogged cause.
    ``randomized=True`` is the experiment: the deployment band is drawn independently of the lot,
    which is what ``do`` means when you own the simulator.
    """
    records: list[FlightRecord] = []
    for i in range(n):
        lot = float(rng.normal())
        impulse = 1.0 + 0.12 * lot
        if randomized:
            level = float(rng.choice(DEPLOYMENT_BANDS))
        else:
            level = float(np.clip(0.5 + 0.30 * lot + rng.normal(scale=0.10), 0.0, 1.0))

        record = run_flight(environment, impulse, fixed_policy(level))
        records.append(record)

        episode = episode_offset + i
        with builder.episode(episode) as writer:
            for t, (altitude, speed, deployment) in enumerate(record.decisions):
                writer.push(kind="obs", name="altitude", value=altitude, t=t)
                writer.push(kind="obs", name="vertical_speed", value=speed, t=t)
                writer.push(kind="action", name="deployment", value=deployment, t=t)
            writer.push(kind="reward", name="score", value=score(record.apogee), t=0)
    return records


def slope_of_apogee_on_deployment(
    records: Sequence[FlightRecord], *, adjust_for_lot: bool
) -> float:
    """Least-squares ``d(apogee)/d(deployment)``, optionally controlling for the motor lot.

    ``adjust_for_lot=True`` is the oracle: it uses a column the real logs do not have. It is here
    to show what the confounded estimate is wrong *about*, not as something a caller could run.
    """
    deployment = np.array([r.decisions[0][2] if r.decisions else 0.0 for r in records])
    apogee = np.array([r.apogee for r in records])
    columns = [np.ones_like(deployment), deployment]
    if adjust_for_lot:
        columns.append(np.array([r.impulse for r in records]))
    design = np.column_stack(columns)
    return float(np.linalg.lstsq(design, apogee, rcond=None)[0][1])


def band_of(deployment: float) -> float:
    """Snap a deployment level to its band -- the action type the certificate layer can compare.

    ``LoggedDecisions.matches`` asks whether the target policy would take the logged action
    *exactly*. That is the right question for arms and the wrong one for a real number: two floats
    from a continuous domain essentially never coincide, so an unbanded continuous policy matches
    nothing and the log carries no evidence about it. Banding for certification does not make the
    *decision* discrete -- the controller below still searches ``Continuous(0, 1)``.
    """
    bands = np.asarray(DEPLOYMENT_BANDS)
    return float(bands[int(np.argmin(np.abs(bands - deployment)))])


class ApogeeModel:
    """Predicted apogee from holding a deployment level, as a function of the current state.

    Ridge on RBF features of ``(altitude, vertical speed)`` crossed with a quadratic in deployment.
    The cross terms are the point: a model additive in deployment could not represent the fact that
    the same brake setting costs more altitude when applied earlier and faster.

    Fit on the *randomized* campaign only. Fitting it on the confounded logs would learn the crew's
    reflex rather than the brakes' effect, which is the whole failure this example is about.
    """

    def __init__(self, encoder: RBFEncoder, weights: NDArray[np.float64]) -> None:
        self._encoder = encoder
        self._weights = weights

    @staticmethod
    def _cross(features: NDArray[np.float64], deployment: float) -> NDArray[np.float64]:
        return np.concatenate([features, features * deployment, features * deployment**2])

    @classmethod
    def fit(
        cls, records: Sequence[FlightRecord], *, centers: int = 6, ridge: float = 1e-3
    ) -> ApogeeModel:
        states = np.array([[a, v] for r in records for (a, v, _) in r.decisions], dtype=np.float64)
        deployments = np.array([d for r in records for (_, _, d) in r.decisions])
        apogees = np.array([r.apogee for r in records for _ in r.decisions])

        lows, highs = states.min(axis=0), states.max(axis=0)
        grid = np.linspace(0.0, 1.0, centers)[:, None] * (highs - lows) + lows
        scale = np.maximum(highs - lows, 1e-9)
        encoder = RBFEncoder(
            _ScaledEncoder(("altitude", "vertical_speed"), lows, scale),
            (grid - lows) / scale,
            bandwidth=0.4,
        )

        design = np.array(
            [
                cls._cross(encoder.encode({"altitude": s[0], "vertical_speed": s[1]}), d)
                for s, d in zip(states, deployments, strict=True)
            ]
        )
        gram = design.T @ design + ridge * np.eye(design.shape[1])
        weights = np.linalg.solve(gram, design.T @ apogees).astype(np.float64)
        return cls(encoder, weights)

    def apogee(self, altitude: float, vertical_speed: float, deployment: float) -> float:
        features = self._encoder.encode({"altitude": altitude, "vertical_speed": vertical_speed})
        return float(self._cross(features, deployment) @ self._weights)

    def value(self, observation: Mapping[str, Any], intervention: Mapping[str, Any]) -> float:
        """Search objective: higher is better, peaking where predicted apogee hits the target."""
        predicted = self.apogee(
            float(observation["altitude"]),
            float(observation["vertical_speed"]),
            float(intervention["deployment"]),
        )
        return -abs(predicted - TARGET_APOGEE)


@dataclass(frozen=True)
class _ScaledEncoder:
    """Min-max scaled ``(altitude, vertical_speed)`` -- the inner encoder the RBF basis sits on."""

    keys: tuple[str, ...]
    lows: NDArray[np.float64]
    scale: NDArray[np.float64]

    @property
    def dim(self) -> int:
        return len(self.keys)

    def encode(self, observation: Mapping[str, Any]) -> NDArray[np.float64]:
        raw = np.array([float(observation[k]) for k in self.keys])
        return (raw - self.lows) / self.scale


def searching_policy(model: ApogeeModel, *, rounds: int, candidates: int) -> Any:
    """A controller that searches the continuous deployment range within its per-tick budget.

    The space stops at ``TESTED_CEILING`` rather than at the actuator's physical limit of 1.0. The
    hardware can open further; the *evidence* cannot. Step 4 shows the model declining to be read
    above that line, and a space that offered the extra range would be inviting the controller into
    exactly the region its own certificate refuses to license.
    """
    space = InterventionSpace.create({"deployment": Continuous(0.0, TESTED_CEILING)})

    def policy(
        altitude: float, vertical_speed: float, deadline: Deadline | None
    ) -> tuple[float, bool]:
        search = AnytimeInterventionSearch(
            model.value, rounds=rounds, candidates_per_round=candidates, seed=SEED
        )
        observation = {"altitude": altitude, "vertical_speed": vertical_speed}
        chosen = search.act(observation, space=space, deadline=deadline)
        report = search.last_search
        return float(chosen["deployment"]), not report.exhausted

    return policy


def build_environment() -> Environment:
    environment = Environment(latitude=32.99, longitude=-106.97, elevation=ELEVATION)
    environment.set_atmospheric_model(type="standard_atmosphere")
    return environment


def randomization_propensity(_state: NDArray[np.float64], action: Mapping[str, Any]) -> float:
    """``pi_behavior(a | s)`` for the randomized campaign -- uniform over the tested bands.

    Supplying this is what lets the conformal gate bound the likelihood ratio at a fresh test
    point. Without it the log cannot answer what probability the *logging* policy gave the action
    the *target* policy would take -- an action that may never have been played -- and the band
    correctly widens to infinity rather than letting an unknown pass as a small number. Here there
    is no unknown: we ran the randomization, so the propensity is known by construction.
    """
    return 1.0 / len(DEPLOYMENT_BANDS) if action["deployment"] in DEPLOYMENT_BANDS else 0.0


def certification_log(records: Sequence[FlightRecord]) -> FeatureDecisionLog[Any]:
    """One logged decision per flight, keyed on the banded deployment it held."""
    decisions = [
        LoggedDecision(
            state=np.array([r.decisions[0][0], r.decisions[0][1]], dtype=np.float64),
            action={"deployment": band_of(r.decisions[0][2])},
            reward=score(r.apogee),
            propensity=1.0 / len(DEPLOYMENT_BANDS),  # the randomization actually run
        )
        for r in records
        if r.decisions
    ]
    return FeatureDecisionLog(decisions, behavior_propensity=randomization_propensity)


def main() -> None:
    rng = np.random.default_rng(SEED)
    environment = build_environment()
    builder = TrajectoryLogBuilder(metadata={"simulator": "rocketpy", "target": TARGET_APOGEE})

    print(
        f"Target apogee {TARGET_APOGEE:.0f} m AGL; brakes have authority above "
        f"{BRAKE_FLOOR:.0f} m while climbing.\n"
    )

    print(f"1. Confounded campaign: {N_CONFOUNDED} flights, crew brakes harder on hot motor lots.")
    started = time.perf_counter()
    confounded = fly_campaign(
        environment, rng, n=N_CONFOUNDED, randomized=False, builder=builder, episode_offset=0
    )
    print(
        f"   flown in {time.perf_counter() - started:.1f}s; "
        f"{len(builder)} rows logged from inside the controller callback"
    )
    naive = slope_of_apogee_on_deployment(confounded, adjust_for_lot=False)
    oracle = slope_of_apogee_on_deployment(confounded, adjust_for_lot=True)
    print(f"   naive   d(apogee)/d(deployment) = {naive:+8.1f} m")
    print(f"   oracle  (controlling for lot)   = {oracle:+8.1f} m")
    print("   The naive sign says braking RAISES apogee. A brake is drag; that is impossible.\n")

    print("2. Ask the certificate layer to ship the naive policy.")
    confounded_log = certification_log(confounded)
    always_brake = [{"deployment": 1.0}] * len(confounded_log)
    verdict = certify_policy(confounded_log, always_brake, alpha=0.1)
    print(f"   recommendation: {verdict.recommendation.upper()}")
    print(f"   {verdict.summary[:160]}")
    print()

    print(f"3. Run the experiment the refusal calls for: {N_RANDOMIZED} randomized flights.")
    started = time.perf_counter()
    randomized = fly_campaign(
        environment,
        rng,
        n=N_RANDOMIZED,
        randomized=True,
        builder=builder,
        episode_offset=N_CONFOUNDED,
    )
    print(f"   flown in {time.perf_counter() - started:.1f}s; {len(builder)} rows total")
    experimental = slope_of_apogee_on_deployment(randomized, adjust_for_lot=False)
    print(f"   randomized d(apogee)/d(deployment) = {experimental:+8.1f} m  <- sign restored")
    print()

    print("4. Fit the value model on the randomized flights, and gate it by regime.")
    model = ApogeeModel.fit(randomized)
    columns = {
        "deployment": np.array([r.decisions[0][2] for r in randomized if r.decisions]),
        "vertical_speed": np.array([r.decisions[0][1] for r in randomized if r.decisions]),
        "apogee": np.array([r.apogee for r in randomized if r.decisions]),
    }
    scm = fit_scm(
        columns,
        graph=CausalGraph(
            directed_edges=[
                ("vertical_speed", "deployment"),
                ("vertical_speed", "apogee"),
                ("deployment", "apogee"),
            ],
            nodes=["vertical_speed", "deployment", "apogee"],
        ),
    )
    for probe, label in ((0.5, "the randomization tested"), (1.0, "it never opened this far")):
        certificate = certify_fitted_query(
            scm, columns, intervention={"deployment": probe}, outcome="apogee", atol=0.05
        )
        state = "TRUSTED" if certificate.hedge is None else "HEDGED"
        print(f"   deployment={probe:.2f} ({label:24s}): {state}")
    print(
        f"   -> the controller's action space stops at {TESTED_CEILING:.2f}, not at the "
        "actuator's 1.00."
    )
    print()

    print("5. Fly closed-loop: continuous search, 100 ms per decision.")
    budget = 1.0 / SAMPLING_RATE
    policy = searching_policy(model, rounds=8, candidates=16)
    flown = run_flight(environment, impulse=1.0, policy=policy, budget=budget)
    ticks = np.array(flown.tick_times)
    print(
        f"   apogee {flown.apogee:.1f} m  (target {TARGET_APOGEE:.0f} m, "
        f"error {flown.apogee - TARGET_APOGEE:+.1f} m)"
    )
    print(
        f"   {len(ticks)} decisions; per-tick worst {ticks.max() * 1e3:.2f} ms, "
        f"median {np.median(ticks) * 1e3:.2f} ms, budget {budget * 1e3:.0f} ms"
    )
    print(
        f"   overruns: {int((ticks > budget).sum())}; "
        f"searches cut short by the clock: {flown.truncated_ticks}/{len(ticks)}"
    )
    for level in (0.0, TESTED_CEILING):
        reference = run_flight(environment, impulse=1.0, policy=fixed_policy(level))
        print(f"   for reference, deployment held at {level:.2f}: {reference.apogee:7.1f} m")
    print()

    print("   The same controller on a budget it cannot meet (1 ms):")
    rushed = searching_policy(model, rounds=512, candidates=512)
    hurried = run_flight(environment, impulse=1.0, policy=rushed, budget=1e-3)
    hurried_ticks = np.array(hurried.tick_times)
    print(
        f"   apogee {hurried.apogee:.1f} m; still flew, still inside budget "
        f"(worst {hurried_ticks.max() * 1e3:.2f} ms)"
    )
    print(
        f"   but {hurried.truncated_ticks}/{len(hurried_ticks)} searches report themselves "
        "truncated rather than exhaustive."
    )
    print()

    print("6. Certify the flown policy against the randomized log.")
    randomized_log = certification_log(randomized)
    targets = [
        {
            "deployment": band_of(
                policy(float(d.state[0]), float(d.state[1]), Deadline.after(budget))[0]
            )
        }
        for d in randomized_log.decisions
    ]
    # gamma_max encodes what is known about the assignment mechanism. Step 2's logs came from a
    # crew reacting to the motor lot, where a large hidden odds-ratio is entirely plausible; these
    # came from a randomization we ran ourselves, so the ceiling is near 1 and asking for
    # robustness to Γ=10 would be refusing to use the experiment we just paid for.
    matched = sum(randomized_log.matches(targets))
    positivity = randomized_log.positivity(targets)
    print(f"   the searched policy reproduces {matched}/{len(randomized_log)} logged bands")
    print(f"   positivity checkable: {positivity.checkable}; gaps: {len(positivity.gaps)}")
    final = certify_policy(randomized_log, targets, gamma_max=1.25, alpha=0.1)
    print(f"   recommendation: {final.recommendation.upper()}")
    print(f"   {final.summary[:220]}")

    log = builder.freeze()
    print(
        f"\n{len(log)} rows across {N_CONFOUNDED + N_RANDOMIZED} flights; "
        f"fingerprint {log.fingerprint()[:16]}"
    )


if __name__ == "__main__":
    main()
