"""Two agents fly the mission as a game, because one agent deciding twice is not the same thing.

``rocketpy_autopilot.py`` gives one pilot both levers, and it resolves them **greedily and in
order**: the brake targets apogee, then recovery makes the best of whatever apogee it was handed.
That ordering is an architectural choice nobody argued for, and it throws away the coupling --
braking does not only lower apogee, it shortens the descent and therefore the wind drift that
recovery is judged on. A brake setting that is slightly *wrong* for apogee can be right for the
mission.

So the honest model is not one agent with two decisions. It is two agents whose payoffs disagree:

* **apogee** chooses the deployment band, and is scored on ``|apogee - target|``;
* **recovery** chooses the main-release band, and is scored on drift, with an impact-speed gate.

That is a finite game, and :mod:`causalrl.magames` already computes what to do with one. Three
solution concepts, all over the same payoff tables, all built from the *learned* models so solving
costs no flights:

===================  =======================================================================
greedy               what the single pilot does: apogee picks first and best-responds to
                     nothing; recovery best-responds to that. A Stackelberg play with the
                     leader ignoring the follower entirely.
nash                 :func:`~causalrl.pure_nash_equilibria` of the selfish game -- nobody
                     can unilaterally improve. Not the same as the mission optimum, and the
                     gap between them is the price of letting the levers have owners.
team                 both agents handed the *shared* mission score. The joint argmax, which
                     is what the mission actually wants, and what greedy cannot reach.
===================  =======================================================================

**What it measures, and what it found.** The three concepts share a granularity and a commitment
point, so comparing them isolates the decomposition and nothing else -- and on this mission they
are indistinguishable: identical play at the probe context, and 0.569 / 0.569 / 0.568 mission score
in flight. The coupling is real in the physics but too weak to move the argmax, because wherever
the brake is worth using for drift it is already wanted for apogee, so the agents never actually
disagree.

The gap that *does* exist is elsewhere. The single sequential pilot scores 0.642 against these
0.569, and the reason is not its decomposition -- ``greedy`` here **is** its decomposition, banded.
The difference is that the pilot searches ``Continuous(0.0, 0.75)`` afresh every tick while these
commit to a band once. Resolution and feedback beat coordination, which is the same verdict
``rocketpy_baselines.py`` reaches from the other direction.

So this is a negative result, kept because it is a useful one: it says the single pilot's silent
architectural choice was not costing anything here, and it says what would have to change for a
game formulation to earn its place -- payoffs that genuinely conflict at the optimum, not merely
objectives that differ in name.

    pip install "causalrl[rocketpy]"
    python examples/rocketpy_multiagent.py
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from itertools import product
from typing import Any

import numpy as np

from causalrl import (
    CausalGame,
    CausalGraph,
    Deadline,
    Intervention,
    InterventionSpace,
    pure_nash_equilibria,
    run_no_regret,
)
from causalrl.games import decision_node, utility_node

try:
    import rocketpy  # noqa: F401
except ImportError as exc:  # pragma: no cover - the example is opt-in
    raise SystemExit(
        "This example needs RocketPy: pip install 'causalrl[rocketpy]'\n"
        "RocketPy is an optional extra -- causalrl itself never imports it."
    ) from exc

from examples.rocketpy_airbrakes import TESTED_CEILING
from examples.rocketpy_autopilot import (
    LOT_SPREAD,
    MAX_IMPACT_SPEED,
    TARGET_APOGEE,
    AscentModel,
    DescentModel,
    RocketPilot,
    build_environment,
    fly_mission,
    randomized_campaigns,
)

DEPLOY_BANDS = (0.0, 0.25, 0.5, TESTED_CEILING)
MAIN_BANDS = (150.0, 300.0, 450.0, 600.0)
AGENTS = ("apogee", "recovery")
SAFETY_MARGIN = 1.5
N_EVAL = 12
NO_REGRET_ROUNDS = 4000
SEED = 0


def apogee_payoff(ascent: AscentModel, altitude: float, speed: float, deployment: float) -> float:
    """The apogee agent's score: 1 at the target, 0 once 400 m off. Bounded, as the game needs."""
    predicted = ascent.apogee(altitude, speed, deployment)
    return float(max(0.0, 1.0 - abs(predicted - TARGET_APOGEE) / 400.0))


def recovery_payoff(descent: DescentModel, apogee: float, main_altitude: float) -> float:
    """The recovery agent's score: low drift, and zero if the impact gate is not cleared.

    The gate is a gate, not a weight. A landing that breaks the airframe did not partly succeed,
    and a payoff that averaged it against good drift would tell the equilibrium otherwise.
    """
    if descent.impact_speed(apogee, main_altitude) > MAX_IMPACT_SPEED - SAFETY_MARGIN:
        return 0.0
    return float(max(0.0, 1.0 - descent.drift(apogee, main_altitude) / 2000.0))


def mission_payoff(
    ascent: AscentModel,
    descent: DescentModel,
    altitude: float,
    speed: float,
    deployment: float,
    main_altitude: float,
) -> float:
    """The shared score the mission is actually judged on -- what the team game maximises."""
    predicted = ascent.apogee(altitude, speed, deployment)
    if descent.impact_speed(predicted, main_altitude) > MAX_IMPACT_SPEED - SAFETY_MARGIN:
        return 0.0
    apogee_term = max(0.0, 1.0 - abs(predicted - TARGET_APOGEE) / 400.0)
    drift_term = max(0.0, 1.0 - descent.drift(predicted, main_altitude) / 2000.0)
    return float(0.5 * apogee_term + 0.5 * drift_term)


def build_game(
    ascent: AscentModel, descent: DescentModel, altitude: float, speed: float, *, team: bool
) -> CausalGame:
    """The finite game at one decision context, from the learned models alone.

    ``team=True`` hands both agents the shared mission score, turning the game into a coordination
    problem whose equilibria include the joint optimum. ``team=False`` gives each agent its own
    objective, which is the honest model of two subsystems with separate acceptance criteria -- and
    whose equilibrium need not be the mission's.
    """
    profiles = list(product(range(len(DEPLOY_BANDS)), range(len(MAIN_BANDS))))
    utilities: dict[str, dict[tuple[int, ...], float]] = {a: {} for a in AGENTS}
    for i, j in profiles:
        deployment, main_altitude = DEPLOY_BANDS[i], MAIN_BANDS[j]
        predicted = ascent.apogee(altitude, speed, deployment)
        if team:
            shared = mission_payoff(ascent, descent, altitude, speed, deployment, main_altitude)
            utilities["apogee"][i, j] = shared
            utilities["recovery"][i, j] = shared
        else:
            utilities["apogee"][i, j] = apogee_payoff(ascent, altitude, speed, deployment)
            utilities["recovery"][i, j] = recovery_payoff(descent, predicted, main_altitude)

    graph = CausalGraph(
        directed_edges=[
            (decision_node("apogee"), utility_node("apogee")),
            (decision_node("apogee"), utility_node("recovery")),  # braking changes descent time
            (decision_node("recovery"), utility_node("recovery")),
        ],
        nodes=[decision_node(a) for a in AGENTS] + [utility_node(a) for a in AGENTS],
    )
    return CausalGame(
        agents=AGENTS,
        actions={
            a: tuple(range(len(DEPLOY_BANDS if a == "apogee" else MAIN_BANDS))) for a in AGENTS
        },
        utilities={a: dict(utilities[a]) for a in AGENTS},
        graph=graph,
    )


def greedy_profile(
    ascent: AscentModel, descent: DescentModel, altitude: float, speed: float
) -> tuple[int, int]:
    """The single pilot's play: apogee optimises alone, recovery best-responds to the result."""
    i = max(
        range(len(DEPLOY_BANDS)),
        key=lambda k: apogee_payoff(ascent, altitude, speed, DEPLOY_BANDS[k]),
    )
    predicted = ascent.apogee(altitude, speed, DEPLOY_BANDS[i])
    j = max(
        range(len(MAIN_BANDS)),
        key=lambda k: recovery_payoff(descent, predicted, MAIN_BANDS[k]),
    )
    return i, j


def team_profile(
    ascent: AscentModel, descent: DescentModel, altitude: float, speed: float
) -> tuple[int, int]:
    """The joint argmax of the shared mission score -- what greedy cannot see."""
    return max(
        product(range(len(DEPLOY_BANDS)), range(len(MAIN_BANDS))),
        key=lambda ij: mission_payoff(
            ascent, descent, altitude, speed, DEPLOY_BANDS[ij[0]], MAIN_BANDS[ij[1]]
        ),
    )


def nash_profile(
    ascent: AscentModel, descent: DescentModel, altitude: float, speed: float
) -> tuple[int, int]:
    """A pure Nash equilibrium of the selfish game; the mission-best one when several exist."""
    game = build_game(ascent, descent, altitude, speed, team=False)
    equilibria = pure_nash_equilibria(game)
    if not equilibria:
        return greedy_profile(ascent, descent, altitude, speed)
    best = max(
        equilibria,
        key=lambda e: mission_payoff(
            ascent,
            descent,
            altitude,
            speed,
            DEPLOY_BANDS[e["apogee"]],
            MAIN_BANDS[e["recovery"]],
        ),
    )
    return best["apogee"], best["recovery"]


class GamePilot(RocketPilot):
    """A pilot whose two levers are decided by a solved game rather than in sequence.

    The game is solved at the *first* ascent decision and the profile held, because the levers are
    resolved jointly or not at all: re-solving mid-coast with the brake already partly committed
    would reintroduce exactly the sequential dependence the game exists to remove.
    """

    def __init__(self, ascent: AscentModel, descent: DescentModel, *, solver: str = "team") -> None:
        super().__init__(ascent, descent)
        self.solver = solver
        self._profile: tuple[int, int] | None = None

    def _solve(self, altitude: float, speed: float) -> tuple[int, int]:
        if self._profile is None:
            chooser = {"greedy": greedy_profile, "nash": nash_profile, "team": team_profile}[
                self.solver
            ]
            self._profile = chooser(self.ascent, self.descent, altitude, speed)
        return self._profile

    def act(
        self,
        observation: Mapping[str, Any],
        *,
        space: InterventionSpace,
        deadline: Deadline | None = None,
    ) -> Intervention:
        self.decisions += 1
        if not space.variables:
            return {}
        if "deployment" in space.variables:
            i, _ = self._solve(float(observation["altitude"]), float(observation["vertical_speed"]))
            return {"deployment": DEPLOY_BANDS[i]}
        if self._profile is None:  # descent reached without an ascent decision
            return {"release_main": 0.0}
        _, j = self._profile
        return {"release_main": 1.0 if float(observation["altitude"]) <= MAIN_BANDS[j] else 0.0}


def evaluate(environment: Any, lots: Sequence[float], make_pilot: Any) -> dict[str, float]:
    scores, errors, drifts, unsafe = [], [], [], 0
    for lot in lots:
        mission = fly_mission(environment, 1.0 + LOT_SPREAD * float(lot), make_pilot(), budget=0.1)
        scores.append(mission.score())
        errors.append(abs(mission.apogee - TARGET_APOGEE))
        drifts.append(mission.drift)
        unsafe += int(not mission.safe)
    return {
        "score": float(np.mean(scores)),
        "apogee": float(np.mean(errors)),
        "drift": float(np.mean(drifts)),
        "unsafe": float(unsafe),
    }


def main() -> None:
    rng = np.random.default_rng(SEED)
    environment = build_environment()
    started = time.perf_counter()

    print("Fitting the two agents' models from randomized campaigns...")
    ascent_rows, descent_rows = randomized_campaigns(environment, rng)
    ascent, descent = AscentModel.fit(ascent_rows), DescentModel.fit(descent_rows)
    print(f"   done ({time.perf_counter() - started:.0f}s)\n")

    probe_altitude, probe_speed = 900.0, 150.0
    selfish = build_game(ascent, descent, probe_altitude, probe_speed, team=False)
    equilibria = pure_nash_equilibria(selfish)
    print(f"1. The selfish game at ({probe_altitude:.0f} m, {probe_speed:.0f} m/s):")
    print(
        f"   {len(DEPLOY_BANDS)}x{len(MAIN_BANDS)} profiles, "
        f"{len(equilibria)} pure Nash equilibrium/equilibria"
    )
    for e in equilibria:
        joint = mission_payoff(
            ascent,
            descent,
            probe_altitude,
            probe_speed,
            DEPLOY_BANDS[e["apogee"]],
            MAIN_BANDS[e["recovery"]],
        )
        print(
            f"     deployment={DEPLOY_BANDS[e['apogee']]:.2f} "
            f"main={MAIN_BANDS[e['recovery']]:.0f} m -> mission {joint:.3f}"
        )
    gi, gj = greedy_profile(ascent, descent, probe_altitude, probe_speed)
    ti, tj = team_profile(ascent, descent, probe_altitude, probe_speed)
    greedy_value = mission_payoff(
        ascent, descent, probe_altitude, probe_speed, DEPLOY_BANDS[gi], MAIN_BANDS[gj]
    )
    team_value = mission_payoff(
        ascent, descent, probe_altitude, probe_speed, DEPLOY_BANDS[ti], MAIN_BANDS[tj]
    )
    print(
        f"   greedy  : deployment={DEPLOY_BANDS[gi]:.2f} main={MAIN_BANDS[gj]:.0f} m "
        f"-> mission {greedy_value:.3f}"
    )
    print(
        f"   team    : deployment={DEPLOY_BANDS[ti]:.2f} main={MAIN_BANDS[tj]:.0f} m "
        f"-> mission {team_value:.3f}"
    )
    print()

    print(f"2. No-regret play on the team game, {NO_REGRET_ROUNDS} rounds:")
    team_game = build_game(ascent, descent, probe_altitude, probe_speed, team=True)
    run = run_no_regret(team_game, NO_REGRET_ROUNDS, seed=SEED)
    print(
        f"   measured external regret {run.regret:.2e} -- the coordination the greedy ordering "
        "never negotiates"
    )
    print()

    print(f"3. Flying {N_EVAL} motor lots under each decomposition.")
    lots = rng.normal(size=N_EVAL)
    arms = {
        "single pilot (sequential)": lambda: RocketPilot(ascent, descent),
        "game: greedy profile": lambda: GamePilot(ascent, descent, solver="greedy"),
        "game: nash (selfish)": lambda: GamePilot(ascent, descent, solver="nash"),
        "game: team optimum": lambda: GamePilot(ascent, descent, solver="team"),
    }
    print(
        f"   {'decomposition':28s} {'score':>7s} {'apogee err':>11s} {'drift':>10s} {'unsafe':>8s}"
    )
    for name, make_pilot in arms.items():
        stats = evaluate(environment, lots, make_pilot)
        print(
            f"   {name:28s} {stats['score']:7.3f} {stats['apogee']:9.1f} m "
            f"{stats['drift']:8.1f} m {int(stats['unsafe']):5d}/{N_EVAL}"
        )
    print(f"\nTotal wall clock {time.perf_counter() - started:.0f}s.")


if __name__ == "__main__":
    main()
