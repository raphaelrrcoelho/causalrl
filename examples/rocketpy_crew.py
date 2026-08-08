"""A crew of four agents: the guidance half is competitive, the safety half costs more than it buys.

Every previous attempt on this branch lost to a closed form, and the reason was always the same
architectural mistake: the learned agent was put *in place of* classical control rather than around
it. ``rocketpy_baselines.py`` measured the cost -- a response surface over 60 flights, beaten 17x on
median apogee error by ``ln(1 + k v^2/g)/2k`` on zero flights -- and ``rocketpy_multiagent.py``
measured that re-partitioning the same losing decomposition changes nothing (0.569 against 0.569).

So the rule this design follows: **a competitive architecture contains the best baseline as a
component and spends its agents where that baseline is silent.** The closed form is silent about
three things, and each gets an agent:

===========  ==================================================================================
navigator    The closed form needs ``Cd``, which nobody knows before flying a new brake. The
             navigator estimates it *between* flights, by back-door adjustment when the logs
             are confounded -- the one place on this branch where a causal estimator beat a
             naive one outright (6.6x, and the naive error does not shrink with more flights).
guidance     The closed form itself, per tick, over ``Continuous(0, ceiling)``, using the
             navigator's ``Cd``. This is the component that wins, kept intact.
recovery     The closed form says nothing about when to release the main. This decision is
             irreversible -- there is no next tick to correct on -- which is exactly the regime
             where ``rocketpy_baselines.py`` found estimation worth 84.1 m against 12.8 m. It
             therefore acts on an upper *bound* on impact speed, not a point estimate.
safety       Payoff: the airframe. It disagrees with the mission at the optimum by construction,
             since its preferred action is always the conservative one, and it holds a veto.
===========  ==================================================================================

**The safety agent is why this is genuinely multi-agent and the last attempt was not.** In
``rocketpy_multiagent.py`` the two agents' objectives differed in name but agreed at the argmax, so
greedy, Nash and team play were indistinguishable. Safety cannot agree: a higher main release is
strictly better for impact speed and strictly worse for drift, so wherever the mission is at its
optimum, safety wants to move. It is not modelled as another payoff to average in -- averaging is
how a broken airframe becomes a half-success -- but as an agent that **narrows the admissible
:class:`~causalrl.InterventionSpace`**, which is what ``InterventionSpace.__and__`` exists for. The
mission agents then optimise inside whatever is left. Veto, not vote.

That narrowing is triggered by regime, not by a threshold on the action: when the vehicle is outside
the envelope the models were characterized in, the safety agent restricts the space rather than
trusting an extrapolation. It is the running-loop form of the hedge ``certify_fitted_query`` returns
offline in ``rocketpy_airbrakes.py``.

**What it measured.** Half the design works and half does not, and the split is informative.

*Guidance is competitive*, which is the thing every earlier attempt failed at: the crew's apogee
error is 86.4 m against the learned autopilot's 84.3 m nominal, because the crew's guidance agent
*is* the closed form rather than a rival to it. Containing the baseline instead of replacing it is
what closed that gap, and the navigator identifying ``Cd = 1.146`` from six flights (true 1.2,
pre-flight guess 0.6) is what let it.

*Safety costs more than it buys here.* The crew is safe -- 0/12 and 0/8 -- but so is the plain
autopilot, whose flat 1.5 m/s margin already sufficed, so the veto protects against nothing while
charging 449 m of extra drift (1505.9 against 1057.1) and, off-nominal, 83 m of apogee error from
capping brake authority outside the characterized envelope. Mission score 0.516 against 0.630.

The band is not mistuned; it is honest, and honesty is what costs. Split-conformal at 95% returns
**+2.12 m/s** where the first version of this file used a hand-picked +0.76 and produced unsafe
landings the plain autopilot avoided. The residual near the low-release cliff is far heavier-tailed
than the campaign average, which is exactly what a homoscedastic sigma hides and a conformal
quantile does not.

So the generalisable finding is about redundancy, not about causality: **a safety layer over a base
policy that is already conservative pays full price and collects nothing.** Its value would have to
be demonstrated against a base policy that actually fails -- ``main@120`` scores 0.000 with 12/12
unsafe, so such conditions exist in this mission -- and this evaluation does not create them.

    pip install "causalrl[rocketpy]"
    python examples/rocketpy_crew.py
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from causalrl import Continuous, Deadline, Discrete, Intervention, InterventionSpace
from causalrl.agents.interventional import InterventionalAgent
from causalrl.conformal import conformal_quantile

try:
    import rocketpy  # noqa: F401
except ImportError as exc:  # pragma: no cover - the example is opt-in
    raise SystemExit(
        "This example needs RocketPy: pip install 'causalrl[rocketpy]'\n"
        "RocketPy is an optional extra -- causalrl itself never imports it."
    ) from exc

from examples.rocketpy_autopilot import (
    LOT_SPREAD,
    MAIN_WINDOW,
    MAX_IMPACT_SPEED,
    TARGET_APOGEE,
    DescentModel,
    Mission,
    RocketPilot,
    build_environment,
    fly_mission,
    randomized_campaigns,
)
from examples.rocketpy_baselines import CD_BRAKE_GUESS, analytic_policy, calibrate_cd

N_CALIBRATION = 6
N_NOMINAL = 12
N_OFF_NOMINAL = 8
CHARACTERIZED_SPEED = 200.0  # fastest vertical speed the campaigns ever saw at a brake decision
SEED = 0


class Navigator:
    """Estimates the one parameter the closed form needs, between flights.

    Not an in-flight agent: it acts on the logs, on the ground, which is the honest place for an
    estimator whose input is a completed campaign. Its output is a single number that the guidance
    agent consumes, which is the whole argument for pinning a known mechanism -- one scalar to
    identify instead of a response surface to learn.
    """

    def __init__(self, cd_brake: float, *, source: str) -> None:
        self.cd_brake = cd_brake
        self.source = source


class SafetyOfficer:
    """Narrows what the mission agents may do. Never chooses an action itself.

    Two restrictions, both stated as *space* restrictions rather than as penalties:

    * **Out of characterized regime.** Above the fastest state the campaigns covered, the models
      are extrapolating, so braking authority is capped rather than trusted. This is the live
      counterpart of the hedge ``certify_fitted_query`` returns on the same question offline.
    * **Impact margin.** The main may only be withheld while releasing later still leaves an
      upper-bounded impact speed under the limit. Once that stops being true, ``release_main``
      is narrowed to ``Discrete((1.0,))`` -- a domain of one, which is how this type says
      "the decision has been made for you".
    """

    def __init__(
        self,
        descent: DescentModel,
        *,
        impact_band: float,
        characterized_apogee: tuple[float, float],
        out_of_regime_inflation: float = 4.0,
    ):
        self.descent = descent
        self.impact_band = impact_band
        self.characterized_apogee = characterized_apogee
        self.out_of_regime_inflation = out_of_regime_inflation
        self.vetoes = 0

    def impact_upper_bound(self, apogee: float, main_altitude: float) -> float:
        """Point prediction plus a split-conformal band, widened outside the characterized regime.

        The first version of this used point + 2 residual sigma and produced unsafe landings the
        plain autopilot avoided -- twice the embarrassment, since a homoscedastic sigma is exactly
        the wrong uncertainty model for a constraint that is a cliff. Near the low-release edge the
        residual is far larger than the campaign average, so a global sigma understates precisely
        where the constraint binds.

        :func:`~causalrl.conformal.conformal_quantile` fixes the in-regime half: a distribution-free
        band with marginal coverage ``1 - alpha``, calibrated on held-out flights, which is the
        right guarantee for "this irreversible action must not violate the limit more often than
        alpha". It says nothing about a query outside the calibration distribution, so an apogee
        beyond the characterized range inflates the band instead of pretending the calibration
        transfers -- the live counterpart of the hedge ``certify_fitted_query`` returns offline.
        """
        band = self.impact_band
        low, high = self.characterized_apogee
        if not low <= apogee <= high:
            band *= self.out_of_regime_inflation
        return self.descent.impact_speed(apogee, main_altitude) + band

    def restrict(
        self, observation: Mapping[str, Any], space: InterventionSpace
    ) -> InterventionSpace:
        if "deployment" in space.variables:
            if float(observation["vertical_speed"]) > CHARACTERIZED_SPEED:
                self.vetoes += 1
                # Outside the characterized envelope: allow trim, not authority.
                return space & InterventionSpace.create({"deployment": Continuous(0.0, 0.25)})
            return space
        if "release_main" in space.variables:
            altitude = float(observation["altitude"])
            apogee = float(observation["apogee"])
            # Would waiting one more decision still be recoverable? MAIN_WINDOW[0] is the last
            # altitude at which a release is physically possible at all.
            if self.impact_upper_bound(apogee, MAIN_WINDOW[0]) > MAX_IMPACT_SPEED and (
                altitude <= self.floor_for(apogee)
            ):
                self.vetoes += 1
                return space & InterventionSpace.create({"release_main": Discrete((1.0,))})
            return space
        return space

    def floor_for(self, apogee: float) -> float:
        """Lowest altitude at which release still clears the bound; below it, safety takes over."""
        grid = np.linspace(MAIN_WINDOW[0], MAIN_WINDOW[1], 48)
        safe = [h for h in grid if self.impact_upper_bound(apogee, float(h)) <= MAX_IMPACT_SPEED]
        # min(): lowest admissible release, since lower is strictly better for drift and the bound
        # is what keeps it honest. The bound, not a hand-tuned margin, is the only thing between
        # this and the cliff.
        return float(min(safe)) if safe else float(MAIN_WINDOW[1])


class CrewPilot(InterventionalAgent):
    """Guidance, recovery and safety, composed. Safety restricts; the others optimise inside.

    The composition is deliberately lexicographic rather than a weighted objective. A veto that can
    be outvoted by enough mission payoff is not a veto, and the failure it exists to prevent -- a
    broken airframe -- is not the kind of outcome that trades against apogee accuracy.
    """

    def __init__(self, navigator: Navigator, descent: DescentModel, safety: SafetyOfficer) -> None:
        self.navigator = navigator
        self.descent = descent
        self.safety = safety
        self._guidance = analytic_policy(navigator.cd_brake)
        self.decisions = 0
        # fly_mission reads this after the flight. Always zero here: the guidance agent solves a
        # bisection with a fixed iteration count rather than searching under the deadline, so there
        # is no budget for the clock to truncate -- which is itself part of why it is competitive.
        self.truncated = 0

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
        allowed = self.safety.restrict(observation, space)
        if not allowed.variables:
            return {}

        if "deployment" in allowed.variables:
            wanted = self._guidance(
                float(observation["altitude"]), float(observation["vertical_speed"])
            )
            # project(), not a hand-rolled clip: the domain knows how to make a value admissible,
            # and asking a Continuous domain for a value *list* is what the library refuses.
            return {"deployment": float(allowed.domain("deployment").project(wanted))}

        domain = allowed.domain("release_main")
        if isinstance(domain, Discrete) and len(domain.values) == 1:
            return {"release_main": float(domain.values[0])}  # safety has taken the decision
        altitude = float(observation["altitude"])
        return {
            "release_main": 1.0
            if altitude <= self.safety.floor_for(float(observation["apogee"]))
            else 0.0
        }

    def update(
        self, observation: Mapping[str, Any], intervention: Intervention, reward: float
    ) -> None:
        """No-op: the navigator refits between flights, not from a mid-flight reward."""


def calibrate_impact_band(
    rows: Sequence[tuple[float, float, float, float]], *, alpha: float = 0.05
) -> tuple[DescentModel, float, tuple[float, float]]:
    """Split-conformal band on impact speed: fit on half, calibrate on the other half.

    Fitting and calibrating on the same flights would give a band tuned to residuals the model has
    already seen, which is how an in-sample sigma ends up smaller than the error that actually
    matters. The split is what makes ``1 - alpha`` coverage mean anything.

    One-sided scores: only *under*-prediction of impact speed can hurt, so the score is the signed
    residual rather than its absolute value, and the band is added to the point prediction.
    """
    ordered = list(rows)
    cut = len(ordered) // 2
    fit_rows, calibration_rows = ordered[:cut], ordered[cut:]
    model = DescentModel.fit(fit_rows)
    scores = [
        speed - model.impact_speed(apogee, main_altitude)
        for apogee, main_altitude, _drift, speed in calibration_rows
    ]
    band = float(conformal_quantile(scores, alpha))
    apogees = [row[0] for row in ordered]
    return model, band, (float(min(apogees)), float(max(apogees)))


def summarise(missions: Sequence[Mission]) -> dict[str, float]:
    return {
        "score": float(np.mean([m.score() for m in missions])),
        "apogee": float(np.mean([abs(m.apogee - TARGET_APOGEE) for m in missions])),
        "drift": float(np.mean([m.drift for m in missions])),
        "unsafe": float(sum(not m.safe for m in missions)),
    }


def main() -> None:
    rng = np.random.default_rng(SEED)
    environment = build_environment()
    started = time.perf_counter()

    print("1. Navigator: identify the closed form's one unknown from the campaign.")
    cd_hat = calibrate_cd(rng, N_CALIBRATION)
    navigator = Navigator(cd_hat, source=f"{N_CALIBRATION} calibration flights")
    print(
        f"   Cd_brake = {cd_hat:.3f} from {N_CALIBRATION} flights "
        f"(pre-flight guess would have been {CD_BRAKE_GUESS})"
    )

    print("2. Recovery + safety: fit the descent model and calibrate an honest bound.")
    _, descent_rows = randomized_campaigns(environment, rng)
    descent, band, characterized = calibrate_impact_band(descent_rows)
    print(
        f"   split-conformal impact band = +{band:.2f} m/s at 95% coverage; "
        f"characterized apogee {characterized[0]:.0f}-{characterized[1]:.0f} m"
    )
    print(
        f"   outside that range the band inflates 4x rather than being trusted "
        f"({time.perf_counter() - started:.0f}s)\n"
    )

    def make_crew() -> CrewPilot:
        return CrewPilot(
            navigator,
            descent,
            SafetyOfficer(descent, impact_band=band, characterized_apogee=characterized),
        )

    ascent_rows, _ = randomized_campaigns(environment, np.random.default_rng(SEED + 1))
    from examples.rocketpy_autopilot import AscentModel

    ascent = AscentModel.fit(ascent_rows)

    arms: dict[str, Any] = {
        "fixed schedule (main@400)": {"fixed_deployment": 0.0, "fixed_main": 400.0},
        "learned autopilot": lambda: RocketPilot(ascent, descent),
        "crew (guidance+recovery+safety)": make_crew,
    }

    for label, lots in (
        (
            "3. Nominal lots (inside the characterized envelope)",
            rng.normal(size=N_NOMINAL) * LOT_SPREAD,
        ),
        (
            "4. Off-nominal lots (hotter than anything characterized)",
            np.abs(rng.normal(size=N_OFF_NOMINAL)) * 0.18 + 0.12,
        ),
    ):
        print(label)
        print(
            f"   {'architecture':32s} {'score':>7s} {'apogee err':>11s} {'drift':>9s} "
            f"{'unsafe':>8s}"
        )
        for name, arm in arms.items():
            missions = []
            for lot in lots:
                impulse = 1.0 + float(lot)
                if isinstance(arm, dict):
                    missions.append(fly_mission(environment, impulse, None, **arm))
                else:
                    missions.append(fly_mission(environment, impulse, arm(), budget=0.1))
            stats = summarise(missions)
            print(
                f"   {name:32s} {stats['score']:7.3f} {stats['apogee']:9.1f} m "
                f"{stats['drift']:7.1f} m {int(stats['unsafe']):5d}/{len(lots)}"
            )
        print()

    print(f"Total wall clock {time.perf_counter() - started:.0f}s.")


if __name__ == "__main__":
    main()
