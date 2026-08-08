"""Where causal methods actually buy you flights: the y-axis is flight count, not accuracy.

``rocketpy_airbrakes.py`` and ``rocketpy_autopilot.py`` both learn from *randomized* campaigns, and
randomization is the brute-force substitute for causal inference. Given 120 randomized flights,
adjustment cannot beat plain regression -- both converge to the same answer, so any comparison at
that sample size measures nothing. Real campaigns do not get 120 randomized flights. They get a
handful, plus logs from flights that were flown for other reasons.

So the question "do causal methods fly better" is best asked as **how many flights does it take**,
and the two mechanisms that answer it are on opposite sides of what you are allowed to do:

**A. Counterfactual pairing (L3).** Because a simulator's exogenous draws can be held fixed, the
same motor lot can be flown twice under different deployments. That is a genuine counterfactual
pair, and differencing within it cancels every between-flight source of variance -- which is the
entire reason the slope is hard to estimate. *This is a simulator privilege.* A real campaign cannot
re-fly a motor it has already burned, so this buys sample efficiency in sim-to-real training, never
in a flight-test programme. Stated here because the alternative is implying otherwise.

**B. Back-door adjustment (L2).** The motor lot is stamped on the casing -- in reality it *is*
recorded. When the confounder is observed, operational logs from a reacting crew are usable without
randomizing anything, provided the crew's policy leaves enough overlap. Whether it does is an
empirical question this script answers rather than assumes, and a positivity failure is a real
outcome, not a bug.

Both parts report error against the same estimand -- the OLS slope of apogee on deployment under
randomization -- so the three estimators are directly comparable at equal flight count.

    pip install "causalrl[rocketpy]"
    python examples/rocketpy_sample_efficiency.py
"""

from __future__ import annotations

import time
from collections.abc import Sequence

import numpy as np

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
    build_environment,
    build_rocket,
)

# Motor-lot impulse dispersion: the exogenous draw, and the confounder. 5% is what a QC'd lot
# actually varies by. The sister examples use 12% to make confounding vivid, but that is wide
# enough that cold lots never reach BRAKE_FLOOR at all -- the brakes then have no authority, those
# flights contribute an identically-zero contrast, and every estimator is attenuated toward zero
# for a reason that has nothing to do with the estimator.
LOT_SPREAD = 0.05
LOT_CLIP = 2.0  # lots beyond 2 sigma are screened out before flight, as they are in practice
D_LOW, D_HIGH = 0.0, TESTED_CEILING  # the two levels both designs in part A contrast
SEED = 0

TRUTH_PAIRS = 60  # pairs used to pin the reference slope
SAMPLE_SIZES = (8, 16, 32)  # flights per estimate
REPLICATES = 12  # independent repetitions; an RMSE over few replicates is itself noise
ADJUST_SIZES = (10, 20, 40)
ADJUST_REPLICATES = 8

_environment = None


def environment():
    global _environment
    if _environment is None:
        _environment = build_environment()
    return _environment


def draw_lots(rng: np.random.Generator, size: int) -> np.ndarray:
    """Screened motor lots: standard normal, clipped at :data:`LOT_CLIP` sigma."""
    return np.clip(rng.normal(size=size), -LOT_CLIP, LOT_CLIP)


def impulse_of(lot: float) -> float:
    """Motor impulse multiplier for a lot."""
    return 1.0 + LOT_SPREAD * float(lot)


def apogee_of(impulse: float, deployment: float) -> float:
    """Fly once and return apogee AGL. Deterministic in ``(impulse, deployment)``.

    That determinism is what makes part A possible: holding ``impulse`` fixed and varying
    ``deployment`` *is* the counterfactual, with no abduction step needed because the exogenous
    draw was ours to set in the first place.
    """

    def controller(time_, sampling_rate, state, history, observed, interactive, *rest):
        brakes = interactive[0] if isinstance(interactive, (list, tuple)) else interactive
        altitude, vertical_speed = state[2] - ELEVATION, state[5]
        brakes.deployment_level = (
            deployment if (altitude > BRAKE_FLOOR and vertical_speed > 0) else 0.0
        )
        return (time_, brakes.deployment_level)

    flight = Flight(
        rocket=build_rocket(impulse, controller),
        environment=environment(),
        rail_length=5.2,
        inclination=85,
        heading=0,
        terminate_on_apogee=True,
    )
    return float(flight.apogee) - ELEVATION


def ols_slope(deployments: Sequence[float], apogees: Sequence[float], *covariates) -> float:
    """OLS coefficient on deployment, optionally adjusting for further columns."""
    columns = [np.ones(len(deployments)), np.asarray(deployments, dtype=float)]
    columns.extend(np.asarray(c, dtype=float) for c in covariates)
    design = np.column_stack(columns)
    return float(np.linalg.lstsq(design, np.asarray(apogees, dtype=float), rcond=None)[0][1])


def crew_deployment(lot: float, rng: np.random.Generator) -> float:
    """The confounded logging policy: a crew that brakes harder on lots it knows fly high."""
    return float(np.clip(0.5 + 0.30 * lot + rng.normal(scale=0.10), 0.0, TESTED_CEILING))


def paired_slope(lots: Sequence[float]) -> float:
    """Slope from counterfactual pairs: each lot flown at both levels, differenced within.

    Uses ``2 * len(lots)`` flights. The lot term is identical in both members of a pair, so it
    subtracts out exactly rather than being averaged over -- which is the whole variance reduction.
    """
    differences = [apogee_of(impulse_of(u), D_HIGH) - apogee_of(impulse_of(u), D_LOW) for u in lots]
    return float(np.mean(differences) / (D_HIGH - D_LOW))


def independent_slope(lots: Sequence[float], assignment: Sequence[float]) -> float:
    """Slope from the same flight count and the same two levels, but unpaired lots.

    The only difference from :func:`paired_slope` is that no two flights share a lot, which
    isolates the effect of pairing from every other design choice.
    """
    apogees = [apogee_of(impulse_of(u), d) for u, d in zip(lots, assignment, strict=True)]
    high = np.array([a for a, d in zip(apogees, assignment, strict=True) if d == D_HIGH])
    low = np.array([a for a, d in zip(apogees, assignment, strict=True) if d == D_LOW])
    return float((high.mean() - low.mean()) / (D_HIGH - D_LOW))


def part_a(rng: np.random.Generator, truth: float) -> None:
    print("A. Counterfactual pairing vs independent sampling, at equal flight count.")
    print("   Same two deployment levels, same lot distribution; only the pairing differs.\n")
    print(f"   {'flights':>8s} {'paired RMSE':>13s} {'independent RMSE':>18s} {'ratio':>7s}")
    for n_flights in SAMPLE_SIZES:
        paired_errors, independent_errors = [], []
        for _ in range(REPLICATES):
            lots = draw_lots(rng, n_flights // 2)
            paired_errors.append(paired_slope(lots) - truth)

            free_lots = draw_lots(rng, n_flights)
            assignment = np.array([D_LOW, D_HIGH] * (n_flights // 2))
            rng.shuffle(assignment)
            independent_errors.append(independent_slope(free_lots, assignment) - truth)
        paired_rmse = float(np.sqrt(np.mean(np.square(paired_errors))))
        independent_rmse = float(np.sqrt(np.mean(np.square(independent_errors))))
        ratio = independent_rmse / paired_rmse if paired_rmse > 0 else float("inf")
        print(f"   {n_flights:8d} {paired_rmse:11.1f} m {independent_rmse:16.1f} m {ratio:6.1f}x")
    print()


def part_b(rng: np.random.Generator, truth: float) -> None:
    print("B. Operational logs (confounded, lot recorded) vs randomizing, at equal flight count.")
    print(
        "   naive = regress on deployment; adjusted = also on the lot; randomized = experiment.\n"
    )
    print(
        f"   {'flights':>8s} {'naive':>12s} {'adjusted':>12s} {'randomized':>12s} {'overlap':>9s}"
    )
    for n_flights in ADJUST_SIZES:
        naive, adjusted, randomized, overlaps = [], [], [], []
        for _ in range(ADJUST_REPLICATES):
            lots = draw_lots(rng, n_flights)
            crew = [crew_deployment(u, rng) for u in lots]
            logged = [apogee_of(impulse_of(u), d) for u, d in zip(lots, crew, strict=True)]
            naive.append(abs(ols_slope(crew, logged) - truth))
            adjusted.append(abs(ols_slope(crew, logged, lots) - truth))

            free = rng.uniform(0.0, TESTED_CEILING, size=n_flights)
            experimental = [
                apogee_of(impulse_of(u), d)
                for u, d in zip(draw_lots(rng, n_flights), free, strict=True)
            ]
            randomized.append(abs(ols_slope(free, experimental) - truth))

            # Overlap diagnostic: how much of the deployment range the crew actually explores at a
            # given lot. Near zero means the adjustment is extrapolating, not adjusting.
            residual = np.asarray(crew) - (0.5 + 0.30 * np.asarray(lots))
            overlaps.append(float(residual.std()))
        print(
            f"   {n_flights:8d} {np.mean(naive):10.1f} m {np.mean(adjusted):10.1f} m "
            f"{np.mean(randomized):10.1f} m {np.mean(overlaps):8.3f}"
        )
    print()


def main() -> None:
    rng = np.random.default_rng(SEED)
    started = time.perf_counter()

    print("Reference slope: d(apogee)/d(deployment) under randomization.")
    truth = paired_slope(draw_lots(rng, TRUTH_PAIRS))
    print(
        f"   {truth:+.1f} m per unit deployment, from {2 * TRUTH_PAIRS} paired flights "
        f"({time.perf_counter() - started:.0f}s)\n"
    )

    part_a(rng, truth)
    part_b(rng, truth)
    print(f"Total wall clock {time.perf_counter() - started:.0f}s.")


if __name__ == "__main__":
    main()
