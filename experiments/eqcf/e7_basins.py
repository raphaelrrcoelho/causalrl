"""E7 — The decisive T1-boundary probe: local certification vs global equilibrium selection.

Bistable scalar mean dynamics  x' = F(x) = tanh(beta x) - x + u  (beta = 3): two stable
equilibria x-* < 0 < x+* separated by an unstable one. At EACH stable equilibrium the linearized
stability margin is positive — the local T1 certificate says IDENTIFIED for populations starting
in that basin. But an intervention do(u) moves the basin boundary, so a dispersed population
redistributes across attractors: the *global* learning-limit do() is a mixture the equilibrium
analysis cannot see. This is Blom-Bongers-Mooij's no-SCM-for-initial-condition-dependence caveat
made quantitative: the certified-local claim is right, and its scope ends exactly at the basin
mass we measure here.

Method: stochastic-approximation ensemble (N learners, constant gain, Gaussian noise), initial
conditions ~ N(0, 1.5^2). Report per-equilibrium linearized margins (what the local certificate
sees) and the ensemble mass per basin under u = 0 vs do(u = 0.2) — the selection effect.

Run:  uv run python experiments/eqcf/e7_basins.py
"""

from __future__ import annotations

import numpy as np

BETA = 3.0
GAIN = 0.05
NOISE = 0.1
N_LEARNERS = 100_000
T_STEPS = 4_000


def mean_field(x: np.ndarray, u: float) -> np.ndarray:
    return np.tanh(BETA * x) - x + u


def equilibria(u: float) -> list[tuple[float, float]]:
    """(root, stability margin -F'(root)) for each equilibrium of tanh(bx) + u = x."""
    grid = np.linspace(-3.0, 3.0, 600001)
    f = mean_field(grid, u)
    roots = []
    sign_change = np.where(np.diff(np.sign(f)) != 0)[0]
    for k in sign_change:
        lo, hi = grid[k], grid[k + 1]
        for _ in range(60):  # bisection
            mid = 0.5 * (lo + hi)
            if mean_field(np.array([lo]), u)[0] * mean_field(np.array([mid]), u)[0] <= 0:
                hi = mid
            else:
                lo = mid
        root = 0.5 * (lo + hi)
        slope = BETA / np.cosh(BETA * root) ** 2 - 1.0  # F'(root)
        roots.append((float(root), float(-slope)))  # margin = -F' (positive = locally stable)
    return roots


def ensemble_basin_mass(u: float, seed: int = 0, spread: float = 1.5) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, spread, N_LEARNERS)
    for _ in range(T_STEPS):
        x = x + GAIN * (mean_field(x, u) + rng.normal(0.0, NOISE, N_LEARNERS))
    return {"negative": float(np.mean(x < 0)), "positive": float(np.mean(x > 0)),
            "mean_neg": float(x[x < 0].mean()) if np.any(x < 0) else float("nan"),
            "mean_pos": float(x[x > 0].mean()) if np.any(x > 0) else float("nan")}


def main() -> None:
    print("=" * 78)
    print("E7 basin probe — locally certified, globally selected")
    print("=" * 78)
    for u in (0.0, 0.2):
        label = f"u = {u}" + ("  (the intervention do(u=0.2))" if u else "  (baseline)")
        print(f"\n--- {label} ---")
        eq = equilibria(u)
        for root, margin in eq:
            kind = "STABLE (local T1 certificate: IDENTIFIED for its basin)" if margin > 0 else \
                   "unstable (basin boundary)"
            print(f"  equilibrium x* = {root:+.4f}   margin = {margin:+.3f}   {kind}")
        mass = ensemble_basin_mass(u)
        print(f"  SA-ensemble basin mass: negative {mass['negative']:.3f} "
              f"(mean {mass['mean_neg']:+.3f}), positive {mass['positive']:.3f} "
              f"(mean {mass['mean_pos']:+.3f})")

    # Referee-proofing: the crossing mass depends on the (arbitrary) initial dispersion — report
    # the whole curve rather than one point.
    print("\n--- selection effect vs initial dispersion (mass moved to + basin by do(u=0.2)) ---")
    for spread in (0.25, 0.5, 1.0, 1.5, 2.0, 3.0):
        b = ensemble_basin_mass(0.0, spread=spread)
        s = ensemble_basin_mass(0.2, spread=spread)
        mix_b = b["negative"] * b["mean_neg"] + b["positive"] * b["mean_pos"]
        mix_s = s["negative"] * s["mean_neg"] + s["positive"] * s["mean_pos"]
        print(f"  spread {spread:4.2f}: moved {s['positive'] - b['positive']:+.3f}   "
              f"mixture mean {mix_b:+.3f} -> {mix_s:+.3f}")

    base = ensemble_basin_mass(0.0)
    shifted = ensemble_basin_mass(0.2)
    moved = shifted["positive"] - base["positive"]
    mixture_mean = (
        shifted["negative"] * shifted["mean_neg"] + shifted["positive"] * shifted["mean_pos"]
    )
    tracked = equilibria(0.2)[-1][0]  # what equilibrium-tracking at x+* would predict
    print("\n=== verdict ===")
    print(f"do(u=0.2) moves {moved:+.1%} of the population across the basin boundary.")
    print(f"equilibrium-tracking prediction at x+*: {tracked:+.3f}; "
          f"population mixture mean: {mixture_mean:+.3f} "
          f"(gap {abs(tracked - mixture_mean):.3f} — the selection error a local certificate "
          f"cannot see)")
    print(
        "Both stable equilibria carry positive margins — each local certificate is correct — yet\n"
        "the population-level interventional outcome is a basin-mass mixture that shifts with the\n"
        "intervention. Point identification is local by nature; the honest global object is the\n"
        "mixture over stable sigma-solutions weighted by basin mass (the CCM caveat, measured).\n"
        "A multiplicity-aware hedge (report all stable roots + ensemble masses) is the missing\n"
        "diagnostic the certificate layer should gain when cyclic SCMs go nonlinear."
    )


if __name__ == "__main__":
    main()
