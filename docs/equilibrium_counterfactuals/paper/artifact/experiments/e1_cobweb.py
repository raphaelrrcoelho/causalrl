"""E1 — Cobweb: the certificate ladder on the canonical stage.

One market, four regimes. Demand D(p) = a - d p; supply S(p^e) = c + s p^e under naive
expectations p^e_{t+1} = p_t; a per-unit tax tau enters as do() (a cost shock shifting the
supply intercept). The cyclic SCM has variables (p, pe):

    p  = -(s/d) pe + (a - c + s*tau)/d ,   pe = p .

Regimes:
  A  s/d < 1   : naive unrolling contracts            -> IDENTIFIED (equilibrium do = learning do)
  B  s/d > 1   : naive cobweb oscillation diverges, but the mean dynamics are stable
                 (margin > 0) -> actionable hedge; rerun with learning_rate < gamma*
                 (Nerlove's adaptive-expectations rescue, certified)         -> IDENTIFIED
  C  trend-extrapolating demand b > 1: margin < 0, no gain rescues           -> certified failure
  D  nonlinear (arctan) supply + adaptive expectations (Hommes 1994): with naive expectations a
     monotone cobweb map can only fix or 2-cycle, so chaos needs the adaptive gain -- the very
     device that rescues regime B becomes chaotic in the nonlinear class (positive Lyapunov).
     The linearized certificate correctly hedges at that gain (gamma > gamma*) -> EMPIRICAL,
     and the realized long-run average is quantitatively far from the equilibrium do().

Run:  uv run python experiments/eqcf/e1_cobweb.py
"""

from __future__ import annotations

import numpy as np

from eqcert.experimental.cyclic import LinearCyclicSCM, compare_equilibrium_unrolling

import common

A, D, C = 10.0, 1.0, 1.0  # demand intercept/slope, supply intercept
TAU = 1.0  # per-unit tax (the intervention): supply intercept c -> c + s*tau equivalent shift


def cobweb_scm(s: float, tau: float = 0.0) -> LinearCyclicSCM:
    return LinearCyclicSCM(
        [[0.0, -s / D], [1.0, 0.0]],
        ["p", "pe"],
        noise_mean=[(A - C + s * tau) / D, 0.0],
    )


def trend_scm(b: float, tau: float = 0.0) -> LinearCyclicSCM:
    # Speculative demand: p = b * pe + u (trend extrapolation); b > 1 is E-unstable.
    return LinearCyclicSCM([[0.0, b], [1.0, 0.0]], ["p", "pe"], noise_mean=[1.0 + tau, 0.0])


def report(label: str, cert) -> None:
    print(f"\n--- {label} ---")
    print(cert)
    w = cert.witness.detail if cert.witness else {}
    print(
        f"    margin={w.get('stability_margin'):+.3f}  rho={w.get('spectral_radius'):.3f}  "
        f"gamma*={w.get('max_stable_learning_rate'):.3f}  gap={w.get('gap'):.3g}"
    )


def main() -> None:
    print("=" * 78)
    print("E1 cobweb ladder — intervention: per-unit tax do(tau=1)")
    print("=" * 78)

    # Regime A: s/d = 0.5 -> contractive.
    scm = cobweb_scm(s=0.5, tau=TAU)
    report("A: s/d=0.5, naive unrolling", compare_equilibrium_unrolling(scm, horizon=400, tol=1e-6))
    print(f"    equilibrium do(tax): p* = {scm.solve().mean_dict()['p']:.4f}")

    # Regime B: s/d = 1.5 -> naive cobweb diverges, margin still positive.
    scm_b = cobweb_scm(s=1.5, tau=TAU)
    naive = compare_equilibrium_unrolling(scm_b, horizon=60)
    report("B: s/d=1.5, naive unrolling (period-2 divergence)", naive)
    gamma_star = scm_b.max_stable_learning_rate()
    rescued = compare_equilibrium_unrolling(
        scm_b, horizon=4000, tol=1e-6, learning_rate=0.5 * gamma_star
    )
    report(f"B: s/d=1.5, adaptive learning_rate={0.5 * gamma_star:.3f}", rescued)
    print(f"    equilibrium do(tax): p* = {scm_b.solve().mean_dict()['p']:.4f}")

    # Regime C: trend extrapolation b=1.2 -> margin < 0, certified non-identification.
    scm_c = trend_scm(b=1.2, tau=TAU)
    report("C: trend b=1.2, naive", compare_equilibrium_unrolling(scm_c, horizon=60))
    report(
        "C: trend b=1.2, any learning rate",
        compare_equilibrium_unrolling(scm_c, horizon=400, learning_rate=0.2),
    )

    # Regime D: nonlinear (arctan) supply + adaptive expectations pe' = (1-g) pe + g f(pe).
    lam, scale = 12.0, 2.2
    p_star_grid = np.linspace(0.0, 10.0, 200001)

    def price_map(pe: float, tau: float = TAU) -> float:
        supply = C + scale * (np.arctan(lam * (pe - tau - 5.0)) / np.pi + 0.5) * 4.0
        return float((A - supply) / D)

    values = np.array([price_map(p) for p in p_star_grid])
    p_star = float(p_star_grid[np.argmin(np.abs(values - p_star_grid))])
    slope = (price_map(p_star + 1e-6) - price_map(p_star - 1e-6)) / 2e-6
    linearized = LinearCyclicSCM(
        [[0.0, slope], [1.0, 0.0]], ["p", "pe"], noise_mean=[p_star * (1 - slope), 0.0]
    )
    gamma_star_lin = linearized.max_stable_learning_rate()

    chaotic_gamma, lyap = None, -np.inf
    for gamma in np.arange(0.08, 0.52, 0.02):
        ly = common.lyapunov_1d(
            lambda pe, g=gamma: (1.0 - g) * pe + g * price_map(pe), x0=p_star + 0.01, n=8000
        )
        if ly > lyap:
            chaotic_gamma, lyap = float(gamma), ly
        if ly > 0.01:
            break
    assert chaotic_gamma is not None
    report(
        f"D: nonlinear supply, adaptive gain gamma={chaotic_gamma:.2f} (linearized at p*)",
        compare_equilibrium_unrolling(linearized, horizon=200, learning_rate=chaotic_gamma),
    )
    pe, trajectory = p_star + 0.01, []
    for _ in range(20000):
        pe = (1.0 - chaotic_gamma) * pe + chaotic_gamma * price_map(pe)
        trajectory.append(price_map(pe))
    realized = float(np.mean(trajectory[2000:]))
    print(f"    fixed point p* = {p_star:.4f}, map slope f'(p*) = {slope:.3f}, "
          f"gamma* = {gamma_star_lin:.4f}")
    print(f"    largest Lyapunov exponent at gamma={chaotic_gamma:.2f}: {lyap:.4f}  "
          f"({'CHAOTIC' if lyap > 0 else 'regular'})")
    print(
        f"    equilibrium do() prediction {p_star:.4f} vs realized long-run mean {realized:.4f} "
        f"(abs error {abs(realized - p_star):.4f}) — wrong exactly where the certificate hedges"
    )


if __name__ == "__main__":
    main()
