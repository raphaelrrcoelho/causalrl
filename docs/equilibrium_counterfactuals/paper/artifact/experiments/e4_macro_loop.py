"""E4 — The macro loop where the paradox bites: a certified policy-conclusion flip.

New-Keynesian toy (Dogra setting), reduced to its expectations loop. Structural block:
x = -sigma (i - pi_e) + u,  pi = beta pi_e + kappa x,  i = phi pi (policy rule). Solving the
within-period block given expectations pi_e yields the expectations recursion

    pi = T(phi) pi_e + u',     T(phi) = (beta + kappa sigma) / (1 + kappa sigma phi)

with naive expectations pi_e' = pi closing the cyclic SCM  B = [[0, T], [1, 0]].
E-stability threshold: T(phi) < 1  <=>  phi > phi* = 1 - (1 - beta) / (kappa sigma) — a
Taylor-principle-TYPE threshold for this bespoke static-timing toy (it recovers phi > 1 exactly
in the patient limit beta -> 1; it is not the Bullard-Mitra condition, whose model timing and
expectations differ).

The exhibit: under a positive demand shock do(u = +1),
  - active rule (phi = 1.5): equilibrium do() and learning do() agree (certified IDENTIFIED);
  - passive rule (phi = 0.5): T > 1, so equilibrium do() predicts a *negative* inflation response
    pi* = u / (1 - T) < 0, while the learning dynamics diverge to +infinity — the sign of the
    policy conclusion flips between the two semantics, and the certificate says margin < 0,
    no learning rate rescues it (certified non-identification).

Run:  uv run python experiments/eqcf/e4_macro_loop.py
"""

from __future__ import annotations

import numpy as np

from eqcert.experimental.cyclic import LinearCyclicSCM, compare_equilibrium_unrolling

BETA, SIGMA, KAPPA = 0.99, 1.0, 0.3
SHOCK = 1.0


def t_slope(phi: float) -> float:
    return (BETA + KAPPA * SIGMA) / (1.0 + KAPPA * SIGMA * phi)


def macro_scm(phi: float, shock: float = 0.0) -> LinearCyclicSCM:
    return LinearCyclicSCM(
        [[0.0, t_slope(phi)], [1.0, 0.0]], ["pi", "pi_e"], noise_mean=[shock, 0.0]
    )


def learning_path(phi: float, shock: float, steps: int = 30) -> list[float]:
    pi_e, path = 0.0, []
    for _ in range(steps):
        pi = t_slope(phi) * pi_e + shock
        path.append(pi)
        pi_e = pi  # naive expectations
    return path


def main() -> None:
    phi_star = 1.0 - (1.0 - BETA) / (KAPPA * SIGMA)
    print("=" * 78)
    print(f"E4 macro loop — E-stability threshold phi* = {phi_star:.4f} "
          f"(T(phi*) = {t_slope(phi_star):.4f})")
    print("=" * 78)

    for phi in (1.5, 0.5):
        scm = macro_scm(phi, SHOCK)
        cert = compare_equilibrium_unrolling(scm, horizon=2000, tol=1e-6)
        sol = scm.solve()
        eq_pi = sol.mean_dict()["pi"]
        path = learning_path(phi, SHOCK)
        print(f"\n--- policy rule phi = {phi} (T = {t_slope(phi):.4f}) ---")
        print(cert)
        w = cert.witness.detail
        print(f"    margin={w['stability_margin']:+.4f}  gamma*={w['max_stable_learning_rate']:.4f}")
        print(f"    equilibrium do(u=+1): pi* = {eq_pi:+.4f}")
        print(f"    learning do(u=+1), first steps: "
              f"{['%+.3f' % v for v in path[:6]]} ... pi_30 = {path[-1]:+.3f}")
        if t_slope(phi) > 1.0:
            print("    SIGN FLIP: equilibrium predicts a negative inflation response; "
                  "learning explodes upward. The certificate is the machine-checkable "
                  "Lucas-critique warning.")

    # The certified flip is sharp at phi*: margin crosses zero exactly there.
    margins = {phi: macro_scm(phi).stability_margin() for phi in
               (phi_star - 0.05, phi_star + 0.05)}
    print(f"\nmargin just below/above phi*: {margins}")


if __name__ == "__main__":
    main()
