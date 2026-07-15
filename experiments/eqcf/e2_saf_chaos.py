"""E2 — Sato-Akiyama-Farmer learning chaos as the adversarial test of the T2 instrument.

Perturbed rock-paper-scissors (SAF, PNAS 2002): tie payoffs (eps_x, eps_y) break the zero-sum
structure and make learning dynamics non-convergent/chaotic. The intervention is a tax tau on
player X's first action (payoff modification -> the intervened game G_do is built explicitly).

Pipeline: mixed-Nash point prediction of G_do  vs  realized time-averaged play of a no-regret
Hedge population on G_do  vs  the CCE interval (exact and measured-epsilon). Claim checked: the
Nash point prediction misses; the realized time-average lies inside the certified interval.

Run:  uv run python experiments/eqcf/e2_saf_chaos.py
"""

from __future__ import annotations

import numpy as np

from causalrl.games import mixed_nash_equilibria
from causalrl.magames import cce_bounds, cce_regret, certify_cce_do

import common

EPS_X, EPS_Y = 0.5, -0.1  # SAF perturbations (non-zero-sum: chaotic learning regime)
TAU = 0.3  # tax on X's action 0 (the intervention)
HORIZON = 200_000


def saf_matrices(eps_x: float, eps_y: float, tau: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    ux = np.array([[eps_x, -1.0, 1.0], [1.0, eps_x, -1.0], [-1.0, 1.0, eps_x]])
    uy = np.array([[eps_y, -1.0, 1.0], [1.0, eps_y, -1.0], [-1.0, 1.0, eps_y]]).T
    ux = ux.copy()
    ux[0, :] -= tau  # do(): tax X's first action
    return ux, uy


def main() -> None:
    print("=" * 78)
    print(f"E2 SAF chaos — eps_x={EPS_X}, eps_y={EPS_Y}, intervention: tax {TAU} on X action 0")
    print("=" * 78)
    ux, uy = saf_matrices(EPS_X, EPS_Y, tau=TAU)
    game_do = common.bimatrix_game(ux, uy, names=("X", "Y"))

    def payoff_x(profile) -> float:
        return float(ux[profile["X"], profile["Y"]])

    # Nash point prediction(s) of the intervened game.
    nash = mixed_nash_equilibria(game_do)
    nash_values = []
    for eq in nash:
        px = np.array([eq["X"].get(a, 0.0) for a in range(3)])
        py = np.array([eq["Y"].get(a, 0.0) for a in range(3)])
        nash_values.append(float(px @ ux @ py))
    print(f"Nash equilibria found: {len(nash)}; X-payoff point predictions: "
          f"{[f'{v:.4f}' for v in nash_values]}")

    # No-regret population on the intervened game.
    mu = common.hedge_population(game_do, HORIZON, seed=7)
    realized = common.expected_functional(mu, game_do.agents, payoff_x)
    eps_measured = cce_regret(game_do, mu)
    print(f"realized time-averaged X payoff over {HORIZON} rounds: {realized:.4f}")
    print(f"measured realized regret (epsilon_T): {eps_measured:.5f}")

    exact = cce_bounds(game_do, payoff_x)
    inflated = cce_bounds(game_do, payoff_x, epsilon=max(eps_measured, 0.0))
    print(f"CCE interval (exact):    [{exact.lower:.4f}, {exact.upper:.4f}]")
    print(f"CCE interval (measured): [{inflated.lower:.4f}, {inflated.upper:.4f}]")

    cert = certify_cce_do(game_do, payoff_x, no_regret=False, epsilon=max(eps_measured, 0.0))
    print(f"\ncertificate: {cert}")

    inside = inflated.lower - 1e-9 <= realized <= inflated.upper + 1e-9
    misses = [abs(realized - v) for v in nash_values]
    print(f"\nrealized inside measured-epsilon CCE interval: {inside}")
    if misses:
        print(f"Nash point-prediction error(s): {[f'{m:.4f}' for m in misses]}")
    assert inside, "T2 finite-time containment violated — impossible by construction"


if __name__ == "__main__":
    main()
