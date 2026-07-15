"""E3 — Do (approximately no-regret) RL populations respect the CCE bounds?

Two-firm discretized Cournot quantity competition: P = a - b (q1 + q2), profit_i = (P - c_i) q_i,
five quantity levels each. Independent stateless epsilon-greedy Q-learners are NOT no-regret, so
whether their interventional time-averages land inside CCE(G_do) is a real empirical question —
either answer is a finding. The measured realized regret quantifies the no-regret approximation
error, and the measured-epsilon interval is guaranteed to contain the realized average (T2).

Interventions: (a) do(firm2 = fixed quantity) — a pinned-action intervention through the library's
`do` parameter; (b) a cost shock to firm 1 — an intervened game built explicitly.

Run:  uv run python experiments/eqcf/e3_rl_pricing.py
(The tabular Q-learners stand in for the PPO/d3rlpy variant; local torch is unavailable and the
scale-up path is documented in RESULTS.md.)
"""

from __future__ import annotations

import numpy as np

from causalrl.magames import cce_bounds, cce_regret

import common

A, B = 12.0, 1.0
LEVELS = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
HORIZON = 100_000


def cournot_matrices(c1: float, c2: float) -> tuple[np.ndarray, np.ndarray]:
    q1 = LEVELS[:, None]
    q2 = LEVELS[None, :]
    price = A - B * (q1 + q2)
    return (price - c1) * q1, (price - c2) * q2


def run_case(label, game, functional, do, mu) -> None:
    realized = common.expected_functional(mu, game.agents, functional)
    eps = cce_regret(game, mu, do=do)
    exact = cce_bounds(game, functional, do=do)
    inflated = cce_bounds(game, functional, do=do, epsilon=max(eps, 0.0))
    inside_exact = exact.lower - 1e-9 <= realized <= exact.upper + 1e-9
    print(f"\n--- {label} ---")
    print(f"realized time-average: {realized:.4f}   measured regret eps_T: {eps:.4f}")
    print(f"exact CCE interval:    [{exact.lower:.4f}, {exact.upper:.4f}]"
          f"   realized inside: {inside_exact}")
    print(f"measured-eps interval: [{inflated.lower:.4f}, {inflated.upper:.4f}]"
          f"   (contains realized by construction)")


def main() -> None:
    print("=" * 78)
    print("E3 Cournot pricing with independent Q-learners (not no-regret)")
    print("=" * 78)
    u1, u2 = cournot_matrices(c1=2.0, c2=2.0)
    game = common.bimatrix_game(u1, u2, names=("F1", "F2"))

    def profit1(profile) -> float:
        return float(u1[profile["F1"], profile["F2"]])

    def welfare(profile) -> float:
        return float(u1[profile["F1"], profile["F2"]] + u2[profile["F1"], profile["F2"]])

    # Baseline: no intervention.
    mu = common.q_population(game, HORIZON, seed=11)
    run_case("baseline, firm-1 profit", game, profit1, None, mu)
    run_case("baseline, total profit", game, welfare, None, mu)

    # Intervention (a): pin firm 2 to the highest quantity.
    mu_do = common.q_population(game, HORIZON, seed=12, do={"F2": 4})
    run_case("do(F2 = q_max), firm-1 profit", game, profit1, {"F2": 4}, mu_do)

    # Intervention (b): cost shock to firm 1 (intervened game built explicitly).
    v1, v2 = cournot_matrices(c1=5.0, c2=2.0)
    game_shock = common.bimatrix_game(v1, v2, names=("F1", "F2"))

    def profit1_shock(profile) -> float:
        return float(v1[profile["F1"], profile["F2"]])

    mu_shock = common.q_population(game_shock, HORIZON, seed=13)
    run_case("cost shock c1: 2 -> 5, firm-1 profit", game_shock, profit1_shock, None, mu_shock)

    # Same intervened game, genuinely no-regret learners, for contrast.
    mu_hedge = common.hedge_population(game_shock, HORIZON, seed=14)
    run_case("cost shock, Hedge (no-regret) population", game_shock, profit1_shock, None, mu_hedge)


if __name__ == "__main__":
    main()
