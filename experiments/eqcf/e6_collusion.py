"""E6 — The decisive T2-boundary probe: can stateful learners escape the stage-game CCE?

Memory-1 Q-learners (state = last joint action; Calvano et al. 2020 style: discounting, slowly
decaying exploration, optimistic init) in Cournot quantity competition. History-dependent
strategies can sustain reward-punishment schemes (folk theorem), whose *stage-game* empirical
distribution is NOT a stage-game CCE: the measured deviation gain eps_T stays bounded away
from zero even as play stabilises. Either outcome is decisive for the T2 instrument's scope:

- eps_T -> 0 with Nash profits: stateful RL behaves no-regret; the exact-CCE bound's reach
  extends beyond its assumptions;
- eps_T bounded away from 0 with supra-competitive profits: tacit collusion sits OUTSIDE the
  stage-game polytope, the certificate honestly abstains/inflates, and the measured eps_T reads
  the forgone stage-game deviation gain (a collusion signature when combined with concentrated
  supra-competitive play — eps_T alone also arises from mere exploration).

Game: P = 13 - (q1 + q2), c = 1, quantities {2,3,4,5,6}. Static Nash (4,4): profit 16 each.
Joint-monopoly split (3,3): 18 each, with stage deviation temptation 20 (undercut... overproduce).

Comparison axis: memory-1 Q vs stateless Q vs Hedge (provably no-regret) on the same game.

Run:  uv run python experiments/eqcf/e6_collusion.py
"""

from __future__ import annotations

import numpy as np

from causalrl.magames import cce_bounds, cce_regret, certify_cce_do

import common

A, B_SLOPE, C = 13.0, 1.0, 1.0
LEVELS = np.array([2.0, 3.0, 4.0, 5.0, 6.0])
N_ACT = len(LEVELS)
T_TOTAL = 2_000_000
T_MEASURE = 500_000  # empirical joint distribution over the last window
DELTA = 0.95  # discount factor (the ingredient that makes punishment schemes learnable)
ALPHA = 0.15
BETA_EXPLORE = 3e-6  # eps_t = exp(-beta t): slow decay, Calvano-style


def profit(q_own: float, q_other: float, cost: float = C) -> float:
    return (A - B_SLOPE * (q_own + q_other) - cost) * q_own


def payoff_matrices() -> tuple[np.ndarray, np.ndarray]:
    q1, q2 = LEVELS[:, None], LEVELS[None, :]
    price = A - B_SLOPE * (q1 + q2)
    return (price - C) * q1, (price - C) * q2


def memory1_q_run(seed: int) -> dict[tuple[int, ...], float]:
    """Two independent memory-1 Q-learners; returns the stage-game empirical joint (last window)."""
    rng = np.random.default_rng(seed)
    u1, u2 = payoff_matrices()
    n_states = N_ACT * N_ACT
    optimistic = float(np.max(u1)) / (1.0 - DELTA)
    q1 = np.full((n_states, N_ACT), optimistic)
    q2 = np.full((n_states, N_ACT), optimistic)
    state = int(rng.integers(n_states))
    counts = np.zeros((N_ACT, N_ACT))
    explore_thresholds = np.exp(-BETA_EXPLORE * np.arange(T_TOTAL))
    uniform_draws = rng.random((T_TOTAL, 2))
    random_actions = rng.integers(0, N_ACT, size=(T_TOTAL, 2))
    for t in range(T_TOTAL):
        eps = explore_thresholds[t]
        a1 = random_actions[t, 0] if uniform_draws[t, 0] < eps else int(np.argmax(q1[state]))
        a2 = random_actions[t, 1] if uniform_draws[t, 1] < eps else int(np.argmax(q2[state]))
        r1, r2 = u1[a1, a2], u2[a1, a2]
        nxt = a1 * N_ACT + a2
        q1[state, a1] += ALPHA * (r1 + DELTA * np.max(q1[nxt]) - q1[state, a1])
        q2[state, a2] += ALPHA * (r2 + DELTA * np.max(q2[nxt]) - q2[state, a2])
        state = nxt
        if t >= T_TOTAL - T_MEASURE:
            counts[a1, a2] += 1.0
    counts /= counts.sum()
    return {(i, j): float(counts[i, j]) for i in range(N_ACT) for j in range(N_ACT) if counts[i, j]}


def summarize(label: str, game, mu, u1, u2) -> tuple[float, float]:
    def profit1(profile) -> float:
        return float(u1[profile["F1"], profile["F2"]])

    def welfare(profile) -> float:
        return float(
            u1[profile["F1"], profile["F2"]] + u2[profile["F1"], profile["F2"]]
        )

    realized = common.expected_functional(mu, game.agents, profit1)
    total = common.expected_functional(mu, game.agents, welfare)
    eps = cce_regret(game, mu)
    exact = cce_bounds(game, profit1)
    exact_w = cce_bounds(game, welfare)
    inside = exact.lower - 1e-9 <= realized <= exact.upper + 1e-9
    inside_w = exact_w.lower - 1e-9 <= total <= exact_w.upper + 1e-9
    print(f"\n--- {label} ---")
    print(f"  avg firm-1 profit: {realized:.3f}   (static Nash 16.0, joint-monopoly split 18.0)")
    print(f"  measured stage-game regret eps_T: {eps:.4f}")
    print(f"  exact stage-CCE interval for firm-1 profit: [{exact.lower:.3f}, {exact.upper:.3f}]"
          f"   realized inside: {inside}")
    print(f"  total profit {total:.3f} vs exact stage-CCE interval "
          f"[{exact_w.lower:.3f}, {exact_w.upper:.3f}]   realized inside: {inside_w}")
    top = sorted(mu.items(), key=lambda kv: -kv[1])[:3]
    pretty = ", ".join(f"q=({LEVELS[i]:.0f},{LEVELS[j]:.0f}):{w:.2f}" for (i, j), w in top)
    print(f"  modal joint quantities: {pretty}")
    return realized, eps


def main() -> None:
    print("=" * 78)
    print("E6 collusion probe — memory-1 Q vs stateless Q vs Hedge, same Cournot stage game")
    print("=" * 78)
    u1, u2 = payoff_matrices()
    game = common.bimatrix_game(u1, u2, names=("F1", "F2"))

    def profit1(profile) -> float:
        return float(u1[profile["F1"], profile["F2"]])

    results = {}
    mus = {}
    for seed in (0, 1, 2):
        mu = memory1_q_run(seed)
        mus[seed] = mu
        results[seed] = summarize(f"memory-1 Q-learners (seed {seed})", game, mu, u1, u2)

    mu_stateless = common.q_population(game, 500_000, seed=3)
    summarize("stateless Q-learners", game, mu_stateless, u1, u2)
    mu_hedge = common.hedge_population(game, 500_000, seed=4)
    summarize("Hedge (provably no-regret)", game, mu_hedge, u1, u2)

    eps_values = [eps for _, eps in results.values()]
    profits = [p for p, _ in results.values()]
    print("\n=== verdict ===")
    print(f"memory-1 Q: profits {['%.2f' % p for p in profits]}, eps_T {['%.3f' % e for e in eps_values]}")
    cert = certify_cce_do(
        game, profit1, no_regret=False, epsilon=max(cce_regret(game, mus[0]), 0.0)
    )
    print(f"certificate at measured eps (seed 0): {cert}")
    print(
        "reading: supra-competitive profits + eps_T bounded away from 0 = history-dependent play\n"
        "outside the stage-game CCE; the certificate inflates/abstains instead of endorsing the\n"
        "static analysis, and measured eps_T acts as a collusion meter. If instead eps_T ~ 0 at\n"
        "Nash profits, stateful RL behaved no-regret and the exact bound's reach extends."
    )


if __name__ == "__main__":
    main()
