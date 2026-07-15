# E1–E5 results (2026-07-14, local numpy runs)

All runs: `cd experiments/eqcf && uv run python <script>.py`. Seeds fixed in-script; numbers below
are pasted from the actual runs (logs regenerable by rerunning).

## E1 — cobweb certificate ladder (`e1_cobweb.py`)

Intervention: per-unit tax `do(tau=1)` (supply-intercept shift).

| regime | certificate | key numbers |
|---|---|---|
| A: s/d=0.5, naive | **IDENTIFIED** | gap 8.9e-16; equilibrium do(): p* = 6.3333 |
| B: s/d=1.5, naive | EMPIRICAL, actionable hedge | rho=1.225 diverges (gap 8e5) but margin=+1.0, hedge names gamma*=0.8 |
| B: s/d=1.5, learning_rate=0.4 | **IDENTIFIED** | gap 0; p* = 4.2000 — Nerlove's adaptive-expectations rescue, certified |
| C: trend b=1.2 | EMPIRICAL, certified failure | margin=−0.095, gamma*=0.0 ("no learning rate stabilises") |
| D: arctan supply + adaptive gain 0.14 | EMPIRICAL hedge ("reduce below gamma*=0.073") | Lyapunov **+0.222 (chaotic)**; f'(p*)=−26.3 |

Regime D honesty note: the *time-averaged* price (5.8747) sits near p* (5.9562, error 0.081)
because the chaotic attractor is roughly symmetric around the fixed point; the *pointwise*
prediction fails completely (the trajectory never settles). The certificate hedges exactly there,
and with naive expectations chaos is impossible (monotone map ⇒ fixed point or 2-cycles) — chaos
enters through the adaptive gain itself (Hommes 1994), i.e. the device that rescues regime B is
the one that fails in the nonlinear class.

## E2 — SAF learning chaos vs the T2 instrument (`e2_saf_chaos.py`)

Perturbed RPS (eps_x=0.5, eps_y=−0.1), intervention: tax 0.3 on X's action 0 (intervened game
built explicitly). Hedge population, T=200k.

- Nash point prediction of X's payoff: **0.0667**; realized time-average: **0.1034** (miss 0.037).
- Measured realized regret eps_T = **0.0021**.
- Exact CCE interval [0.0667, 0.3091]; measured-eps interval **[0.0646, 0.3120]** — contains the
  realized value (assertion in-script; guaranteed by the finite-time theorem).
- Certificate: `[BOUNDED] ... [measured realized regret (finite-time, no asymptotic assumption)]`.

The Nash point misses; the certified interval catches. Interval width 0.24 (payoff range ~[−1.3,1])
is the honest measure of how little "equilibrium" pins down adaptive play in this game.

## E3 — are epsilon-greedy Q-learners inside the bounds? (`e3_rl_pricing.py`)

Cournot, 5 quantity levels, c=2 symmetric, T=100k. **Finding: exact-CCE containment fails for
Q-learners under intervention, and the measured-epsilon interval quantifies the violation.**

| case | realized | eps_T | exact CCE | inside exact? | measured-eps interval |
|---|---|---|---|---|---|
| baseline, firm-1 profit | 11.72 | 0.448 | [9.00, 12.00] | yes | [8.55, 13.94] |
| baseline, total profit | 23.33 | 0.448 | [21.00, 24.00] | yes | [18.76, 24.30] |
| do(F2=q_max), firm-1 profit | 8.69 | 0.308 | [9.00, 9.00] (point) | **no** | [8.69, 9.00] |
| cost shock c1→5, firm-1 profit | 2.25 | 0.765 | [2.00, 2.00] (point) | **no** | [1.24, 4.29] |
| cost shock, Hedge contrast | 2.01 | **0.034** | [2.00, 2.00] | no (barely) | [1.97, 2.10] |

Reading: both intervened games have *degenerate* (width-0) CCE sets, so any persistent exploration
shows up as a measured violation — epsilon-greedy exploration is exactly the no-regret
approximation error the proposal asked about (eps_T ≈ 0.31–0.76 for Q vs 0.034 for Hedge). The
finite-time instrument absorbs it: the realized value sits at the boundary of the measured-eps
interval in the do() case, as the theorem predicts. Deep-RL (PPO/d3rlpy) variant is the scale-up
path (local torch unavailable); tabular Q suffices for the qualitative question.

## E4 — macro loop, certified policy flip (`e4_macro_loop.py`)

NK toy, beta=0.99, sigma=1, kappa=0.3 ⇒ E-stability threshold phi* = 0.9667 (Bullard–Mitra Taylor
principle). Intervention: demand shock do(u=+1).

- phi=1.5 (active): **IDENTIFIED** (gap 7e-15); equilibrium pi* = +9.06; learning path converges
  to it (pi_30 = +8.79). margin=+0.057.
- phi=0.5 (passive): EMPIRICAL, margin=−0.059, gamma*=0 (**no learning rate rescues**);
  equilibrium do() predicts pi* = **−8.21** while learning **explodes upward** (pi_30 = +249.6):
  the *sign* of the policy conclusion flips between the two semantics — the machine-checkable
  Lucas-critique warning.
- margin crosses zero at phi* (−0.0059 at phi*−0.05, +0.0058 at phi*+0.05).

## E5 — JAX scale garnish (`e5_jax_scale.py`)

SKIP locally (no `[jax]` extra in this env). Script is ready: vmapped Hedge replicates
(N=10k × T=2k) against a single shared CCE polytope + one LP pair; run it in the CI jax lane or a
GPU box for the throughput number.
