# E1–E7 results (2026-07-14/15, local runs; E3b on GPU)

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
built explicitly). Hedge population, T=200k. Honesty note: chaos of the learning dynamics at
*these* parameters was not separately verified in-run (SAF prove it for replicator dynamics in
their parameter families); the demonstrated facts are non-convergence to Nash of the time-average
and the certified containment — the label should be read as "non-convergent learning", with a
Lyapunov diagnostic of the coupled learning flow as the cheap follow-up.

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

NK toy, beta=0.99, sigma=1, kappa=0.3 ⇒ E-stability threshold phi* = 0.9667 — a
Taylor-principle-*type* threshold in this bespoke static-timing toy (phi* = 1 − (1−beta)/(kappa
sigma), which recovers the textbook phi > 1 exactly in the patient limit beta → 1; it is NOT the
Bullard–Mitra condition, whose timing and expectations differ). Intervention: demand shock
do(u=+1).

- phi=1.5 (active): **IDENTIFIED** (gap 7e-15); equilibrium pi* = +9.06; learning path converges
  to it (pi_30 = +8.79). margin=+0.057.
- phi=0.5 (passive): EMPIRICAL, margin=−0.059, gamma*=0 (**no learning rate rescues**);
  equilibrium do() predicts pi* = **−8.21** while learning **explodes upward** (pi_30 = +249.6):
  the *sign* of the policy conclusion flips between the two semantics — the machine-checkable
  Lucas-critique warning.
- margin crosses zero at phi* (−0.0059 at phi*−0.05, +0.0058 at phi*+0.05).

## E6 — the collusion probe: stateful learners escape the stage-game CCE (`e6_collusion.py`)

The decisive T2-boundary probe (2026-07-15). Cournot P=13−(q1+q2), c=1, q∈{2..6}: static Nash
(4,4) profit 16 each (welfare 32), joint-monopoly split (3,3) profit 18 each (welfare 36, stage
temptation 20). Calvano-style memory-1 Q-learners (state = last joint action, δ=0.95, slow
exploration decay, optimistic init), 2M steps, empirical joint over the last 500k.

**Key structural fact found by the LP: the exact stage-CCE interval for total profit is the
degenerate point {32}** — welfare is point-identified (T2 degeneracy rung) for *every* no-regret
population in this game.

| population | firm-1 profit | total profit vs CCE {32} | eps_T |
|---|---|---|---|
| memory-1 Q, seed 0 | 17.94 | **35.89 — outside** | **1.941** |
| memory-1 Q, seed 1 | 17.92 | **33.72 — outside** | **1.550** |
| memory-1 Q, seed 2 | 17.95 | **35.88 — outside** | **1.981** |
| stateless Q | 16.03 | 32.12 (ε-explained) | 0.481 |
| Hedge (no-regret) | 15.98 | 31.97 (ε-explained) | 0.016 |

Modal play for seeds 0/2: 96–98% mass on the joint-monopoly (3,3). Measured eps_T ≈ 1.94 ≈
(temptation 2.0) × (collusive mass 0.97): **the measured realized regret reads off the forgone
stage-game deviation gain that the learned punishment scheme is enforcing.** Qualification: a
large eps_T alone is not a collusion verdict — exploration also produces it (stateless Q reads
0.48 at Nash profits); the collusion signature is eps_T bounded away from zero *combined with*
concentrated supra-competitive play. The trichotomy in one game and one functional: no-regret
populations sit on the point-identified value; farsighted history-dependent populations escape it
and are detected, not missed — the certificate inflates to the measured-ε interval instead of
endorsing the static analysis.

**Verification + disclosure (adversarial check, 2026-07-15).** All four headline LP values
(welfare {32}, profit [12, 20]) carry machine-checked weak-duality certificates (dual-feasible
multipliers, slack ~1e-14), independent of the simplex internals. Structural cause found: the
1-unit grid creates *three* pure Nash on the total-quantity-8 anti-diagonal — (3,5), (4,4), (5,3)
— whose hull the CCE contains; welfare is constant (32) on that line, which is why the welfare
CCE is degenerate and the profit interval spans [12, 20] "structurally". Disclosure: the
degeneracy is thus partly a discretization artifact (continuous Cournot has a unique Nash); the
collusion *escape* (35.9 > 32) is NOT an artifact — no CCE weight can exceed 32 (duality-
certified), and (3,3)'s deviation gain is strict. Referee-proofing: rerun on an off-grid
discretization where Nash is unique, expect a nondegenerate welfare interval and the same escape.

## E7 — the basin probe: locally certified, globally selected (`e7_basins.py`)

The decisive T1-boundary probe (2026-07-15). Bistable mean dynamics x′ = tanh(3x) − x + u; SA
ensemble N=100k, gain 0.05, noise 0.1, x0 ~ N(0, 1.5²).

- u=0: stable equilibria ±0.995 (margins **+0.969 each** — both locally certified IDENTIFIED),
  unstable boundary at 0; ensemble splits 50.2/49.8.
- do(u=0.2): roots move to −0.782 / +1.199 (margins +0.892/+0.991), boundary to −0.105; ensemble
  re-partitions to 47.4/52.6 — **2.8% of the population crosses the basin boundary**.
- **Equilibrium-tracking at x+\* predicts +1.198; the population mixture mean is +0.261 — a gap of
  0.938** that no local certificate can see. Every local certificate is *correct*; the global
  interventional object is the basin-mass mixture over stable σ-solutions (the Blom–Bongers–Mooij
  caveat, measured). Missing diagnostic identified: a multiplicity-aware hedge (all stable roots +
  ensemble basin masses) for the nonlinear cyclic layer.
- **Dispersion-robustness (referee-proofing curve, 2026-07-15):** crossing mass under do(u=0.2)
  by initial spread — 0.25: +16.4%, 0.5: +8.4%, 1.0: +4.3%, 1.5: +2.8%, 2.0: +2.1%, 3.0: +1.4%;
  post-intervention mixture means 0.53 / 0.37 / 0.29 / 0.26 / 0.25 / 0.23 — the gap to the
  tracked root (+1.20) persists at every dispersion (0.67–0.97), and the selection effect is
  *largest* exactly when the population starts concentrated near the old boundary (the
  near-indifference case). The conclusion is not an artifact of the initial-condition choice.

## E3b — deep-RL populations vs the bounds, batched on GPU (`e3b_deep_rl.py`)

CUDA run, R=256 independent replicate populations per condition (one batched computation — the
replicate axis is the power), neural (2-layer MLP) policy-gradient learners with baseline —
the PPO family's core update — on the same Cournot game as E6, 20k steps, last-5k window.

| condition | eps_T quartiles | profit quartiles | outside exact CCE | supra-competitive |
|---|---|---|---|---|
| stateless PG | 0.000 / 0.001 / 0.002 | 16.00 / 16.00 / 16.00 | 0/256 | 0/256 |
| memory-1 PG | 0.000 / 0.001 / 0.001 | 16.00 / 16.00 / 16.00 | 0/256 | 0/256 |

**Finding (paired with E6), stated with its confounds: memory alone does not break the stage-game
bounds; the folk-theorem ingredient that does is the intertemporal objective.** These PG learners
maximize *immediate* stage reward (REINFORCE with baseline, no discounting), so they cannot
represent punishment threats **by construction** — the memory-1 condition isolates "state" from
"farsightedness", and with state alone all 512 populations stay empirically no-regret (median
eps_T = 0.001) on the Nash/point-identified values, where E6's *discounted* memory-1 Q-learners
(δ=0.95, bootstrapped values) escape to 35.9 welfare with eps_T ≈ 1.9. Two honest caveats:
(i) the E6/E3b horizon differs 100× (2M vs 20k steps), so this is not a matched-horizon
comparison; (ii) actual PPO deployments use γ > 0 — this run says nothing about *farsighted*
policy-gradient agents, which remain the open cell of the 2×2 (state × farsightedness). The
matched-horizon, γ-swept version of that 2×2 is the single most valuable follow-up experiment.

The certificate's new ε-sensitivity reads: at the measured median regret, the firm-1-profit
interval grows at 8.0 width-units per unit of additional regret — and its ε=0 width is already
8.0 (structural, the game): the looseness here is the game's, not the learners'.

## E8 — the open cell closed: farsightedness alone does not collude (`e8_farsighted_2x2.py`)

Matched-horizon decisive pair (2026-07-16, CUDA, T=2,000,000 per cell — E6's full horizon — R=64
replicates; farsightedness via n-step (n=32) truncated **Monte-Carlo discounted returns, no
critic**, isolating the intertemporal objective from value bootstrapping):

| memory-1 population | eps_T q25/50/75 | welfare q25/50/75 | collusive |
|---|---|---|---|
| PG myopic γ=0 (control) | 0.000 / 0.000 / 0.000 | 32.00 / 32.00 / 32.00 | 0/64 |
| PG farsighted γ=0.95 (THE cell) | 0.000 / 0.005 / 0.013 | 31.99 / 32.00 / 32.00 | 0/64 |
| Q bootstrapped+farsighted (anchor) | 1.745 / 1.941 / 1.961 | 34.80 / 35.88 / 35.88 | 3/3 |

**Finding: the folk-theorem prediction for the 2×2 was falsified in the interesting direction.**
State + motive (memory + γ=0.95) was NOT sufficient for these on-policy Monte-Carlo
policy-gradient populations — all 64 land exactly on the point-identified welfare 32 — while the
bootstrapped-value family (Q: greedy bootstrapped targets, optimistic init) escapes at the same
horizon in the same game. The collusion driver, within this design, narrows to **value
bootstrapping**, not memory and not farsightedness per se. This matches the known Q-vs-PG
asymmetry in the algorithmic-collusion literature and gives the certificate story its sharpest
form: the stage-CCE bounds hold empirically for on-policy learning families (myopic or
farsighted), and the measured-ε abstention fires exactly for the family known to collude.
Scope qualifications: MC returns truncated at n=32 (effective γ-horizon ≈ 20 < 32, so punishment
phases up to ~20 steps were representable — the motive channel was genuinely open); one game,
one seed batch per cell (R=64 replicates each); entropy-regularized on-policy PG has a known
bias toward stage-Nash play, which is part of the explanation, not a confound of it.

## E5 — JAX scale garnish (`e5_jax_scale.py`)

SKIP locally (no `[jax]` extra in this env). Script is ready: vmapped Hedge replicates
(N=10k × T=2k) against a single shared CCE polytope + one LP pair; run it in the CI jax lane or a
GPU box for the throughput number.
