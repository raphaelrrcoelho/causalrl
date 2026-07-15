# Equilibrium counterfactuals — progress map

Resume here. Branch: `equilibrium-counterfactuals` (off `main` = v2.1.0). Plan:
`docs/superpowers/plans/2026-07-14-equilibrium-counterfactuals.md`. Proposal (post-due-diligence
revision, verbatim): `PROPOSAL.md`. Theory: `THEORY.md`.

## Status (2026-07-14)

| piece | state | where |
|---|---|---|
| Novelty due diligence | DONE (externally, by project owner; verdicts folded into PROPOSAL.md §Positioning) | companion report held by owner |
| `solve_lp` pure-numpy two-phase simplex | DONE + tests | `src/causalrl/magames/_lp.py`, `tests/test_magames_lp.py` |
| `cce_polytope` / `cce_bounds` (ε-CCE) / `cce_regret` / `certify_cce_do` | DONE + tests, exported top-level | `src/causalrl/magames/cce.py`, `tests/test_magames_cce.py` |
| `stability_margin` / `spectral_abscissa` / `max_stable_learning_rate` | DONE + tests | `src/causalrl/experimental/cyclic/scm.py` |
| Comparator promotion (`learning_rate` mode, stability witness/hedges) | DONE + tests, backcompat | `src/causalrl/experimental/cyclic/comparator.py` |
| T1/T2/T3 statements + proofs (linear class / finite-time form) | DRAFTED | `THEORY.md` |
| E1 cobweb ladder | script + run | `experiments/eqcf/e1_cobweb.py`, `RESULTS.md` |
| E2 SAF chaos (T2 adversarial test) | script + run | `experiments/eqcf/e2_saf_chaos.py`, `RESULTS.md` |
| E3 RL pricing (Q-learners vs bounds) | script + run (tabular; PPO/d3rlpy = scale-up, local torch broken) | `experiments/eqcf/e3_rl_pricing.py`, `RESULTS.md` |
| E4 macro loop (certified policy flip) | script + run | `experiments/eqcf/e4_macro_loop.py`, `RESULTS.md` |
| E5 JAX scale garnish | script (skips without `[jax]` extra) | `experiments/eqcf/e5_jax_scale.py` |

## Design decision worth remembering

The proposal imagined one headline function (`compare_equilibrium_unrolling` returning all three
rungs). The shipped design splits by solvable object class — linear cyclic SCMs (T1/T3 rungs, the
comparator) vs finite games (T2 rungs, `certify_cce_do`) — because BOUNDED-via-CCE is only defined
for games. `THEORY.md` §3 states the unified ladder across both instruments.

## Leap round (2026-07-15, GPU box): decisive probes instead of batteries

| piece | state | outcome |
|---|---|---|
| ε-sensitivity instrument (LP duals in witness) | DONE + tests (lib, on local main) | certificate now prices marginal regret: structural vs statistical looseness |
| Main Theorem (unified trichotomy) + T1′ (hyperbolic) | DRAFTED in THEORY.md §0.5/§1 | the paper's spine |
| E6 collusion probe (T2 boundary) | RUN, 3 seeds | memory-1 Q collude (35.9 vs degenerate CCE welfare {32}); eps_T ≈ 1.94 = temptation×mass = collusion meter |
| E7 basin probe (T1 boundary) | RUN | both margins +0.9 (locally right) yet do(u) mixture mean +0.26 vs tracked +1.20 — selection gap 0.94 |
| E3b deep-RL vs bounds (GPU, R=512 populations) | RUN | PG-family stays exactly no-regret/Nash even with memory — collusion driver = value bootstrapping, not memory |
| local torch fix | `uv pip uninstall triton` (broken triton 3.7.0) | torch 2.12+cu130 fully works; reverts on next `uv sync` |

## Next (not started)

- Residual pre-submission reading checks listed in PROPOSAL.md (Hammond et al. pre-policy §,
  Mishra–Fox, Magnolfi–Roncoroni ANR, EC 2024–26 scan, CE-complexity citations).
- E3 deep-RL variant (PPO via [examples]/d3rlpy path) on CI or a GPU box — tabular Q stands in.
- Paper 1 (EC/NeurIPS) draft: T2 finite-time theorem + E2 + certify_cce_do; then T1 + E1/E4.
- Conformal finite-sample wrappers around the empirical checks (supporting cast; lib already
  ships `certify_conformal_interval`).
- Merge decision for this branch (owner call; keep out of a release until the docs/paper cut).
