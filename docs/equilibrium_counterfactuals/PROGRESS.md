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
| E6 collusion probe (T2 boundary) | RUN, 3 seeds; LP values duality-certified | memory-1 Q collude (35.9 vs degenerate CCE welfare {32}); eps_T ≈ 1.94 = temptation×mass (collusion signature with concentrated supra-competitive play); grid multiplicity disclosed |
| E7 basin probe (T1 boundary) | RUN | both margins +0.9 (locally right) yet do(u) mixture mean +0.26 vs tracked +1.20 — selection gap 0.94 |
| E3b deep-RL vs bounds (GPU, R=512 populations) | RUN | *myopic* PG stays no-regret/Nash even with memory — the folk-theorem ingredient is the intertemporal objective, not memory alone; farsighted-PG cell + matched horizon = key follow-up |
| Adversarial self-review | DONE 2026-07-15 | ADVERSARIAL_REVIEW.md — LP duality certificates, Main-Theorem exclusivity fixed, E3b/E4/E2 overclaims corrected, venue verdict |
| E8 matched-horizon decisive pair (GPU, 2M steps/cell) | RUN 2026-07-16 | farsighted MC-PG does NOT collude (64/64 at welfare 32) vs Q anchor 3/3 collusive — driver narrows to VALUE BOOTSTRAPPING; horizon confound closed |
| E7 dispersion curve | RUN 2026-07-15 | selection effect robust across spreads 0.25–3.0 (crossing 16.4%→1.4%, gap to tracked root persists 0.67–0.97) |
| local torch fix | `uv pip uninstall triton` (broken triton 3.7.0) | torch 2.12+cu130 fully works; reverts on next `uv sync` |

## Paper 1 draft (2026-07-16): the CLeaR-shaped causal-semantics paper

`paper/main.tex` + `paper/references.bib` + `paper/figs/` — full compilable draft of
**"When Can You Trust an Equilibrium Counterfactual? A certified identification trichotomy for
interventions on learning systems."** Contents: Main Theorem (trichotomy, global queries, with a
new explicit regularity Assumption (R) for the nonlinear class — sharper than THEORY.md §0.5,
which should be read as superseded by the paper's statement), T1 + equations-vs-maps corollary +
T1′ with FULL noise-condition proofs (appendices A/B: martingale assumptions, Kushner–Yin
limit-set, Pemantle/Brandière–Duflo converse), T2 as the set-identification rung (compact, with
provenance paragraph crediting the four adjacent literatures), certificate-ladder table,
experiments E1/E2/E4/E7 with the honesty notes from ADVERSARIAL_REVIEW.md baked in, related
work, limitations. Figures regenerated deterministically by `experiments/eqcf/paper_figs.py`
(E7 numbers reproduced exactly). Paper-split decision: this paper carries semantics+selection
(E1/E2/E4/E7); the EC-shaped companion carries T2-instrument+demarcation (E6/E8/E3b) and is NOT
folded in. Pre-submission TODOs are in the main.tex header comment (CLeaR style swap,
anonymization, owner's residual reading checks, Dogra/Mishra–Fox citation verification).

## Pre-submission round (2026-07-17): style, artifact, reading checks — DONE

- **CLeaR/PMLR style swap done**: main.tex now typesets under `jmlr` class `[pmlr]` mode
  (jmlr.cls + jmlrutils.sty + algorithm2e.sty vendored in `paper/`); 22pp, 0 errors,
  0 undefined refs. Swap to the CLeaR 2027 wrapper .sty when its author kit is released
  (CFP not yet up). **Page budget**: main text ends p.15; CLeaR's historical limit is 12pp
  excl. references/appendix — ~2.5pp of trimming remains (candidates in main.tex header).
- **Anonymized code artifact built**: `paper/artifact/` (+ artifact.zip) — `eqcert/` = minimal
  24-file closure of the four experiments, package renamed, CausalRLError→EqcertError,
  identification engine replaced by a Domain stub, identity scan clean; e1/e2/e4/e7 reproduce
  the paper's numbers standalone (README documents the runs).
- **Reading checks done via web research (3 agents), results folded into the paper**:
  - "Mishra–Fox 2024" was WRONG: actual = Mishra, Fox & Wooldridge, "Characterising
    Interventions in Causal Games", UAI 2024, PMLR 244:2560–2572. Fixed in bib + prose.
  - Dogra = FRBNY Staff Report **1093**, March 2024. Magnolfi–Roncoroni REStud 90(4):2006–2041,
    assumption = **BCE** (not no-regret) — prose corrected. STZ still unpublished (working
    paper). P–R JACM 55(3), optimize-over-CE NP-hardness confirmed in that same paper.
  - Hammond et al. read in full: their Def. 21 IS model-theoretic set-valued interventional
    prediction over equilibrium sets (credal-bounds remark; ε-equilibria via rationality
    relations; explicit cyclic-SCM bridge). §4 delta re-scoped to the identification layer;
    related work expanded; pin-the-action mapped to their pre/post-policy taxonomy.
  - **EC 2024–26 scan found the nearest neighbor the DD missed: Lomys–Magnolfi (EC 2025),
    "Estimation of Games under No Regret"** — ANR⇔CCE-set convergence, ε-BCCE set estimators,
    LP duality, counterfactual bounds. T2's delta re-anchored on the 4 surviving choices:
    measured realized regret (no algorithm-class assumption), containment on the INTERVENED
    game, abstention, duals as regret prices. Also added: Kline–Tamer 2024, Hartline 2026
    survey (NST ex-post folklore made explicit), Hartline et al. 2024 (regret audits),
    Weinberger 2023 (equilibrium-causal-models philosophy). **Paper 2's EC positioning must
    engage Lomys–Magnolfi head-on.** Deltas B (trichotomy) and C (E-stability bridge) came
    back CLEAR of direct overlap.

## Page-budget trim (2026-07-17): DONE

Main text (abstract through Conclusion) now ends exactly on p.12 in PMLR format (References
start p.13; 19pp total incl. appendices) — matching CLeaR's historical 12pp main-text limit.
Five trim rounds: prose tightening throughout, E1/E2 honesty notes condensed (regime-D
time-average note moved to Appendix E), §4 Provenance merged with the Related-Work
learning-in-games paragraph (deduplication; the four-choices delta vs Lomys–Magnolfi now lives
in §7 with a §4 pointer), figure heights reduced (figsize 3.0→2.55/2.6 in paper_figs.py, both
copies), and \looseness=-1 on four late paragraphs. No content dropped: all honesty items,
theorem statements, proofs, and novelty armor intact.

## Next (not started)

- Swap in the CLeaR 2027 wrapper .sty + re-fit when the author kit is released (CFP expected
  ~Oct 2026; cadence: CLeaR 2025 deadline was Nov 2, 2024). Attach artifact.zip at submission.
- Paper 2 (EC-shaped) prerequisites: E6 Calvano-scale seeds, off-grid Cournot with unique Nash,
  optional Lyapunov diagnostic for E2's learning flow.
- Multiplicity-aware hedge (all stable σ-solutions + ensemble basin masses) promoted from E7
  into the library's nonlinear-cyclic certificate layer (lib change → goes to main, not here).
- E3 deep-RL variant (PPO via [examples]/d3rlpy path) on CI or a GPU box — tabular Q stands in.
- Conformal finite-sample wrappers around the empirical checks (supporting cast; lib already
  ships `certify_conformal_interval`).
- Merge decision for this branch (owner call; keep out of a release until the docs/paper cut).
