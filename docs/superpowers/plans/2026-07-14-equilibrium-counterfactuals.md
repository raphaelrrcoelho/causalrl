# Certified Equilibrium Counterfactuals Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the "When Can You Trust an Equilibrium Counterfactual?" flagship: CCE partial-identification instrument (T2), E-stability comparator promotion (T1), certified-failure diagnostics (T3), experiments E1–E5, theory + novelty docs.

**Architecture:** Two certified front doors, matching the two solvable object classes the library actually ships. (1) Finite games (`magames`): `cce_polytope`/`cce_bounds`/`certify_cce_do` — LP-computed BOUNDED certificates over the coarse-correlated-equilibrium polytope of the intervened game, with an IDENTIFIED degeneracy rung and an EMPIRICAL no-no-regret rung. Pure-numpy two-phase simplex (`_lp.py`) since core has no scipy. (2) Linear cyclic SCMs (`experimental/cyclic`): `stability_margin` (E-stability margin, mean-ODE spectral abscissa) + `compare_equilibrium_unrolling(..., learning_rate=γ)` unrolling the damped/constant-gain adaptive dynamics `x⁺ = x + γ(Bx + u − x)` — IDENTIFIED exactly on the E-stable class (T1), hedged with `max_stable_learning_rate` otherwise (T3). The proposal imagined one function; the honest shipped design splits by object class and the THEORY doc states the bridge.

**Tech Stack:** numpy only in `src/` (generality lint bans domain nouns there — economics vocabulary stays in `experiments/eqcf/` and `docs/`). pytest+hypothesis for tests. Experiments pure numpy (local torch broken; CI is truth).

## Global Constraints

- Core deps stay `networkx>=3.4, gymnasium>=1.0, numpy>=2.0` — **no scipy anywhere in `src/`**.
- Generality lint (invariant I7): zero application-domain nouns in `src/causalrl` identifiers/docstrings.
- Invariant I2: `LearnerTopology` caps point-equilibrium `Kind`s; the CCE certificate's BOUNDED claim is licensed by an explicit, checkable `no-regret` `Assumption` instead — never silently.
- Invariant I3: never fabricate a solution — hedge instead.
- Backcompat: `compare_equilibrium_unrolling` default behaviour (naive unrolling) unchanged; existing tests must pass untouched.
- Commits: no Claude co-author trailer.
- Branch: `equilibrium-counterfactuals` off `main` (v2.1.0).

---

### Task 1: Pure-numpy LP solver

**Files:**
- Create: `src/causalrl/magames/_lp.py`
- Test: `tests/test_magames_lp.py`

**Interfaces:**
- Produces: `LPResult` dataclass `(status: str, x: FloatArray | None, value: float | None)` with `status ∈ {"optimal", "infeasible", "unbounded"}`; `solve_lp(c, *, a_ub=None, b_ub=None, a_eq=None, b_eq=None, tol=1e-9) -> LPResult` minimising `c·x` s.t. `A_ub x ≤ b_ub`, `A_eq x = b_eq`, `x ≥ 0`.

- [x] **Step 1: failing tests** — known optima (min/max over the probability simplex with an equality constraint; a 2-var LP with inequality binding), infeasible detection (`x≥0, x₁+x₂=−1`), unbounded detection (`min −x₁` unconstrained above), degenerate ties (Bland's rule terminates).
- [x] **Step 2: run, verify fail** (`ModuleNotFoundError`).
- [x] **Step 3: implement** — standard-form conversion (slacks on ≤ rows, flip rows so b≥0), Phase-1 artificials driven to 0 else `infeasible`, Phase-2 with Bland's anti-cycling rule, reduced-cost optimality at `tol`.
- [x] **Step 4: tests pass.**
- [x] **Step 5: commit** `feat(magames): pure-numpy two-phase simplex for polytope bounds`.

### Task 2: CCE polytope, bounds, regret, certificate

**Files:**
- Create: `src/causalrl/magames/cce.py`
- Modify: `src/causalrl/magames/__init__.py`, `src/causalrl/__init__.py` (lazy map + `__all__`)
- Test: `tests/test_magames_cce.py`

**Interfaces:**
- Consumes: `solve_lp`/`LPResult` from Task 1; `CausalGame` (`agents`, `actions`, `utilities[agent][profile_tuple]`); `Interval` from `causalrl.identification.bounds`; `Certificate/Kind/Assumption/Witness/Hedge/EstimandSpec/Provenance`.
- Produces:
  - `CCEPolytope` frozen dataclass: `profiles: tuple[tuple[int,...],...]` (joint actions in `game.agents` order, restricted to `do`-consistent ones), `deviation_gains: FloatArray` shape `(n_constraints, n_profiles)` — row (free agent i, deviation a′): `u_i(a′, s₋ᵢ) − u_i(s)`, `constraint_labels: tuple[tuple[str,int],...]`, `agents: tuple[str,...]`.
  - `cce_polytope(game, *, do: Mapping[str,int] | None = None) -> CCEPolytope`
  - `cce_bounds(game, functional: Callable[[Mapping[str,int]], float], *, do=None, epsilon: float | Mapping[str, float] = 0.0) -> Interval` — two LPs (min and max of `Σ μ(s) f(s)` over `deviation_gains·μ ≤ ε(agent), Σμ=1, μ≥0`); the ε-CCE relaxation is the **finite-time, anytime-valid** form (post-due-diligence T2): with ε = the *measured realized regret* at horizon T, the realized empirical joint distribution is exactly ε-CCE-feasible — no asymptotics. Polytope ⊆ simplex ⇒ bounded; infeasibility impossible for ε ≥ 0 (Nash exists) → `RuntimeError` guard.
  - `cce_regret(game, weights: Mapping[tuple[int,...], float] | Sequence[float], *, do=None) -> float` — max deviation gain of a joint distribution: the measured realized regret that feeds `epsilon`.
  - `certify_cce_do(game, functional, *, do=None, no_regret=True, epsilon=0.0, tol=1e-9, seed=0) -> Certificate` ladder: width ≤ tol & no_regret → IDENTIFIED (value=float midpoint); no_regret (asymptotic assumption) or measured `epsilon` supplied → BOUNDED (value=Interval over the ε-CCE); vacuous inflated interval (width ≥ full payoff-functional range) → EMPIRICAL **abstention** hedge; no_regret=False with no measured epsilon → EMPIRICAL (interval as evidence, hedge "no-regret not established"). Witness: interval, width, epsilon, n_profiles, n_constraints, do. Assumptions: `finite-game` (checkable), `no-regret` (checkable, diagnostic = `cce_regret` of realized play; when `epsilon` is measured the assumption is *discharged at the run horizon*, the T2(i) delta).

- [x] **Step 1: failing tests** — dominant-strategy game (defect-style 2×2): CCE is the single dominant profile ⇒ width 0 ⇒ IDENTIFIED; anti-coordination game ("chicken" payoffs, kept domain-neutral in test names): bounds strictly contain all Nash payoffs, width > 0 ⇒ BOUNDED with `Interval`; zero-sum matching-pennies: value functional width ≈ 0; `do` pinning one agent: polytope = other agent's best responses; `cce_regret` 0 at dominant profile, >0 at uniform; `no_regret=False` ⇒ EMPIRICAL + hedge; every Nash mixture's payoff within bounds.
- [x] **Step 2: run, fail.**
- [x] **Step 3: implement** (LP columns = do-consistent profiles; constraints only for free agents).
- [x] **Step 4: pass;** public-API test still green after exports.
- [x] **Step 5: commit** `feat(magames): CCE polytope partial identification + certify_cce_do ladder`.

### Task 3: stability_margin on LinearCyclicSCM

**Files:**
- Modify: `src/causalrl/experimental/cyclic/scm.py`
- Test: `tests/test_cyclic_scm.py` (extend)

**Interfaces:**
- Produces (methods on `LinearCyclicSCM`):
  - `spectral_abscissa() -> float` = `max Re λ(B − I)` — the mean-ODE Jacobian abscissa.
  - `stability_margin() -> float` = `−spectral_abscissa()` = `1 − max Re λ(B)`; `> 0` iff the associated ODE/adaptive-learning dynamics are locally stable (E-stability). Strictly weaker than contractivity: `B=[[−2]]` has `ρ=2` (non-contractive) but margin `3` (stable).
  - `max_stable_learning_rate() -> float` = `min_i −2·Re νᵢ / |νᵢ|²` over eigenvalues `νᵢ` of `B − I` with `Re νᵢ < 0`... returns `0.0` when not stable, `inf` for the empty edge case `dim==0`; capped at nothing (caller clamps to (0,1]).

- [x] Steps: failing tests on `[[−2]]` (margin 3, γ*=2/3... ν=−3 ⇒ −2(−3)/9=2/3), diag(1.5) (margin −0.5, γ*=0), nilpotent `[[0,2],[0,0]]` (margin 1, ν=−1 ⇒ γ*=2 — any γ≤1 works), complex pair rotation ρ>1 with Re<1 stable → margin>0; then implement; pass; commit `feat(cyclic): E-stability margin + max stable learning-rate diagnostics`.

### Task 4: comparator promotion (T1 instrument)

**Files:**
- Modify: `src/causalrl/experimental/cyclic/comparator.py`
- Test: `tests/test_cyclic_comparator.py` (extend; existing tests untouched)

**Interfaces:**
- `compare_equilibrium_unrolling(scm, *, do=None, horizon=256, tol=1e-3, seed=0, learning_rate: float | None = None)`.
  - `learning_rate=None`: exactly today's behaviour + witness gains `stability_margin`, `spectral_abscissa`, `max_stable_learning_rate` fields; the non-contractive hedge detail additionally says when the system is nonetheless stable (margin>0) that a damped `learning_rate < max_stable_learning_rate` unrolling converges.
  - `learning_rate=γ`: iterate `x⁺ = x + γ(Bx + u − x)`; convergence criterion `ρ(I + γ(B−I)) < 1`; IDENTIFIED when that holds and gap ≤ tol (same fixed point `(I−B)⁻¹u` — γ-invariant); hedges otherwise, with γ*, margin in detail. Validation: `0 < γ ≤ 1` else `ValueError`.

- [x] Steps: failing tests — `[[−2]]` naive → EMPIRICAL/non-contractive with margin 3 in witness + actionable hedge; same SCM `learning_rate=0.3` (ρ(M)=0.1) → IDENTIFIED; `learning_rate=0.9 > γ*=2/3` → EMPIRICAL hedge carrying `max_stable_learning_rate`; margin<0 SCM + any γ → EMPIRICAL; `learning_rate=1` ≡ naive dynamics; γ∉(0,1] raises. Implement; pass; commit `feat(cyclic): adaptive-learning unrolling mode — E-stable class certifies IDENTIFIED`.

### Task 5: theory + novelty docs

**Files:**
- Create: `docs/equilibrium_counterfactuals/PROPOSAL.md` (user text verbatim), `THEORY.md`, `NOVELTY.md`, `PROGRESS.md`

**Content:** THEORY.md — notation (two do() operators); **T1** for the linear class: equilibrium do = `(I−B_do)⁻¹E[u_do]`; constant-gain adaptive dynamics converge locally iff `B_do − I` Hurwitz (E-stability) — for γ<γ*; naive unrolling requires the strictly stronger `ρ(B_do)<1`; hence learning-limit do ≡ equilibrium do on the E-stable class, certified by `stability_margin`; proof by eigenvalue algebra, both directions. **T2**: no-regret population ⇒ empirical joint dist → CCE(G_do) (Hart–Mas-Colell); closedness ⇒ any limit of time-averaged linear functional lies in `[min,max]` over the polytope (LPs); degeneracy corollary (width 0 ⇔ equilibrium point prediction valid for every no-regret population); ε-regret robustness remark (bounds inflate continuously — LP sensitivity in the constraint RHS). **T3**: divergence-rate characterization `‖gap_k‖ ~ ρ(M)^k`, chaotic/limit-cycle regimes escape both semantics pointwise but not the T2 set when the game is finite; diagnostics table → certificate rungs. NOVELTY.md from Task 7 searches. PROGRESS.md status map. Commit `docs(eqcf): T1/T2/T3 theory + proposal + progress`.

### Task 6: experiments E1–E5

**Files:**
- Create: `experiments/eqcf/common.py` (Hedge/multiplicative-weights no-regret learner, ε-greedy Q-learner, empirical joint distribution, Lyapunov estimate), `e1_cobweb.py`, `e2_saf_chaos.py`, `e3_rl_pricing.py`, `e4_macro_loop.py`, `e5_jax_scale.py`, `RESULTS.md`

**Specs:**
- **E1 cobweb** (`p = −(s/d)·p̂ + (a−c)/d`, naive expectations = the cyclic SCM's own feedback): regimes s/d ∈ {0.5, 1.5} via `LinearCyclicSCM` (+ `learning_rate` adaptive run), chaotic regime via nonlinear (piecewise/logistic-supply) map unrolled in-script with Lyapunov estimate; intervention: cost shock `do(cost+τ)` implemented as noise-mean shift; report the full certificate ladder + quantitative equilibrium error per regime.
- **E2 SAF chaos**: two-player perturbed RPS (Sato–Akiyama–Farmer), Hedge learners (provably no-regret); intervention = payoff tax on one action (G_do built explicitly); Nash point prediction vs realized time-average vs `cce_bounds(G_do)`; assert realized ∈ interval, report `cce_regret` decay and (chaotic regime) positive Lyapunov exponent of the learning dynamics.
- **E3 RL pricing**: discretized 2-firm quantity competition (5 actions each, linear demand); independent tabular Q-learners (ε-greedy, NOT no-regret); `do` = force one firm's quantity + a cost intervention; question: does the realized time-average land in CCE(G_do)? Report in/out + measured `cce_regret` (either answer is the finding).
- **E4 macro loop**: static NK toy `x = −σ(i − π̂), π = βπ̂ + κx + u, i = φπ, π̂ = π (feedback)` as `LinearCyclicSCM` over (x, π, i, π̂); policy intervention = re-solving with modified Taylor coefficient row (mechanism edit); φ=1.5 (E-stable, margin>0) → IDENTIFIED under `learning_rate`; φ=0.8 → hedged; exhibit the certified flip of the policy conclusion (sign/magnitude of inflation response to a demand intervention under learning vs equilibrium).
- **E5 jax garnish**: `try: import jax` else print SKIP; vmapped Hedge-population time-averages, N=10k learners, wall-clock per certificate.

- [x] Run every script locally (numpy paths), paste real numbers into `RESULTS.md`. Commit `exp(eqcf): E1-E5 certificate-ladder experiments + results`.

### Task 7: novelty due diligence (week-one searches)

- [x] WebSearch (a): "coarse correlated equilibrium" + intervention/comparative statics/partial identification; (b) E-stability + structural causal models / cyclic SCM equilibration. Record hits + verdict in `NOVELTY.md` (honest: if either claim exists, note the pivot per the proposal). Commit with Task 5 docs.

### Task 8: verification

- [x] `uv run pytest` full suite green; `uv run ruff check` + `format --check` clean; pyright known-broken locally (CI is truth) — note in report. Final commit + report (branch left unpushed for user decision).

## Self-Review

- Spec coverage: cce_polytope+LP ✓(T1,2), stability_margin ✓(T3), comparator promotion ✓(T4), T1/T2/T3 ✓(T5), E1–E5 ✓(T6), due diligence ✓(T7). Conformal finite-sample wrappers: deferred — noted in PROGRESS.md as future work (proposal lists them as supporting cast, already shipped elsewhere in lib).
- Naming consistency: `solve_lp/LPResult`, `CCEPolytope/cce_polytope/cce_bounds/cce_regret/certify_cce_do`, `spectral_abscissa/stability_margin/max_stable_learning_rate`, `learning_rate` — used identically across tasks.
- No placeholders; exact formulas inline.
