# When Can You Trust an Equilibrium Counterfactual? — theory notes (T1/T2/T3)

Status: working notes backing the flagship proposal (see `PROPOSAL.md`). Novelty due diligence was
run externally by the project owner (2026-07); verdicts: T1 survives, T2 repositioned as the
bridge-and-operationalization result, T3 unaffected. Library instantiation shipped on branch
`equilibrium-counterfactuals` (see `PROGRESS.md`).

## 0. Setup: two `do()` operators for one query

Fix a system with endogenous variables `x` and exogenous `u`.

- **Equilibrium `do()`** (cyclic-SCM / solution-function semantics, Bongers–Forré–Peters–Mooij):
  mutilate the mechanism, re-solve the fixed point. For the linear cyclic SCM `x = Bx + u`
  (`causalrl.experimental.cyclic.LinearCyclicSCM`), `do(X_i = ξ)` zeroes row `i` and pins `u_i = ξ`;
  the equilibrium counterfactual is `x*_do = (I − B_do)⁻¹ E[u_do]`, defined iff `I − B_do` is
  invertible.
- **Dynamics `do()`** (agent-based / learning semantics): impose the same mutilation on the
  *process* and unroll: `x_{k+1} = F_do(x_k, u_k)`. For a learning population, the process is the
  adaptive dynamics; the object of interest is its limit (T1) or its time-averaged play (T2).

The question throughout: **when do the two operators return the same answer**, and what can be said
when they do not. The trichotomy: point identification under expectational stability (T1), partial
identification by the CCE polytope under (measured or assumed) no-regret (T2), certified
non-identification with diagnostics otherwise (T3).

## 0.5 Main Theorem (the trichotomy, unified)

**Theorem (identification trichotomy for learning populations).** Let `Q = E[f]` be a *global*
interventional query under `do` (the population's long-run behaviour, not conditioned on a basin),
posed of a population of adaptive learners, where the system is in one of the two shipped exact
classes (a cyclic SCM with smooth mean field, or a finite game). Then exactly one of the following
holds, each condition computable by a shipped instrument, and the corresponding certificate is
warranted:

1. **POINT (`IDENTIFIED`).** Either (i) the intervened system has a **unique** stable σ-solution,
   it is hyperbolic, and the mean dynamics are stable there (`stability_margin > 0`; T1/T1′): the
   learning-limit `do()` *equals* the equilibrium `do()` for every sufficiently small gain; or
   (ii) the system is a finite game and `f` is constant over `CCE(G_do)` (LP width ≤ tol): the
   equilibrium point prediction holds for *every* no-regret population — T2 degeneracy.
2. **SET (`BOUNDED`).** The system is a finite game and the population's measured realized regret
   at the horizon actually run is `ε_T`: the realized time-average of `f` lies in the LP interval
   over `CCE_{ε_T}(G_do)`, with **no further assumptions** — T2. The reported dual multipliers
   are the interval's growth rate per unit of additional regret (the certificate's
   ε-sensitivity), so the certificate also prices what more learning time is worth.
3. **NONE (`EMPIRICAL`, hedged with diagnostics).** One of: every σ-solution has unstable mean
   dynamics (`margin < 0`; no admissible gain rescues — T1 converse); the ε-interval is vacuous
   at the measured regret (abstention); or the intervened system has **multiple** stable
   σ-solutions (or a marginal one), so the global learning outcome is a *basin-mass mixture* over
   them — an object plain SCM semantics cannot represent (Blom–Bongers–Mooij), reported as a
   selection hedge — T3. In the multiplicity case, each stable σ-solution still carries a correct
   *basin-local* case-1 certificate (T1′ is local by nature); the trichotomy classifies the
   *global* query, and E7 measures exactly this local-right/global-mixture gap.

The case analysis is exhaustive and decidable within the stated classes: margin sign at each
σ-solution, σ-solution multiplicity, LP width, LP vacuity, and solvability are each finite
computations (`stability_margin`, `compare_equilibrium_unrolling`, `cce_bounds`,
`certify_cce_do`, `solve`); exclusivity holds because case 1(i) requires global uniqueness of the
stable σ-solution, which the multiplicity branch of case 3 negates.

**Scope demarcation (measured by E6/E3b).** Case 2's polytope is the **stage-game** CCE.
Populations with history-dependent strategies *and intertemporal objectives* (discounted
bootstrapped values — the folk-theorem ingredients) can sustain time-averages outside it — but
they are detected, not missed: their measured `ε_T` stays bounded away from zero, and it *equals
the forgone stage-game deviation gain that intertemporal threats are enforcing*. E3b's control:
history-dependent but **myopic** learners (policy gradient on stage reward, no discounting)
cannot represent such threats and empirically stay no-regret — memory alone does not break case 2;
the intertemporal objective does. A large `ε_T` alone is *not* proof of collusion (exploration
also produces it — E6's stateless-Q control reads 0.48); the collusion signature is `ε_T` bounded
away from zero **with concentrated supra-competitive play**. The repeated-game analogue of case 2
is folk-theorem-sized (vacuous), so abstention/inflation is the correct certificate — and the
theorem says exactly when it fires.

## 1. T1 — E-stability is causal validity (linear class, both directions)

**Setting.** Linear cyclic SCM `x = Bx + u`, intervened system `(B_do, u_do)` with `I − B_do`
invertible (unique equilibrium `x* = (I − B_do)⁻¹ E[u_do]`). Adaptive learning with gain sequence
`γ_k`:

```
x_{k+1} = x_k + γ_k ( B_do x_k + u_{do,k} − x_k ),        E[u_{do,k}] = E[u_do]
```

- constant gain `γ_k ≡ γ ∈ (0, 1]` is the comparator's `learning_rate` mode (Euler discretization
  of the mean dynamics);
- decreasing gain (`Σγ_k = ∞, Σγ_k² < ∞`) is Robbins–Monro stochastic approximation, the
  Evans–Honkapohja/Marcet–Sargent least-squares-learning regime.

The **mean dynamics** are the ODE `ẋ = (B_do − I) x + E[u_do]`, whose Jacobian is `B_do − I`.
Write `ν_i = λ_i(B_do) − 1` for its eigenvalues, `margin = −max_i Re ν_i = 1 − max_i Re λ_i(B_do)`
(`LinearCyclicSCM.stability_margin`), and `γ* = min_i −2 Re ν_i / |ν_i|²`
(`max_stable_learning_rate`).

**Theorem T1 (point identification ⇔ E-stability).** The following are equivalent:

1. **E-stability:** `B_do − I` is Hurwitz, i.e. `margin > 0` (max Re λ(B_do) < 1);
2. **constant-gain convergence:** for every `γ ∈ (0, γ*)`, the damped iteration converges to `x*`
   from every initial condition (in mean; with noise, to an `O(γ)` neighbourhood);
3. **decreasing-gain convergence:** Robbins–Monro learning converges to `x*` almost surely.

Under any of these, the learning-limit `do()` **equals** the equilibrium `do()` — the cyclic SCM's
σ-solution is the causally correct description of the learning population's interventional
behaviour. Conversely, if `margin < 0` the mean ODE is unstable at `x*`: constant-gain learning
diverges for every `γ > 0` (every eigenvalue `1 + γν` with `Re ν > 0` has modulus > 1), and with
nondegenerate noise decreasing-gain learning fails to converge to `x*` almost surely
(non-convergence to linearly unstable points, Pemantle-type). The learning-limit `do()` then does
not exist as a point — the equilibrium prediction has no dynamical counterpart (T3 territory).

*Proof of (1) ⇔ (2).* The iteration matrix is `M_γ = I + γ(B_do − I)` with eigenvalues `1 + γν_i`.
`|1 + γν|² = 1 + 2γ Re ν + γ²|ν|² < 1 ⇔ γ < −2 Re ν / |ν|²`, which admits a positive `γ` iff
`Re ν < 0`; taking the minimum over `i` gives exactly `γ*`, positive iff `margin > 0`. The fixed
point of `M_γ` is `x*` for every `γ` (damping never moves the equilibrium — only the convergence
class changes). *(1) ⇔ (3)* is the ODE method of stochastic approximation (Ljung; Kushner–Yin;
Evans–Honkapohja ch. 6): the algorithm's a.s. limit set is the attractor set of the mean ODE, which
is `{x*}` iff Hurwitz; the converse direction is the standard non-convergence-to-unstable-points
theorem. ∎

**Corollary (naive unrolling is a strictly stronger demand).** The literal period-1 iteration
`x_{k+1} = B_do x_k + u` (the comparator's default) converges iff `ρ(B_do) < 1`. Since
`ρ(B) < 1 ⇒ max Re λ(B) < 1` but not conversely (e.g. `B = [[−2]]`: `ρ = 2`, `margin = 3`), there
is a nonempty regime — non-contractive but E-stable — where the "map" diverges while every
sufficiently damped adaptive population converges to the equilibrium counterfactual. This is the
formal content of the equations-vs-maps gap for local interventions in the linear class, and it is
exactly what `compare_equilibrium_unrolling(..., learning_rate=γ)` certifies (`IDENTIFIED` on the
E-stable class, with the Jacobian spectrum as witness).

**Boundary and selection caveat (`margin = 0`, or `I − B_do` singular).** With a continuum of
equilibria or marginal stability, the learning limit depends on the initial condition (selection by
basin). Blom–Bongers–Mooij (UAI 2019) prove plain SCMs cannot represent equilibria that depend on
initial conditions — the causal-constraint-model caveat and the economics (equilibrium selection by
learning) are the same phenomenon. The library refuses to fabricate a solution in this case
(invariant I3: `solve()` returns a typed hedge), which is the operationally honest behaviour.

**T1′ (hyperbolic local version — the nonlinear statement).** Let the mean field be
`F(x) = f(x) − x` with `f` C¹, and let `x*` be a hyperbolic zero of `F` (no eigenvalue of
`J = DF(x*)` on the imaginary axis). Then:

1. `J` Hurwitz ⇒ for every `γ ∈ (0, γ*(J))` (same eigenvalue formula as the linear case, applied
   to `J`) the constant-gain iteration converges locally to `x*` (to an `O(γ)` neighbourhood
   under noise), and decreasing-gain SA converges to `x*` a.s. on the event of remaining in a
   compact subset of its basin — so the learning-limit `do()` equals the equilibrium `do()`
   **locally on the basin of `x*`**.
2. `J` not Hurwitz (some `Re λ > 0`) ⇒ under nondegenerate noise, SA does not converge to `x*`
   a.s. (non-convergence to linearly unstable points) — the equilibrium `do()` at `x*` has no
   learning-limit counterpart.

*Proof sketch.* Linearize at `x*`; hyperbolicity gives a locally conjugate linear system
(Hartman–Grobman), reducing (1)/(2) to the linear T1 plus the standard SA limit-set theorem
(Kushner–Yin; Evans–Honkapohja ch. 6) and the Pemantle-type non-convergence theorem. The gain
threshold `γ*` is the linear formula evaluated at `J`. ∎

**The global caveat is quantitative, not rhetorical (E7).** With multiple stable σ-solutions the
theorem's "locally on the basin" qualifier binds: every local certificate can be simultaneously
correct while the *population-level* interventional outcome is the basin-mass mixture, and
interventions move basin boundaries. E7 measures this: both equilibria of a bistable mean field
carry margins ≈ +0.9 (locally IDENTIFIED), yet `do(u)` both shifts the roots and re-partitions
the ensemble across basins — equilibrium-tracking predicts the shifted root while the population
mean lands far from it. The missing diagnostic is a multiplicity-aware hedge: report all stable
σ-solutions and (when trajectories are available) ensemble basin masses.

**Scope honesty.** T1 as stated is exact for the linear/contractive-adjacent class — the class for
which the shipped cyclic-SCM identification is solid; T1′ extends it locally to hyperbolic
σ-solutions of smooth systems, with the basin scope made explicit above. The bridge claim
("E-stability = the σ-solution is the causally correct object") is the contribution, per the
due-diligence verdict. Framework prior art to cite: White & Chalak's settable systems (JMLR 2009);
Mooij–Janzing–Schölkopf ODE→SCM; Rubenstein et al.; Iwasaki–Simon and Dash on equilibration.

## 2. T2 — CCE partial identification (finite-time, anytime-valid form)

**Setting.** Finite game `G` with agents `N`, action sets `A_i`, utilities `u_i`. An intervention
`do` pins a subset of agents' actions, inducing `G_do`: profiles restricted to the consistent ones,
no-deviation constraints only for free agents (`causalrl.magames.cce_polytope(game, do=...)`). A
population plays `T` rounds of `G_do`, producing the empirical joint distribution `μ_T` over
profiles. For a free agent `i`, the **realized regret** is

```
ε_T(i) = max_{a′ ∈ A_i}  (1/T) Σ_t [ u_i(a′, s_{−i,t}) − u_i(s_t) ]   (clipped at 0)
```

— computable from the trajectory log alone (`causalrl.magames.cce_regret`).

**Theorem T2 (finite-time containment; no assumptions).** For every horizon `T`, every free agent
`i`, and every profile functional `f`:

```
E_{μ_T}[f]  ∈  [ min , max ]  of  E_μ[f]   over   μ ∈ CCE_{ε_T}(G_do)
```

where `CCE_ε(G_do) = { μ ∈ Δ(profiles) : deviation_gains · μ ≤ ε componentwise }`. Both endpoints
are linear programs over a polytope contained in the simplex (`causalrl.magames.cce_bounds` with
`epsilon` = the measured per-agent regrets).

*Proof.* `μ_T ∈ CCE_{ε_T}(G_do)` is the definition of realized regret rearranged: the expected
deviation gain of switching to fixed `a′` under `μ_T` *is* the time-averaged realized gain. The
containment of `E_{μ_T}[f]` in the min/max over the feasible set is then immediate. ∎

The theorem is deliberately assumption-free — it binds at the horizon actually run, for whatever
the learners actually did. This is the certificate's finite-time route (`certify_cce_do(...,
epsilon=measured)`), and it is anytime-valid trivially because it holds pointwise at every `T`.

**Corollary (asymptotic route).** If the population is no-regret (`ε_T → 0`; Hart–Mas-Colell,
Foster–Vohra), every limit point of `{μ_T}` lies in `CCE_0(G_do)`, and by continuity of the LP
value in the constraint right-hand side (Lipschitz, with constant given by the optimal dual
multipliers) the ε-bounds converge to the exact-CCE bounds. This recovers the classical statement
as the `ε → 0` limit of the certified one.

**Corollary (the ε-sensitivity is shipped, not just cited).** `bound(ε)` is a concave/convex
piecewise-linear function of the relaxation for the max/min side respectively, and its
one-sided derivative at the measured `ε_T` is the sum of the optimal LP dual multipliers —
returned by the solver (`LPResult.dual_ub`) and reported in the certificate witness as
`epsilon_sensitivity = {lower, upper, width}`. Reading: `width_sensitivity × regret-rate(T)` is
the marginal certificate tightness bought by running the population longer; a certificate with
large width but tiny sensitivity says *the looseness is structural (the game), not statistical
(the learners)* — a distinction no asymptotic statement makes.

**Corollary (degeneracy = when the equilibrium point prediction is safe).** The equilibrium point
prediction of `f` is valid for *every* no-regret population iff `f` is constant over `CCE(G_do)`
(LP width 0) — a checkable condition, certified `IDENTIFIED`. Example: two-player zero-sum games,
where every CCE achieves the value (each player's no-deviation constraint pins the payoff from both
sides), so value functionals are point-identified for any no-regret population; generic
anti-coordination games are not (width > 0).

**Width as the operational content of "equilibrium".** `width(f, do) = max − min` over
`CCE(G_do)` measures how much predictive content equilibrium analysis has for adaptive populations
under that intervention: 0 = full point identification, full payoff range = none (the certificate
*abstains* rather than reporting a vacuous interval).

**Feasibility caveat.** The LP lives in joint-profile space (`Π_i |A_i|` variables — exponential in
the number of players; hardness results exist for succinct games). Exact bounds are for small
games; sampled/relaxed bounds are the scale path (E5).

**Positioning (post due diligence).** Every atomic piece exists somewhere: set-valued
counterfactuals over BCE (Bergemann–Brooks–Morris 2022); LP identified sets over equilibrium
polytopes with asymptotic-no-regret justifications (Magnolfi–Roncoroni; Syrgkanis–Tamer–Ziani);
no-regret as identifying assumption (Nekipelov–Syrgkanis–Tardos 2015); set-valued `do()` over
equilibrium sets in causal games (Hammond et al. 2023; Mishra–Fox–Wooldridge, UAI 2024 — the
correct citation for what the proposal called "Mishra–Fox 2024": *Characterising Interventions in
Causal Games*, PMLR 244:2560–2572); LP-over-CCE machinery to mean-field scale (Campi et al.
2026); and — found in the 2026-07-17 pre-submission scan, NEAREST of all — **Lomys–Magnolfi (EC
2025), "Estimation of Games under No Regret"**: asymptotic-no-regret ⇔ CCE-set convergence,
ε-BCCE set estimators with LP duality, counterfactual LP bounds. The claimed delta is therefore
narrowed to the conjunction Lomys–Magnolfi do not do: (i) ε = the MEASURED realized regret of the
actual trajectory (they inflate by assumed worst-case algorithm-class rates, and themselves note
realized regrets concentrate better); (ii) containment carried directly to the INTERVENED game's
polytope (they re-solve counterfactuals from identified primitives); (iii) abstention-when-vacuous
semantics; (iv) LP duals read as prices of marginal regret; plus the trichotomy against the
cyclic-SCM equilibrium `do()` as two semantics for one query, and width as the operational
equilibrium-content measure. Paper 1 §4 carries the precise wording; Paper 2's EC positioning
MUST engage Lomys–Magnolfi head-on.

## 3. T3 — Certified failure: characterization + diagnostics

When neither T1 nor T2 licenses a claim, the certificate must still *say something checkable*.

**Linear unstable (`margin < 0`).** The gap between the unrolled and equilibrium means grows
geometrically: `‖x_k − x*‖ ≈ ρ(M)^k ‖x_0 − x*‖` along the unstable eigenspace, where `M` is the
effective iteration matrix (`iteration_spectral_radius` in the comparator witness). No admissible
gain rescues convergence (T1 converse), and the hedge says so explicitly
("no learning rate stabilises these mean dynamics"). This is *certified non-identification*: the
equilibrium `do()` has no learning-limit counterpart, and the certificate's diagnostics (margin,
spectrum) are the proof obligations a reviewer can recheck.

**Non-contractive but stable (`ρ ≥ 1, margin > 0`).** Not a failure — a mislabelled success: the
naive-unrolling hedge carries `max_stable_learning_rate`, the gain below which the adaptive
unrolling certifies `IDENTIFIED`. The diagnostic converts an apparent divergence into an actionable
re-run.

**Nonlinear: cycles and chaos.** In cobweb-with-nonlinear-supply (Hommes; Brock–Hommes) and in
learning dynamics on simple games (Sato–Akiyama–Farmer 2002), trajectories converge to attracting
cycles or chaotic attractors: the pointwise learning limit does not exist, and equilibrium `do()`
is quantitatively wrong as a point prediction. Two honest moves remain: (a) for finite games, the
*time-averaged* play still satisfies Theorem T2 at every horizon with its measured regrets — the
CCE interval is the correct object and typically strictly contains (and is wider than) the Nash
point; (b) runtime diagnostics — the Jacobian spectrum at the σ-solution and a largest-Lyapunov
estimate from trajectory logs — trigger the `EMPIRICAL` hedge before the user trusts the fixed
point. Dash (AISTATS 2005) is the causal-side precursor: equilibration changes causal structure,
so manipulating the equilibrated model misleads.

**Certificate mapping (the ladder, as shipped).**

| condition | instrument | rung |
|---|---|---|
| `ρ(B_do) < 1`, gap ≤ tol | comparator (naive) | `IDENTIFIED` |
| `margin > 0`, `γ < γ*`, gap ≤ tol | comparator (`learning_rate`) | `IDENTIFIED` (T1) |
| `margin > 0`, naive divergence | comparator hedge + `max_stable_learning_rate` | `EMPIRICAL` (actionable) |
| functional constant over `CCE(G_do)` | `certify_cce_do` | `IDENTIFIED` (T2 degeneracy) |
| no-regret assumed or ε measured | `certify_cce_do` | `BOUNDED` (T2) |
| ε-interval vacuous | `certify_cce_do` | `EMPIRICAL` abstention |
| `margin < 0` | comparator hedge (no rescuing γ) | `EMPIRICAL` (T3, certified) |
| `I − B_do` singular / margin ≈ 0 | `solve()` typed hedge (I3) | `EMPIRICAL` (selection/CCM caveat) |

## 4. References (working list)

Bongers–Forré–Peters–Mooij (cyclic SCMs, σ-separation); Mooij–Janzing–Schölkopf (ODE→SCM);
Rubenstein et al.; Blom–Bongers–Mooij (CCMs, UAI 2019); Bongers–Blom–Mooij (SDCMs); Iwasaki–Simon;
Dash (AISTATS 2005); White & Chalak (settable systems, JMLR 2009); Evans & Honkapohja (Learning and
Expectations in Macroeconomics); Marcet–Sargent; Evans–Ramey (learning and the Lucas critique);
Hart–Mas-Colell; Foster–Vohra; Mertikopoulos–Papadimitriou–Piliouras (non-convergence);
Sato–Akiyama–Farmer (PNAS 2002); Hommes; Brock–Hommes; Bergemann–Brooks–Morris (2022);
Magnolfi–Roncoroni; Syrgkanis–Tamer–Ziani; Nekipelov–Syrgkanis–Tardos (2015); Hammond et al.
(2023); Mishra–Fox–Wooldridge (UAI 2024); Lomys–Magnolfi (EC 2025); Kline–Tamer (2024);
Hartline (2026 survey); Perdomo et al. (performative prediction); Dogra (causal interpretation
of equilibrium economics); EECS IV Part I (Geanakoplos; Pangallo).
