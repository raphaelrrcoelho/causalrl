# Adversarial self-review (2026-07-15)

Red-team pass over every result on this branch, done before any external submission. Format:
attack → what the check found → disposition. Fixes were applied in the same commit.

## A. Numerical claims

**A1. "The welfare CCE of the E6 game is degenerate at {32}" — too convenient to trust.**
Checked by machine-verifiable weak-duality certificates independent of the shipped simplex
(scratch script `verify_e6_lp.py`): for all four LPs (welfare min/max, profit min/max) the dual
multipliers are nonnegative and dual-feasible with slack ~1e-14. **Values CONFIRMED**: welfare
CCE = {32} exactly, firm-profit CCE = [12, 20] exactly. The check also *explained* them: the
1-unit grid creates three pure Nash on the Q=8 anti-diagonal — (3,5), (4,4), (5,3) — welfare is
constant on that line. **New disclosure required (applied):** the degeneracy is partly a
discretization artifact (continuous Cournot has a unique Nash); the collusion escape (35.9 > 32)
is *not* an artifact — certified impossible for any CCE. Referee-proofing follow-up: an off-grid
discretization with unique Nash.

**A2. "eps_T is a collusion meter" — overclaim.** Exploration also produces eps_T (stateless Q:
0.48 at Nash profits). A large eps_T alone proves nothing. **FIXED**: the collusion signature is
eps_T bounded away from zero *combined with* concentrated supra-competitive play; wording
corrected in E6, THEORY scope note, RESULTS.

**A3. E6 seed count.** 3 seeds (one of them, seed 1, converging to an asymmetric non-(3,3)
outcome, still supra-competitive). Fine for a probe; thin for a paper. Calvano-scale session
counts needed before submission. **OPEN (follow-up).**

## B. Experimental inference

**B1. E3b's "collusion driver is value bootstrapping, not memory" — partially tautological and
confounded.** The PG learners maximize *immediate* stage reward (no discounting): they cannot
represent punishment threats **by construction**, so "memory-1 PG doesn't collude" was partly
guaranteed by design. Additionally the E6/E3b horizons differ 100× (2M vs 20k). And real PPO uses
γ>0, so "PPO-family respects the bounds" was unsupported. **FIXED**: reframed as a
state-vs-farsightedness decomposition — memory alone does not break the stage-game bounds; the
folk-theorem ingredient is the intertemporal objective. The farsighted-PG cell at matched horizon
is named as the single most valuable follow-up. What survives is still real: 512 independent
myopic neural populations behave no-regret and land on the point-identified values — the
clean control cell of the 2×2.

**B2. E2's "chaos" label.** No Lyapunov diagnostic of the coupled learning dynamics was computed
at these parameters; SAF prove chaos for replicator families, not for this exact configuration
with discrete Hedge. **FIXED**: RESULTS now reads "non-convergent learning"; Lyapunov of the
learning flow is the cheap follow-up. The demonstrated facts (Nash miss 0.037; containment;
measured eps 0.0021) are unaffected.

**B3. E2's "the certificate catches what Nash misses" — underwhelming as stated.** The Nash miss
(0.037) is 6× smaller than the interval width (0.24); catching it with a wide interval is cheap.
The honest reading (already in RESULTS): the width IS the finding — equilibrium has little
predictive content in this game — now quantifiable as *structural* via the ε-sensitivity
instrument. **DISPOSITION: reframe, no change to numbers.**

**B4. E7's tracked-root choice.** The reported gap (0.94) tracks x+\*; tracking x−\* gives 1.04 —
the conclusion is root-invariant (min gap ≈ 0.94), so no cherry-picking issue, but the basin
masses do depend on the (arbitrary) initial distribution N(0, 1.5²) and the 2.8% crossing figure
is specific to it. **DISPOSITION: fine for a probe; a paper should report the mass-shift as a
function of the initial spread (one curve).**

**B5. E4's "sign flip" — a strawman risk.** An economist may object: nobody trusts comparative
statics at an E-unstable REE, so the "flip" attacks a position no one holds. Defense (kept): the
Dogra/practice point is that stability is routinely *not checked*; the contribution is the
machine check, not the phenomenon. Also **FIXED**: the threshold was mislabeled "Bullard–Mitra" —
it is a Taylor-principle-*type* threshold of this bespoke static toy (φ\* = 1 − (1−β)/(κσ),
recovering φ>1 only as β→1). E4 needs a standard-timing NK learning model before facing
referees.

## C. Theory

**C1. Main Theorem exclusivity was FALSE as stated.** A bistable system (E7 itself!) satisfied
both case 1 (locally, margin>0 at each stable root) and case 3 (multiplicity). **FIXED**: the
theorem now classifies the *global* query; case 1(i) requires global uniqueness of the stable
σ-solution; the multiplicity branch of case 3 explicitly notes each root still carries a correct
basin-local certificate. E7 is the measured local-right/global-mixture gap.

**C2. T2 is definitionally true.** Acknowledged in THEORY; the paper's contribution is the
synthesis + instruments + demarcations, and the positioning already credits the adjacent
literatures per the owner's due diligence. Residual risk: an EC referee calls the theorem an
observation. Mitigation: lead with the Main Theorem (trichotomy) and the demarcation experiments,
not T2 alone. Terminology risk: "anytime-valid" has a martingale meaning in sequential testing;
since containment here is deterministic per-T, prefer "horizon-uniform, assumption-free".
**DISPOSITION: wording guidance for the paper draft.**

**C3. T1′ proof burden.** The sketch leans on Hartman–Grobman + Kushner–Yin + Pemantle with
"remaining in a compact subset of the basin" conditioning doing real work. Standard but must be
written in full with noise conditions for submission. **OPEN (paper work).**

## D. Venue verdict (is this main-venue publishable?)

**Not yet — and now we know exactly what stands between here and there.** Current state after
this review: instruments duality-verified, theorems correctly scoped, experiments honest but
probe-grade (1–3 seeds, one game each, one confound explicitly open).

- **CLeaR (best near-term fit).** T1/T1′ + Main Theorem + E7 (the CCM caveat measured) + the
  certificate semantics is a coherent causal-inference paper. Needs: full T1′ proof, the
  multiplicity-aware hedge implemented, E7 with the initial-spread curve. Verdict: *plausibly
  submittable after ~2–4 weeks of writing + the small additions.*
- **EC / algorithmic-collusion venues.** The strongest hook is E6+E3b: certified stage-game
  bounds as a collusion diagnostic with the measured-regret discharge. Needs: the matched-horizon
  farsighted-PG cell (the open 2×2 cell), Calvano-scale seeds, an off-grid game, and engagement
  with the collusion literature. The theory alone (T2) will not carry EC; the theory + the
  demarcation experiments might. Verdict: *a real shot, contingent on the follow-up experiment
  battery it genuinely needs — this is the one place where breadth is not optional.*
- **NeurIPS.** As a "certificates for multi-agent interventional predictions" paper with the
  library: possible, but the algorithmic novelty is thin; would compete as a
  datasets/benchmarks-adjacent or analysis paper. Verdict: *weaker fit than CLeaR/EC.*
- **JEDC (economics paper).** E1/E4 are toys with bespoke timing. Needs a standard NK-learning
  model (correct Bullard–Mitra timing) and a calibrated exercise. Verdict: *months out, as the
  original sequencing said.*

**Bottom line:** the leap round produced the right *skeleton* (trichotomy theorem + three
boundary-mapping probes + a quantitative instrument, all now verified or corrected). What remains
before any main venue is exactly three things: the farsighted-PG/matched-horizon cell, seed/grid
robustness for E6, and the full T1′ write-up — plus the owner's residual reading checks before
the novelty deltas go to print.
