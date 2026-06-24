> **ARCHIVED — superseded by [`CAUSAL_LLM.md`](CAUSAL_LLM.md).** Kept for history; the canonical program doc is `CAUSAL_LLM.md`.

# Frontier proposal — Certified Causal Discovery: a verifier-guided engine that knows the unknowable

> "AlphaProof for causal inference." Use a formal causal verifier (`causalrl`, extended into a
> step-level do-calculus proof checker) inside a search + self-improvement loop to **discover**
> identification derivations, estimands, and bounds — including in regimes where no closed-form
> complete algorithm exists — in a model that is **calibrated on the theorem-certifiable boundary of
> what is causally knowable**.

This is the deliberate frontier bet distilled from the whole `examples/` exploration and the moonshot
research (see `CAUSAL_LLM_RESEARCH.md`). It is designed to escape the trap every prior prototype fell
into — *re-learning a known polynomial-time algorithm* — by using the verifier as an engine of
**discovery**, not a grader of imitation.

## 1. Thesis and why now

The largest recent leaps in machine reasoning — **AlphaProof**, **AlphaGeometry** — share one recipe:
a **sound formal verifier** + **search/self-play**, where a policy proposes steps and the verifier
certifies them, bootstrapping beyond any fixed algorithm or human. The missing ingredient for causal
inference has been the verifier. We have it: `causalrl` already ships a sound-and-complete
identification algorithm, transportability, and partial-identification bounds. Extended into a
**step-level do-calculus proof checker**, it becomes the "Lean for causality."

Crucially, **per-step soundness gives certified discovery beyond complete algorithms.** The verifier
only needs to check that each derivation step is a valid do-calculus / probability rule; a derivation
assembled by search is then *certified correct even in regimes where no closed-form complete algorithm
is known*. That is the leap: not reproducing the ID algorithm, but **finding certified causal results
where the theory has no procedure** — and abstaining, provably, where nothing is knowable.

## 2. What every prior prototype taught us (the trap to avoid)

- d-separation / identifiability as classification → *neural algorithmic reasoning*; ceiling = the
  known algorithm; the model imitates a label.
- twin-network L3 over random SCMs → real generalisation, but on a problem with a closed form.
- RLVR with the oracle as reward → the robust redo (4 seeds, AURC) **refuted** the easy story; only a
  risk-aversion/refusal effect survived, because the task was a known algorithm at toy scale.

**Lesson:** value is created only where the verifier certifies something a fixed algorithm cannot
hand you for free. The proposal targets exactly that frontier.

## 3. System design

A proposer/verifier/search loop with self-improvement (expert iteration):

```
problem (G, query Q, available distributions Z)
      │
      ▼
┌─────────────┐   proposes a derivation step (a do-calculus / algebra rule application)
│  PROPOSER   │──────────────────────────────────────────────────────────────────────┐
│  (policy)   │   priors over steps + value estimate                                  │
└─────────────┘                                                                        ▼
      ▲                                                              ┌──────────────────────────┐
      │ certified successes become training data                    │  VERIFIER (causalrl++)    │
      │ (expert iteration)                                          │  • each step valid?        │
      │                                                              │  • final estimand do-free  │
┌─────────────┐   best-first / MCTS-lite over derivation steps,     │    and == Q ?              │
│   SEARCH    │   verifier prunes invalid branches                  │  • or: certified hedge +   │
│             │◀────────────────────────────────────────────────────│    tightest valid bounds  │
└─────────────┘                                                      └──────────────────────────┘
```

- **Proposer (policy):** a small-to-mid transformer (from-scratch or fine-tuned open LLM) that, given
  `(G, Q, Z)` and the current derivation state, emits the next step: a do-calculus rule (Rules 1-3),
  a probability-algebra manipulation, a c-component factorisation, a transport/selection operation, or
  a bound-introduction step. Outputs a distribution over steps (the search prior) and a value (likely
  to reach a certified estimand).
- **Verifier (`causalrl` extended):** the core new artifact. Given `G` and a claimed step, it
  **certifies validity** (the d-separation precondition of each do-calculus rule is checkable with
  `causalrl._separation`; algebra steps checked symbolically), checks whether the final expression is
  do-free and provably equals `Q`, and — when non-identifiable — emits a **hedge witness** plus the
  tightest **valid bounds** it can certify. Step-level certification gives a **dense reward** (no
  sparse final-only signal).
- **Search:** at inference, best-first / lightweight MCTS over steps, guided by policy priors + value,
  with the verifier pruning invalid branches. This is where capability beyond the base policy is
  unlocked.
- **Self-improvement:** verifier-certified derivations found by search are added to the policy's
  training set (and a value net); iterate (AlphaZero/AlphaProof-style expert iteration). The policy
  bootstraps purely from *certified* successes — no human labels, no leakage.

## 4. The frontier targets (where complete algorithms end)

The leap lives where do-calculus completeness stops or is only partial in practice:

1. **Path-specific / mediation effects under confounding** — the "recanting witness" frontier; no
   single clean algorithm, often only bounds.
2. **Soft / stochastic (policy) interventions** — σ-calculus (Correa–Bareinboim) is sound but
   under-tooled; a search engine over it is novel.
3. **Selection bias × transportability combined** — generalized identification over composite
   selection diagrams; estimand discovery and bounds in messy compositions are open in practice.
4. **Tightest partial-identification bounds** — discovering *valid and tight* bounds (canonical-SCM
   polynomial program; Balke–Pearl-style) is genuinely open for many structures.
5. **Identification + finite-sample estimation jointly** — discovering an estimand that is also
   *estimable* well from the available samples, not just symbolically valid.

The verifier's per-step soundness means a derivation found in any of these is **certified correct**,
even though no closed-form complete procedure exists — this is the discovery, not emulation.

## 5. Evaluation — the signals that the leap is real

- **Pass@k crossover (the AlphaProof signal).** On hard identification-derivation instances, the
  search+verifier system solves problems the base policy *and a strong zero-shot LLM* cannot solve at
  any k. This directly answers the "RLVR only sharpens, never creates" debate (Yue et al.) surfaced in
  the moonshot research: the verifier+search **unlocks** capability absent from the base.
- **Certified discovery beyond the algorithm.** On a held-out suite from §4 (PSE, soft interventions,
  composite selection diagrams), produce **verifier-certified estimands or tight bounds** where the
  complete algorithm is unavailable/partial. One certified result in open territory is the headline.
- **Calibration on knowability (the honesty spine).** Report selective risk / AUROC on the
  identifiable-vs-non-identifiable boundary (which `causalrl` certifies), and **bound tightness** vs
  known bounds. The model's competence is the *boundary of the knowable* — identifiable → estimand;
  partially → tight bounds; unknowable → certified abstention.
- **Baselines:** zero-shot LLM; SFT-on-derivations (no search); the symbolic complete ID algorithm
  (an oracle *ceiling* where it exists); pure search without the policy; policy without verifier
  pruning. **Ablations:** policy-only vs +search vs +verifier-pruning; expert-iteration rounds.

## 6. Phased plan (de-risked; flag-planting first)

- **Phase 0 — the verifier (the foundation, valuable on its own).** Extend `causalrl` into a
  **step-level do-calculus proof checker** with a clean API: certify a single rule application,
  check final-estimand equality, certify hedges + bounds. This is "Lean for causality" and a
  contribution in itself.
- **Phase 1 — flag-planting.** Derivation *discovery* on standard ADMGs: supervised warm-start +
  search; demonstrate **pass@k crossover** (search+verifier solves what the base does not) and
  human-readable **certified derivations**. The direct analog of AlphaProof's first results.
- **Phase 2 — the leap.** Extend to one open frontier from §4 (recommend **path-specific effects**
  or **tight bounds**); demonstrate **certified discovery beyond the complete algorithm**.
- **Phase 3 — the epistemic causal model.** Unify L1/L2/L3 and identifiable / partially-identifiable /
  unknowable with calibrated, theorem-grounded abstention. The differentiated artifact: a model whose
  core skill is *knowing the limits of causal knowability*.

## 7. Honest risks and mitigations

- **Verifier maturity.** Lean is industrial; the `causalrl` checker is young. Building the step-level
  checker (Phase 0) is real research — but it is an *asset you control*, and the per-step rules
  (do-calculus, d-separation) are individually simple and already partly in the lib.
- **Problem-space size.** Causality is smaller / more structured than mathematics — risk of "too easy"
  or "too niche". Mitigation: target the open corners (PSE, soft, bounds, finite-sample), not the
  already-solved interventional case.
- **Reward sparsity / search cost.** Mitigated by *dense step-level certification* (every valid step
  is rewarded), curriculum from short to long derivations, and expert iteration. Search is the main
  compute cost — far below LLM pretraining, but non-trivial.
- **"Is it discovery or retrieval?"** Guard with held-out structures and leakage-controlled generation
  (problems synthesised by `causalrl`, never web-sourced) — the same discipline this repo already used.

## 8. Why it is a frontier leap, and the Anthropic angle

- **First certified causal *discovery* via verifier + search** — not re-deriving a known algorithm, but
  finding certified estimands/bounds where the theory has no procedure. The proven recipe (formal
  verifier + search) that cracked math and geometry, applied to causality for the first time.
- **Theorem-grounded honesty.** The Causal Hierarchy Theorem makes "this is unknowable from this data"
  a *certifiable* ground truth — a rare case of a mathematically-grounded "I cannot know this." A model
  calibrated on that boundary is the causal-inference instantiation of honest, scalable-oversight-
  friendly reasoning — the differentiated, Anthropic-shaped contribution.

## 9. Resourcing (rough)

1–3 researchers; modest compute (search-heavy: CPU + some GPU; no frontier-scale pretraining);
~6–12 months to reach Phase 2. Phase 0 (the verifier) is the gating dependency and the first thing to
build.

---

**One line:** build the "Lean for causality" inside `causalrl`, drive it with a search+self-improvement
loop to *discover certified identification where the complete algorithms end*, in a model that knows —
provably — what it cannot know.
