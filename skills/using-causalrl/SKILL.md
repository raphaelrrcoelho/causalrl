---
name: using-causalrl
description: Use when solving a causal-inference or causal-RL problem with the causalrl Python library, or deciding whether causalrl already covers a task — effect identification (do-calculus / backdoor / ID / g-ID), transportability across domains or populations, which variables to intervene on (POMIS / minimal intervention sets), causal discovery (PC / FCI with latent confounders), off-policy value bounds under unmeasured confounding (marginal sensitivity model / Manski), ship-or-abstain decision certificates, certifying a trained policy against hidden confounding, learning an SCM from data and planning inside it, online causal model-based RL, counterfactuals / abduction, confounded imitation, curricula, reward shaping, causal games, anytime-valid (peeking-safe) policy comparison, continuous-dose/level decisions under a wall-clock budget, or locating which mechanisms shifted between regimes.
---

# Using causalrl

## Overview

`causalrl` makes the Bareinboim 9-task causal-RL taxonomy runnable. Its distinctive value is **honest partial answers**: it will tell you when an effect is *not* identifiable, bound a decision it cannot point-identify, and hand back a **certificate** of how robust a decision is to hidden confounding — rather than returning a confident wrong number. Reach for a library primitive before hand-rolling an estimator; that is the whole point.

## When to use

Use this skill to pick the right entry point for a task, and to avoid three recurring mistakes: hand-rolling an estimator the library already has, misreading a partial-ID/one-sided result as a point answer, and swallowing a "not identifiable" signal. If unsure whether causalrl covers a problem, start here — it usually does.

## Task → entry point

| You want to… | Use (top-level `causalrl.` exports) |
|---|---|
| Decide **ship vs. abstain** from confounded/off-policy logs, with a robustness certificate | `certify_decision` → `DecisionCertificate` (composes `pivotality_certificate`, `tipping_gamma`, `msm_contribution_bounds`) |
| Certify a **trained policy**'s value improvement over the logging policy (any trainer) | `certify_policy(dataset, target_actions, gamma_max=…, alpha=…)`; the `alpha` gate is `conformal_action_value` (distribution-free lower bound on a fresh return). `dataset` is any `LoggedDecisions` — the tabular `ConfoundedTrajectoryDataset`, or `FeatureDecisionLog` when states are vectors and actions are `Intervention`s, so an `InterventionalAgent` certifies directly with no arm codebook |
| **Learn the SCM from data** rather than assuming one | `fit_scm`, `fit_scm_mec` (one SCM per DAG in the CPDAG's equivalence class), `counterfactual_interval`; a fitted SCM is a Gymnasium env via `CausalEnvWrapper` |
| **Act** from confounded logs (plan in a given or learned model) | `CausalMBRLAgent` (front door), `GFormulaBackdoorAgent` (+ `.cate()` per-unit ITE), `BackdoorAdjustedAgent`, `FunctionApproxBackdoorAgent`; all take an observation: `act(obs)` returns `argmax_a mu_a(x)` |
| **Learn the structure while acting** (interventional, online) | `OnlineCausalMBRL` — Thompson sampling over an I-MEC belief, `refit()` from its own `do()` experiments, `probe()` for the next most informative intervention |
| Ask if an effect **P(y \| do(x))** is identifiable, and get the formula/estimate | `is_identifiable`, `identify_effect`, `estimate_effect`; g-ID: `is_gid_identifiable`; backdoor set: `backdoor_adjustment_set`, `is_backdoor_admissible` |
| **Transport** an effect to a different domain/population (selection diagram) | `is_transportable`, `identify_transport`, `transport_formula`, `estimate_transported_effect`; multi-domain / sID-mz: `*_general`; `SelectionDiagram`, `Domain`; regret + abstention: `transport_regret_certificate` |
| Choose **which variables to intervene on** (causal bandits, experiment design) | `pomis`, `minimal_intervention_sets`, `requires_experiment`; agents: `POMISThompsonSampling`, `CausalThompsonSampling` |
| **Compare policies while watching** (peek, stop when decisive, keep the guarantee) | `sequential_policy_comparison(dataset, target_actions, alpha=…, reward_range=…)` → `SequentialVerdict.stop` / `.better`; the primitive is `confidence_sequence`. Use this INSTEAD of a fixed-sample CI whenever you look more than once |
| Decide a **continuous** action (a dose, a budget, a level) under a time budget | `InterventionSpace.create({"x": Continuous(lo, hi)})` + `AnytimeInterventionSearch`; `Deadline.after(seconds)`; `SearchReport.certificate()` hedges if the clock truncated the sweep |
| Fit an SCM whose graph **has latent confounding** (`fit_scm` refuses it) | `fit_scm_bounded(data, graph=…, value_ranges=…)` → `BoundedSCMFit.interval(node, assignment)`; only nodes confounded with their OWN parents are bounded, the rest are point-fitted |
| Check a fitted model **before trusting a `do()` answer** | `certify_fitted_query(model, data, intervention=…, outcome=…)` — hedges when the mechanism mispredicts that regime, or when no logged row lands in it at all |
| Find **which mechanisms differ** between two regimes (rather than asserting it) | `localize_mechanism_shift({name: data, …}, graph=…)` → `ShiftReport.selection`, which is what `identify_transport` takes |
| Assert a node has **no unobserved parents** (randomisation, feature flag, rule-based policy) | `graph.assert_complete_parents("A", reason=…)` — converts BOUNDED to IDENTIFIED and records the assertion in the certificate |
| **Discover structure** from data | `discover` (PC), `discover_interventional`, `discover_latent` (FCI, latent confounders → `PAG`); CI test: `conditional_mutual_information`; output: `CPDAG` / `PAG` |
| **Bound an off-policy value / OPE** under unmeasured confounding | `msm_policy_value_bounds`, `ipw_sensitivity_bounds`, `manski_bounds`, `causal_q_bounds`, `msm_per_step_bounds`, `msm_stratified_bounds`, `tipping_gamma`; point OPE: `ipw_value`, `stream_policy_value`; sequential/DTR: `estimate_sequential_value`, `certify_sequential_value`. All of these also live together under `causalrl.ope` |
| Compute **counterfactuals / abduction** | `StructuralCausalModel` (`.do`, `.abduct`, `.counterfactual`), `counterfactual_expectation`, `effect_of_treatment_on_treated`, `ExogenousPosterior`, `build_unrolled_scm` |
| **Imitate** under confounding | `CausalImitator`, `is_imitable`, `imitation_backdoor_set`, `BehavioralCloning` |
| Build a **curriculum** / prerequisites | `causal_curriculum`, `is_valid_curriculum`, `PrerequisiteLearner`, `curriculum_q_learning` |
| **Reward shaping** (potential-based, causal) | `apply_potential_shaping`, `causal_potential` |
| **Causal games** / equilibria | `pure_nash_equilibria`, `mixed_nash_equilibria`, `best_response`, `is_nash_equilibrium`, `CausalGame` |
| **Gymnasium** envs / RL harness | `CausalEnvWrapper`, `register_envs`, env suite (`MABUCEnv`, …), `run_episodes`, `factored_advantage` |
| Build an **SCM / graph** | `StructuralCausalModel`, `CausalGraph`, `LinearGaussianMechanism`, `NeuralMechanism`, `FunctionalMechanism` |

Everything is a lazy top-level export: `from causalrl import certify_decision` (see `causalrl.__all__`). Torch-backed pieces need the `causalrl[torch]` extra.

## Honest-scope rules (read a result the way the library means it)

1. **A certificate's `decision` is the question, not the answer.** `certify_decision(...).decision` / `.naive_contrast` report the *naive, possibly confounded* contrast under test (e.g. `"prefer action 1"`). The **verdict** is `.certified` plus `.tipping_gamma` / `.msm_certified`, and `.recommendation` resolves them into `"act"` / `"abstain"` — read that. Never read `.decision` as the recommendation: a confounded win still prints `"prefer action 1"` while `.certified` is `False`.
2. **Partial-ID and sign-robustness results are one-sided.** "Not certified" / a bound that includes the reference means *cannot rule out a flip* — **not** proof of a flip. Say "not robust", not "the effect is negative".
3. **Calibrate the sensitivity Γ against a reference.** A `tipping_gamma` is only meaningful next to a yardstick — e.g. compute a *measured* confounder's assignment odds ratio and compare. Tipping at Γ≈1.2 while a known covariate sits at Γ≈9 means fragile.
4. **Let identification failures surface.** `NotIdentifiableError`, `UnverifiedAssumptionError`, `RealizabilityError`, `CausalInterfaceUnavailableError` are information — report them, don't swallow or `except: pass`.
5. **A confidence sequence covers sampling error, not confounding.** `sequential_policy_comparison` is valid under unlimited peeking, but a confounded estimate converges to the wrong number and a time-uniform band just tracks the wrong number more reliably. The two guarantees do not multiply — run `certify_policy` for the confounding layer.
6. **A search cut off by its `Deadline` is not an exhaustive sweep.** Check `SearchReport.exhausted` / the `budget-truncated` hedge before reporting a searched optimum.
7. **The MSM sensitivity is on the propensity weights** (sharp when target supports are disjoint, valid-but-conservative otherwise). It does not certify an arbitrary outcome-model/doubly-robust point estimate.

## Common mistakes

- Running a naive t-test / mean-difference on confounded logs instead of `certify_decision`.
- Reading `cert.decision` and concluding the tool endorses that arm — read `.recommendation` (see rule 1).
- Treating a wide/zero-crossing partial-ID bound as "no effect" (rule 2).
- Reimplementing backdoor adjustment, IPW, transport formulas, or POMIS by hand — they are exported.
- Passing an unconfounded/uniform-propensity assumption where logging was non-uniform.

## Changed in 3.0 (from 2.x)

- `certify_decision(outcomes=, treated=)` is now `certify_decision(rewards=, actions=)`, and
  `DecisionCertificate.decision` reads `"prefer action 1"` / `"prefer action 0"` instead of
  `"prefer treated"` / `"prefer control"` — the front door takes bandit logs, not trial arms.
- `estimate_sequential_value` / `certify_sequential_value` take `(actions=, reward=)`.
- `certify_sequential_transport` → `certify_transported_policy_value`;
  `causalrl.interop.dowhy.from_dowhy_estimate` →
  `policy_contrast_from_dowhy`; `causalrl.interop.econml.from_econml_cate` →
  `policy_from_econml_cate` (neither is a top-level export).
- The OPE surface moved into `causalrl.ope`. Top-level `causalrl.<name>` imports are unchanged.
- Removed, having no RL referent: `certify_mean`, `certify_quantile`, `moment_diagnostic`,
  `tail_index_hill`, `weighted_quantile`, `bootstrap_quantile_ci`, `stream_quantile_certificate`,
  `causalrl.backends.quantile_sketch`, `causalrl.experimental.ope`, `causalrl.meanfield`,
  `causalrl.interference`. For plain sample statistics, reach for numpy/scipy directly.
- `act(observation)` on the back-door planners is now genuinely contextual. In 2.x it returned a
  constant and discarded the observation, so code that ignored its argument silently still worked.

## Changed after 3.0

- `certify_policy` / `conformal_action_value` are generic in the action type (`LoggedDecisions`),
  so feature-space states and `Intervention`-valued actions work without an arm codebook. The
  conformal positivity hedge's detail key is now `"unsupported"` (list of strings) rather than
  `"unsupported_state_action_pairs"`.
- `InterventionSpace` domains may be `Continuous(lo, hi)` as well as `Discrete(values)`. A raw
  value tuple still means `Discrete`. `values(v)` raises `TypeError` on a continuous domain (there
  is no arm list) — use `domain(v)` / `permits` / `project` / `sample`, or search it.
- The PC/FCI skeleton is **PC-stable**: `discover` no longer depends on the order the variables are
  passed in. This can change the returned graph on data where that order-dependence was active.
  `conditional_mutual_information` is 5–10× faster and bit-identical.
- `Agent.observe_step(obs, action, reward, next_obs, done)` is the representation-neutral driver
  hook; `observe_transition` (int states) still works and is what the default forwards to.

## Deeper skills

For the full decision-certification workflow (evidence selection, certificate interpretation, Γ-calibration), see the `causal-decision-certification` skill when present.
