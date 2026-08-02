# Learned SCM — Progress / Resume

**Program:** make causalrl agents *learn* the SCM, not only certify decisions.
Design: `DESIGN.md`. Plan: `plans/2026-08-01-fit-scm.md`. Sequence **1 → 4 → 3 → 2**.

**Branch:** `learn-the-scm`.

The local toolchain works (torch 2.12.0, numpy 2.4.6, `pyright src` clean — checked 2026-08-01), so
tasks verify locally first; CI is still the final gate before anything is called green.

## Sub-project 1 — `fit_scm` (this plan)

| Piece | Status |
|---|---|
| `orient` (CPDAG → DAG) | GO — Task 1, review clean after 4 fix rounds |
| provenance + L3 guard | GO — Task 2, review clean |
| `TabularCPT` | GO — Task 3, review clean |
| `LinearGaussianFit`, `ANMFit` | GO — Task 4, review clean (1 finding refuted with evidence) |
| `NeuralFit` | GO — Task 5, review clean |
| `fit_scm` | GO — Task 6, review clean after 2 fix rounds |
| `fit_scm_mec` | GO — Task 7, review clean |
| `counterfactual_interval` | GO — Task 8, review clean after 1 fix round |
| oracle kill gate | GO — `causal=0.0037 correlational=0.2394 gap=+0.2357 passed=True` |
| NHEFS cross-check | GO — learned SCM +3.909 kg vs g-formula +3.414 kg, gap 0.495 kg (just under the 0.5 kg flag line; see below) |

Every task 1–10 is complete with its review clean (`.superpowers/sdd/2026-08-01-fit-scm/progress.md`
is the ledger). Task 9 (lazy top-level exports of `fit_scm`, `orient`, `fit_scm_mec`,
`counterfactual_interval`, `LinearGaussianFit`, etc.) has no row of its own — it is the plumbing
that makes every row above reachable as `from causalrl import ...`, not a piece with its own
measured number.

## Oracle gate result

```
causal=0.0037  correlational=0.2394  gap=+0.2357  passed=True
```

10 seeds, n=20,000, `examples/learned_scm_oracle_gate.py` (`run_learned_scm_oracle_gate`). The
baseline is load-bearing, not a strawman: an independent saturation check confirmed both the
true-graph fit and the L1-equivalent reversed-order fit reproduce the *observational* distribution
to within the same noise floor (TV(fit, data) = 0.0040 vs 0.0059), while only the causally correct
graph recovers the interventional query (`E[W|do(Z=1)]` = 0.8997 vs oracle 0.9, against the
reversed model's 0.6611 vs the true 0.9). Structure, not fit quality, buys the correct intervention.
Full suite at this point: 877 passed, 3 skipped, 96.68% coverage.

## NHEFS cross-check

```
learned SCM  E[Y|do(A=1)] - E[Y|do(A=0)] = +3.909 kg
targeted g-formula (existing suite)      = +3.414 kg
agreement gap                            = 0.495 kg

literature reference: +3.4 to +3.5 kg
```

(`examples/learned_scm_nhefs.py`, n=1,566 after dropping missing rows, 40,000-draw Monte Carlo
`do()` query — deterministic given the fixed seeds, reproduced bit-for-bit on a second run.)

**Read honestly, not rounded up.** The gap is 0.495 kg — under the brief's 0.5 kg discrepancy
threshold, but by a margin (0.005 kg) too thin to call a clean pass without comment. The g-formula estimate lands
inside the +3.4–3.5 kg literature band; the learned SCM's own number, +3.909 kg, does not — it is
the higher outlier of the three. This is a genuine estimator-choice effect, not a bug or a tuned
result:

- The existing g-formula (`GFormulaBackdoorAgent`, routed by `CausalMBRLAgent(covariates=...)`) is a
  **T-learner**: two separate ridge-regularized (λ=1, standardized covariates) models, one fit per
  treatment arm, standardized over the full sample. Ridge shrinkage and a per-arm slope both pull
  its answer down from the unregularized value.
- The learned SCM's `Y` mechanism (`LinearGaussianFit`) is an **S-learner**: one pooled, unregularized
  closed-form OLS regression of `Y` on `[A, all 9 confounders]`, sharing one confounder slope across
  both arms. `do(A=1)` vs `do(A=0)` reads off exactly that OLS coefficient on `A`.

Neither reproduces the textbook Hernán–Robins model (which adds quadratic terms for age, weight,
smoking intensity and years, and an interaction term) — both are simplified linear-in-the-same-9-
covariates fits, so some spread between them, and between each of them and +3.4–3.5, is expected. No
model parameter was changed to narrow this gap; the brief's code ran as specified. Flagging this as
an **open discrepancy worth tracking**, not papering over it: if sub-project 2's tighter fitting work
touches `LinearGaussianFit`, re-run this cross-check and see whether the gap moves.

What only the fitted SCM answers from the same object — not a second estimator, the same one:
other interventions (any dose, any subset of confounders held fixed), full-distribution rollouts via
`see()`, and (on a model with a non-invertible node) a `counterfactual_interval` — all of which
`GFormulaBackdoorAgent` structurally cannot produce, because it was built to answer exactly one
contrast.

## Scope refinements made during implementation

- `counterfactual_interval` bounds ambiguity **at the target node** exactly, and refuses when a
  non-invertible node lies strictly upstream between an intervention and the target — a loose
  composed bound would be worse than an honest refusal. `tight` is therefore always `True` in
  phase 1; the field exists so sub-project 2 can return valid-but-loose bounds without a breaking
  type change.

## Next

Sub-project 4: plan inside the learned model (`do()` rollouts, counterfactual data augmentation
via `abduct`, policy improvement in the model).
