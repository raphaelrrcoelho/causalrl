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
| NHEFS cross-check | GO — mechanism-family range [+2.990, +3.909] kg contains the targeted g-formula (+3.414 kg, nearest the literature band); the range itself, not a single number, is the finding (see below) |

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
Y ~ LinearGaussianFit (additive, homogeneous effect) = +3.909 kg
Y ~ ANMFit (RBF basis, A x X interactions)            = +2.990 kg
targeted g-formula (existing suite, per-arm ridge)    = +3.414 kg
spread across mechanism families (|linear - ANM|)     =  0.919 kg

literature reference: +3.4 to +3.5 kg
```

(`examples/learned_scm_nhefs.py`, n=1,566 after dropping missing rows, 40,000-draw `do()` query per
family — deterministic given the fixed seeds, reproduced bit-for-bit across repeated runs.)

**The finding is a range, not a point — and the range is the result, not something to explain away.**
The learned SCM's answer to the identical query swings by 0.919 kg depending only on which family
fits `Y`: an additive-linear OLS (`LinearGaussianFit`, forces a homogeneous treatment effect) gives
+3.909 kg; an RBF-basis fit (`ANMFit`, lets the covariate-outcome relationship vary by arm) gives
+2.990 kg. The targeted g-formula (a ridge-regularized T-learner, also arm-varying) lands at
+3.414 kg — *between* the two learned-SCM families and nearest the +3.4–3.5 kg literature band.

The diagnostic that predicts this ships with the model, not bolted on after the fact: `Y`'s
`holdout_score` (out-of-sample R²) is **+0.017** under `LinearGaussianFit` and **-0.099** under
`ANMFit` — worse than predicting the mean. These 9 covariates carry almost no reliable held-out
signal about `Y` under either family, so a contrast read off any single fit is exactly as fragile as
that number warns. `holdout_score` (made honest in Task 6) is doing real diagnostic work here, not
decoration.

This is a **better outcome than a clean single-number agreement would have been**: it demonstrates
the sub-project's own meta-lesson (`DESIGN.md` §7; the same meta-lesson the wider real-data suite
established) instead of merely asserting it. A general fitted SCM spends its capacity on the whole
joint distribution, so its answer to one estimand is correspondingly less reliable than an estimator
built for exactly that estimand — and the model says so itself, via its own diagnostic, rather than
requiring an external check. The surplus this sub-project claims was never point-estimate accuracy:
it is multi-query capability (other interventions, full rollouts via `see()`, and — on a model with a
non-invertible node — a `counterfactual_interval`, all from the same fitted object, none reachable
from `GFormulaBackdoorAgent`'s single-contrast `.contrast` property) plus the diagnostics that flag
when a single-query answer should or shouldn't be trusted. A user who wants g-formula-like behaviour
from the SCM should pass an interaction-capable family for `Y` rather than relying on the
additive-linear default.

## Scope refinements made during implementation

- `counterfactual_interval` bounds ambiguity **at the target node** exactly, and refuses when a
  non-invertible node lies strictly upstream between an intervention and the target — a loose
  composed bound would be worse than an honest refusal. `tight` is therefore always `True` in
  phase 1; the field exists so sub-project 2 can return valid-but-loose bounds without a breaking
  type change.

## Next

Sub-project 4: plan inside the learned model (`do()` rollouts, counterfactual data augmentation
via `abduct`, policy improvement in the model).
