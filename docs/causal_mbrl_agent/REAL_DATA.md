# Causal MBRL Agent — Real-Data Demonstrations

Can `CausalMBRLAgent` solve real problems? We benchmark it against **strong contenders** — the methods
a competent practitioner actually uses — across four real datasets and three regimes (confounded
decisions, individual counterfactuals, off-policy evaluation), always checked against ground truth.
The honest verdict is more valuable than a strawman win, and it is the headline of this page:

> **On real data, causal point estimates do not reliably beat strong contenders. They tie on
> well-behaved problems and are as fragile as everyone else on hard ones. causalrl's defensible edge
> is not a better number — it is the certificate that tells you when to distrust the number.**

The scripts are out-of-CI (network / `causaldata` / `obp` / scikit-learn); run them yourself.

## The engine

`GFormulaBackdoorAgent` (routed by `CausalMBRLAgent(covariates=[...])`) fits a per-action outcome
model over all covariates and standardizes to `E[Y | do(a)]`; `.cate(data)` returns the per-unit
effect (a T-learner). Numpy-ridge default (dependency-free); optional sklearn factory hook.
`examples/_causal_baselines.py` holds the strong contenders: IPW, doubly-robust AIPW, propensity
stratification, and the S-/X-learner ITE meta-learners.

## Confounded decisions — vs IPW / AIPW / propensity stratification

**NHEFS smoking → weight (`examples/causal_mbrl_nhefs.py`) — parity on well-behaved data.** g-formula
+3.41, IPW +3.28, AIPW +3.33, strata +3.23 kg — all agree, all clear the confounded naive +2.54, all
near the textbook +3.4/+3.5. We sit in the pack of serious methods; no win claimed.

**LaLonde job-training (`examples/causal_mbrl_lalonde.py`) — fragility on hard data.** Truth (RCT) is
+$1,794. Estimates scatter: g-formula +1,053, AIPW +508, IPW +231, naive −635, **propensity strata
−597 (wrong sign)**. A *strong* method gets the decision backwards; no point estimate is trustworthy.

## Individual counterfactuals — vs the S-/T-/X-learner (`examples/causal_mbrl_twins.py`)

Twins carries both potential outcomes, so the per-pair effect is known exactly. Estimating individual
effects is something DRL structurally cannot do — but on real, sparse, binary mortality it is
near-unlearnable for the top causal methods too. PEHE (lower better): constant-ATE 0.320, ours
(T-learner) 0.322, S-learner 0.320, **X-learner 0.318** — every method ties with a constant at the
noise floor. The population ATE (−0.025) is recoverable; the per-unit effect is not. An honest null.

## Off-policy evaluation — vs IPS / SNIPS (`examples/causal_mbrl_obd.py`)

The Open Bandit Dataset logs a real recommender under a uniform-random policy (known propensity 1/80),
so OPE is unconfounded and the top estimators are unbiased — nothing to deconfound.

- **Canonical task** (random-policy value from BTS logs): IPS 0.00236, SNIPS 0.00233 vs on-policy
  ground truth 0.00380 — the top estimators agree and are in the ballpark, sample-limited on the 10k
  slice (accurate on the full dataset).
- **Controlled confounding illustration:** induce outcome-selection and IPS balloons from 0.0240 to
  0.0792 (3×) — the top OPE estimator is structurally fooled by confounding.
- **Certificate:** a correct *true-negative* on the clean known-propensity logs. It cannot show its
  edge on OBD, because "a good item beats random" is clear-cut, not a marginal decision.

## Selection bias / MNAR recsys — vs IPS / SNIPS / DR (`examples/causal_mbrl_coat.py`)

You only see ratings for items users *chose* to engage with, so the observed data is Missing Not At
Random and a naive average is biased high. Coat (Schnabel et al.) ships the biased self-selected data,
an unbiased randomly-assigned test set (ground truth), and the observation propensities.

| method | avg rating | error vs truth |
|---|---:|---:|
| TRUE (MCAR test) | 2.229 | — |
| naive (biased) | 2.611 | +0.383 |
| IPS | 2.424 | +0.195 |
| SNIPS | 2.331 | +0.102 |
| doubly-robust | 2.336 | +0.107 |

This is the suite's **cleanest debiasing win**: the top MNAR techniques cut ~50–75% of the selection
bias, landing far closer to the truth than naive (and at parity with each other). **But** it hinges
entirely on the propensity model — under Tan's marginal sensitivity model, propensities off by even
Γ=1.3 put the debiased estimate anywhere in [2.07, 2.61], a band that still covers the naive number
it was supposed to correct. The band first covers the measured truth at Γ≈1.11. The point estimate is
only as trustworthy as the selection model you assume; the band is the honest part.

(Corrected 2026-08-07. The band previously reported here, [1.86, 3.15], came from a hand-rolled loop
in the example that rescaled every propensity by the same Γ without self-normalising — not Tan's MSM,
which lets each unit's odds move independently within [1/Γ, Γ] and takes the sharp extremum over that
set. The example now calls `msm_policy_value_bounds`, and the correct bands are tighter.)

## The certificate — the one place causal clearly earns its keep (`examples/causal_mbrl_certificate.py`)

On LaLonde, where every point estimate (ours and the strong contenders) is untrustworthy,
`certify_decision` **refuses to trust** the naive kill-the-program decision (tips at Γ≈1.27), and the
randomized experiment **vindicates the refusal** — the true effect is the opposite sign. No point
estimator, strong or weak, gives you that honest "don't trust this." (It is conservative by design:
NHEFS also ~Γ1.33.)

## The honest meta-lesson

| Dataset | Regime | Result |
|---|---|---|
| NHEFS | confounded effect | causal **ties** the strong contenders (well-behaved) |
| LaLonde | confounded decision | **all methods fragile**; a strong one gets the sign wrong |
| Twins | individual ITE | **all methods tie** a constant at the noise floor (unlearnable) |
| OBD | off-policy eval | top OPE **unbiased on clean logs**, fooled by confounding, sample-limited |
| Coat | MNAR selection bias | debiasing **works** (closest to truth) — but only if the propensity model is right |

Point estimates — causal or strong — are reliable only when the problem is well-behaved and are
fragile, noisy, or near-unlearnable on hard real data. The value causalrl adds is not a better point
estimate (it usually isn't); it is the **decision + certificate** layer none of the contenders
provide — an honest robustness score that correctly flags untrustworthy decisions (LaLonde, vindicated
by the RCT). No SOTA claims. The point is that correlational RL/DRL and confident point estimators
alike are structurally fooled or overconfident on hard real data, and the certificate is not.
