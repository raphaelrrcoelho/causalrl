# Causal MBRL Agent — Real-Data Demonstrations

Can `CausalMBRLAgent` solve real problems? **Yes — and the honest story is more interesting than
"causal beats naive."** These demos benchmark the causal agent against *strong* contenders — IPW,
doubly-robust AIPW, propensity-score stratification (what a competent practitioner actually uses, and
what DoWhy/EconML compute) — on real data, checked against ground truth. The scripts are out-of-CI
(they need network / `causaldata` / scikit-learn); run them yourself.

## The engine: multivariate g-formula

Real problems carry many mixed-type covariates, where per-stratum back-door adjustment degenerates
and a single-confounder RBF is too narrow. `GFormulaBackdoorAgent` — routed by
`CausalMBRLAgent(covariates=[...])` — fits a per-action outcome model over **all** covariates and
standardizes to `E[Y | do(a)]` (dependency-free numpy ridge default; optional sklearn factory hook).

## The honest benchmark against strong contenders

### NHEFS smoking → weight — parity with the strong methods (well-behaved)

`examples/causal_mbrl_nhefs.py` — Hernán & Robins, *What If* (needs `causaldata`).

| method | effect (kg) |
|---|---:|
| established adjusted (textbook) | +3.4 to +3.5 |
| naive diff-in-means (strawman) | +2.54 |
| **ours: g-formula (linear)** | **+3.41** |
| strong: IPW | +3.28 |
| strong: AIPW (doubly-robust) | +3.33 |
| strong: propensity stratification | +3.23 |

On well-behaved data the causal agent **agrees with the strong contenders** (all ~+3.3 kg), all
clearing the confounded naive. Honest parity — a member of the serious-methods family, not a strawman
win.

### LaLonde job-training — everyone is fragile (pathological)

`examples/causal_mbrl_lalonde.py` — truth = the NSW randomized experiment, +$1,794.

| method | effect ($) | decision |
|---|---:|---|
| randomized-experiment truth | +1,794 | assign |
| naive diff-in-means (strawman) | −635 | kill ✗ |
| **ours: g-formula (linear)** | **+1,053** | assign |
| strong: IPW | +231 | assign |
| strong: AIPW (doubly-robust) | +508 | assign |
| strong: propensity stratification | −597 | kill ✗ |

Point estimates scatter from −$597 to +$1,053, and a **strong** method (propensity stratification)
gets the sign *wrong*. No single number is trustworthy here.

## The certificate — where causalrl actually adds value

`examples/causal_mbrl_certificate.py`. When point estimates are fragile, the honest move is not a
better number — it's quantifying the fragility. `certify_decision` reports the odds-ratio Γ of
unmeasured confounding at which the decision tips:

```
naive decision  : prefer control  (contrast −635)
certified robust: False
tips at         : Γ≈1.27   (unmeasured confounding this strong overturns it)
randomized-experiment truth: +1,794  (prefer treated — the OPPOSITE sign)
```

The certificate **refuses to trust** the naive kill-the-program decision (tips at a modest Γ=1.27),
and the randomized experiment **vindicates the refusal**: the true effect is the opposite sign. A
confident point estimate — strong or naive — ships the wrong call; the certificate flags it. (It is
conservative by design: NHEFS's cleaner decision also carries a moderate Γ≈1.33 — an honest "proceed
with caution", not a false all-clear.)

## Honest bottom line

- On observed-confounder problems, causalrl **ties** the strong contenders where the data is
  well-behaved (NHEFS) and is **as fragile as everyone else** where it isn't (LaLonde) — no
  point-estimate edge, and we say so.
- causalrl's real edge is the **decision + certificate** layer none of them provide: an honest
  robustness score that correctly flags untrustworthy decisions (LaLonde, vindicated by the RCT).

## Where RL/DRL is fooled — the suite (roadmap)

| Failure mode | Causal edge | Dataset | Status |
|---|---|---|---|
| Offline decision from confounded logs | g-formula + certificate | LaLonde | ✓ done (RCT-verified) |
| Confounded medical effect | g-formula + certificate | NHEFS | ✓ done (textbook-verified) |
| Off-policy evaluation under confounding | MSM bounds | Open Bandit Dataset | planned |
| Individual / counterfactual decisions | potential outcomes | Twins / IHDP | planned |
| Selection bias / MNAR feedback | deconfounded value | Coat / KuaiRec | planned |

**Guardrail:** benchmarked against strong contenders (not strawmen); parity claimed only where it
holds; negatives reported. No SOTA claims — the point is that correlational RL/DRL and confident point
estimators alike are *structurally* fooled or overconfident here, and the certificate is not.
