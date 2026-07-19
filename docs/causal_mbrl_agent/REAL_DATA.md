# Causal MBRL Agent — Real-Data Demonstrations

Can `CausalMBRLAgent` solve real problems? **Yes — in the confounded/offline regime, on real data,
verified against ground truth.** Each demo points the agent at a real dataset where a
naive/correlational comparison is fooled, and checks the causal decision against an experimental or
established answer. The scripts are out-of-CI (they need network / an optional package); run them
yourself.

## The engine: multivariate g-formula

Real problems carry many mixed-type covariates, where per-stratum back-door adjustment degenerates
(each stratum has ~1 row) and a single-confounder RBF is too narrow. `GFormulaBackdoorAgent` — routed
by `CausalMBRLAgent(covariates=[...])` as the `"g_formula"` strategy — fits a per-action outcome
model over **all** covariates and standardizes to `E[Y | do(a)]`. The default is a dependency-free
numpy ridge; pass `outcome_model=<factory>` for a flexible sklearn model.

## LaLonde job-training — verifiable against a randomized experiment

`examples/causal_mbrl_lalonde.py` — Dehejia & Wahba observational data (NSW trainees vs PSID controls).

| | earnings effect ($) | decision |
|---|---:|---|
| randomized-experiment truth | +1,794 | assign |
| naive correlational (confounded) | −635 | **kill the program** ✗ |
| causal g-formula (linear) | +1,053 | **assign training** ✓ |
| causal g-formula (gradient-boosted) | −223 | kill the program ✗ |

The correlational agent kills a program that worked; the causal **linear** g-formula flips the
decision to correct and lands within ~$740 of the experimental truth. **Honest caveat:** on this
famously hard dataset the estimate is model-sensitive — the boosted model does *not* recover it. The
robust claim is the sign-flip to the right decision, not the exact dollar figure.

## NHEFS smoking → weight — verifiable against the textbook adjusted effect

`examples/causal_mbrl_nhefs.py` — Hernán & Robins, *Causal Inference: What If* (needs `causaldata`).

| | weight-change effect (kg) |
|---|---:|
| established adjusted (IP-weighting / standardization) | +3.4 to +3.5 |
| crude (naive), confounded | +2.54 |
| causal g-formula (linear) | **+3.41** |

On real medical data, the agent's g-formula recovers the canonical adjusted effect where the crude
comparison is biased low.

## Where RL/DRL is fooled — the suite (roadmap)

| Failure mode | Causal edge | Dataset | Status |
|---|---|---|---|
| Offline decision from confounded logs | g-formula | LaLonde | ✓ done (RCT-verified) |
| Confounded medical effect | g-formula | NHEFS | ✓ done (textbook-verified) |
| Off-policy evaluation under confounding | MSM bounds | Open Bandit Dataset | planned |
| Individual / counterfactual decisions | potential outcomes | Twins / IHDP | planned |
| Selection bias / MNAR feedback | deconfounded value | Coat / KuaiRec | planned |

**Guardrail:** each edge is verified against ground truth or paired with a confounding certificate,
and negatives are reported (see the LaLonde boosted-model caveat). No SOTA claims — the point is that
correlational RL is *structurally* fooled here, and the causal agent is not.
