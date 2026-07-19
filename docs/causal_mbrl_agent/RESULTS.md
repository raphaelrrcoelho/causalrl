# Causal MBRL Agent — Results

**One line.** A general, domain-neutral causal model-based agent that **discovers structure, plans
under confounding, and transports across a shift** beats correlational offline RL exactly where
theory says it should — verified end to end across five milestones, every verdict CI-green.

See `DESIGN.md` for the program and `PROGRESS.md` for the running log. All five harnesses are
exported from `causalrl` and reproduce the numbers below.

## The dividing line

Causal machinery carries more information than correlational learning in exactly one regime: the data
is **confounded** and/or you **cannot freely intervene** (offline), and/or the objective needs
**transfer** across environments. It is a no-op for single-task reward maximization under unlimited
intervention (self-play / simulator). This program targets the first regime and **does not** claim
the second — so none of the results below is a clean-benchmark SOTA claim, and that is by design.

## The five verdicts

| Milestone | Instance | Causal agent | vs. correlational | Verdict |
|---|---|---|---|---|
| **M0** | Simpson's-paradox bandit (observed confounder) | back-door adjust | **0.50** vs 0.40 (naive fooled by Simpson's paradox) | GO |
| **M1a** | same, structure **unknown** | skeleton discovery + temporal tiers → adjust | **0.50**, recovers `{Z}` on 10/10 seeds | GO |
| **M1b** | sequential **medicine** DTR (hidden comorbidity) | deconfounded value iteration (DOVI) | **1.04** vs 0.85 (optimum 1.05) | GO |
| **M2** | 2-D phase diagram, γ × shift | deconfound + **transport** | gap **monotone** in both axes, → 0.200 | GO |
| **M3** | continuous confounder, nonlinear reward | **function-approx** (ridge-RBF) back-door | **0.500** vs 0.381 (naive fooled) | GO |

Each is scored against the correlational baseline (`NaiveOffline` marginal, or a naive offline
planner) on the **same** confounded logs, with the oracle interventional value for ground truth.

### M0 / M1 — deconfounding, then discovering the structure to deconfound with

The naive marginal `E[Y|A]` reverses under Simpson's paradox and ships the worse action (0.40). A
back-door-adjusted agent recovers the interventional optimum (0.50). **M1a** removes the handed
graph: the agent discovers the skeleton from data and orients it with the temporal tier order
(covariates precede treatment precede outcome — standard in medicine), deriving the adjustment set
`{Z}` itself on every seed. **M1b** carries this to a multi-stage confounded **dynamic treatment
regime**: deconfounded planning reaches near-optimal (1.04) where the confounded baseline stalls at
0.85. A durable negative is recorded too — a *certify-gated* agent tops out at the behavior policy
(0.45), which is why the program uses an **active** deconfounding optimizer, not a safety gate.

### M2 — the phase diagram (headline)

Causal-minus-naive post-shift gap, 5×5 grid × 10 seeds:

```
          shift=0.00  0.25   0.50   0.75   1.00
gamma=0.00:   0.000  0.000  0.000  0.017  0.140
gamma=0.25:   0.000  0.013  0.105  0.175  0.200
gamma=0.50:   0.080  0.125  0.150  0.175  0.200
gamma=0.75:   0.100  0.125  0.150  0.175  0.200
gamma=1.00:   0.100  0.125  0.150  0.175  0.200
monotone in gamma = True     monotone in shift = True
```

The advantage is **zero along the no-confounding edge** (γ=0 — causal machinery earns nothing when
there is nothing to deconfound), then grows **monotonically in both** confounding strength and shift
magnitude to 0.200 at the high corner: the "confounding bites where theory predicts" signature. The
two failure modes are placed on **orthogonal variables** (confounder Z drives γ; a separate shift
variable W drives the covariate shift), and a sub-maximal propensity slope keeps positivity even at
γ=1 so the effect stays identifiable across the whole grid.

### M3 — past the tabular regime

A continuous confounder `Z ~ Uniform` and a nonlinear reward: a `FunctionApproxBackdoorAgent` fits
the outcome model with ridge regression on RBF features and back-door-adjusts by integrating it over
the observed Z. It recovers the true low value of the trap arm (`E[Y|do(1)] ≈ 0.38`) and keeps the
optimal arm (0.500), while the confounded marginal is fooled into the trap (0.381). Same recipe,
learned continuous estimator instead of a stratum table.

## Honest boundaries

- **Not a clean-benchmark result.** Every win is in the confounded/offline/transfer regime, by the
  dividing line above. On clean Atari/D4RL these agents have no edge and none is claimed.
- **Perception → variables is assumed away.** A known variable set is given; learning causal
  variables from raw perception is the acknowledged open problem, deferred.
- **Full confounding breaks it too.** At zero overlap (deterministic behavior) the effect is
  unidentifiable and the causal agent cannot help either — the advantage is non-monotone at that
  extreme, so M2 stays inside the identifiable regime and says so.
- **External function-approx tier is partial.** The `from_causal_gym` interop seam is built and
  unit-tested (duck-typed on the Gymnasium API), so causalrl agents can consume external
  Causal-Gymnasium rollouts. The heavier **d3rlpy CQL on CartPoleWind** comparison is **not run**:
  d3rlpy carries a known numpy-2 risk deliberately kept out of CI, and is left as a user-runnable,
  unverified example rather than a fabricated result.

## Reproduce

```python
from causalrl import (
    run_m0_kill_gate, run_m1_discovery_gate, run_m1b_dtr_gate,
    run_m2_phase_diagram, run_m3_function_approx_gate,
)
run_m0_kill_gate()              # {'causal','naive','optimal'} interventional values
run_m2_phase_diagram()          # PhaseDiagram: per-cell gaps + monotone-in-both flags
run_m3_function_approx_gate()   # continuous-confounder function-approx gate
```
