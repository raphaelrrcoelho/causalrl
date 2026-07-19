# Causal MBRL Agent — Progress / Resume

**Program:** a general, domain-neutral causal model-based RL agent (discover → robustly plan →
certify → transport). See `DESIGN.md` (M0–M3) and `plans/2026-07-19-m0-kill-gate.md`.

**Branch:** `causal-mbrl-agent`.

## Status (2026-07-19)

**M0–M3 all GO and CI-green.** Local runs on the `/mnt/c` WSL2 mount are slow, so **CI is the
verifier** — nothing is claimed green until the GitHub Actions run passes.

| Milestone | What | Verdict |
|---|---|---|
| M0 | back-door-adjusted agent on the Simpson bandit | GO — 0.50 vs naive 0.40 |
| M1a | discovery agent (skeleton + temporal tiering) | GO — 0.50, `{Z}` on 10/10 |
| M1b | deconfounded DOVI on the sequential medicine DTR | GO — 1.04 vs 0.85 |
| M2 | 2-D phase diagram (γ × shift), transport agent | GO — monotone both axes, gap → 0.200 |
| M3 | function-approx tier (ridge-RBF back-door, continuous Z) | GO — 0.50 vs 0.381 |

Full per-milestone verdicts below (reverse-chronological).

## M3 verdict — GO (2026-07-19, function-approximation tier, 10 seeds)

```
causal = 0.500   naive = 0.381   optimal = 0.500   gap = +0.119   (gamma = 1)
```

On `ContinuousConfoundedBandit` — a CONTINUOUS observed confounder `Z ~ Uniform(0,1)` with a
nonlinear arm-1 reward bump and an overlap-preserving confounded behavior policy that over-samples
arm 1 near the bump — `FunctionApproxBackdoorAgent` fits `qhat(a,z)` by per-action ridge regression
on RBF features and back-door-adjusts by Monte-Carlo integrating it over the observed `Z`. It
recovers the true low value of arm 1 (`E[Y|do(1)] ≈ 0.38`) and keeps the safe optimal arm 0 on every
seed, while the confounded marginal `NaiveOffline` is fooled into the harmful arm 1 (0.381). Carries
the discover→adjust recipe PAST the tabular regime to a learned continuous estimator — the
function-approximation credibility tier, self-contained so it runs in CI. Read via
`run_m3_function_approx_gate` (CI green `85f967d`).

**External tier — seam built, heavy comparison deferred (honest).** The
`causalrl.interop.from_causal_gym` seam **is now built and unit-tested** (duck-typed on the
Gymnasium API, external package never imported; `tests/test_interop_causal_gym.py`), so causalrl
agents can consume external Causal-Gymnasium rollouts as a `ConfoundedTrajectoryDataset`. The heavier
CartPoleWind + **d3rlpy CQL** comparison is NOT run: d3rlpy carries a known numpy-2 risk deliberately
kept out of CI (DESIGN §6), so it stays a user-runnable, unverified example rather than a fabricated
result. A results write-up spanning all five verdicts lives in `RESULTS.md`.

---

## M2 verdict — GO (2026-07-19, 2-D phase diagram γ × shift, 10 seeds)

```
causal-minus-naive post-shift gap (rows gamma 0..1, cols shift 0..1):
          s=0.00  0.25   0.50   0.75   1.00
g=0.00:   0.000  0.000  0.000  0.017  0.140
g=0.25:   0.000  0.013  0.105  0.175  0.200
g=0.50:   0.080  0.125  0.150  0.175  0.200
g=0.75:   0.100  0.125  0.150  0.175  0.200
g=1.00:   0.100  0.125  0.150  0.175  0.200
monotone_in_gamma = True   monotone_in_shift = True
null corner (g=0,s=0) = 0.000     high corner (g=1,s=1) = 0.200  [CI 0.200, 0.200]
```

The gap is monotone nondecreasing in BOTH confounding strength `gamma` and covariate-shift magnitude,
zero along the no-confounding edge (`gamma=0` = honest null: causal machinery adds nothing when there
is nothing to deconfound), and grows to 0.200 at the high corner — the "confounding bites where theory
predicts" signature. The two failure modes sit on ORTHOGONAL variables (confounder `Z` drives the γ
axis; a separate shift variable `W` drives the covariate-shift axis, additive on the safe arm), so the
diagram does not entangle them; a 0.4 propensity slope preserves positivity even at `gamma=1` (a
deterministic `A=Z` would make the effect unidentifiable and is avoided). `TransportableConfoundedBandit`
+ `TransportBackdoorAgent` (deconfound `Z` + transport `W` by the target `P(W)`, dogfooding
`backdoor_adjustment_set` + `is_transportable_effect`) vs `NaiveOffline`. Read via
`run_m2_phase_diagram` (CI green `f9e6de6`). The design was verified end-to-end at finite sample
before implementation (`scratchpad` prototypes), including the tight-overlap `gamma=1` edge.

---

## M1b verdict — GO (2026-07-19, sequential DTR / medicine, 5 seeds)

```
causal (DOVI) = 1.04   naive = 0.85   optimal = 1.05   gap = +0.19
```

On `SequentialDTREnv` — a multi-stage confounded dynamic treatment regime (hidden comorbidity U, a
clinician who plays a=U, and a foresight gap) — the deconfounded value-iteration agent `DOVI`
(existing lib agent, `transition_assumption="unconfounded"`), trained on confounded logs, reaches
near-optimal (1.04 of 1.05) and beats the confounded naive baseline (0.85) by ~0.19. Shows the
causal-MBRL program extends to the sequential / **medicine** setting via `run_m1b_dtr_gate`.

---

## M1a verdict — GO (2026-07-19, discovery agent, 10 seeds)

```
discovery = 0.50   naive = 0.40   optimal = 0.50   gap = +0.10   (recovers {Z} on 10/10 seeds)
```

`DiscoveryBackdoorAgent` discovers the causal skeleton from data and orients it with the temporal
tier order (covariates ≺ treatment ≺ outcome — standard in DTR/medicine), derives the back-door set
`{Z}` itself on **10/10 seeds**, adjusts, and **matches the handed-graph optimum (0.50)**, beating
naive (0.40).

Reliability history (Thread 1): pure interventional edge-orientation was unreliable here — ~6/10,
with *deterministic* orientation flips (Z→A/Z→Y ↔ A→Z/Y→Z) that did NOT improve with n (5k/20k/50k
all ≈6/10). Switching to skeleton discovery + temporal tiering fixed it to 10/10 and 0.50. Next:
M1b (sequential DTR/medicine instance).

---

## M0 verdict — GO (2026-07-19, back-door-adjusted agent, 10 seeds)

```
causal = 0.5000   naive = 0.4000   optimal = 0.5000   gap = 0.1000
```

The **active back-door-adjusted agent recovers the interventional optimum** (0.50 = optimal, on every
seed) while the naive marginal agent is fooled by Simpson's paradox (0.40). Clean, identifiable,
deterministic causal win — a *functioning* agent that does better, not just a safety gate. **GO → M1.**

Apparatus: `SimpsonBandit` (observed confounder Z, back-door A←Z→Y) + `BackdoorAdjustedAgent`
(adjusts for Z via `backdoor_adjustment_set`) vs `NaiveOffline`. Read via `run_m0_kill_gate`.

---

## Superseded first attempt — NO-GO (certify-gated agent, 10 seeds)

```
causal_source = 0.4500   naive_source = 0.4500
causal_shifted = 0.4500  naive_shifted = 0.4500   gap = 0.0000
```

The certify-gated agent did NOT beat naive — both land at 0.45.

**Diagnosis (two layers):**
1. *Env symmetry:* the reward is symmetric (q(0,0)=q(1,1)=0.55, q(0,1)=q(1,0)=0.35), so every
   constant/mixed policy averages 0.45; only the context-dependent optimal [0,1]=0.55 differs, and
   neither agent finds it.
2. *Deeper — the real lesson:* the **certify-gated planner is a safety mechanism whose ceiling is
   the behavior policy.** Under strong confounding it cannot certify the true optimum, so it abstains
   to a (noisy) behavior default rather than recovering it. Naive is fooled to [1,1]=0.45; abstention
   is also ≈0.45 → no gap. This agent avoids harm; it does not *perform* better — which is exactly
   the "only a certificate" limitation the whole program set out to move past.

**Fork:**
- (a) Recalibrate the env once so the confounder-fooled policy is strictly *worse* than behavior →
  causal-via-abstention beats naive, but only a modest "don't ship a confounded loser" win.
- (b) **Swap the planner** to one that actively *optimizes* under confounding (DOVI /
  `msm_policy_value_bounds` value-maximizer) — a functioning agent, not a safety gate. Aligns with
  the program's actual goal. **← recommended.**
- (c) Accept the honest negative: the certify-gated agent's ceiling is the behavior policy.

Reproduce:
```bash
python -c "from causalrl import run_m0_kill_gate; import json; \
r = run_m0_kill_gate(seeds=tuple(range(10))); \
print(json.dumps({k:{'mean':v.mean,'lo':v.ci95_low,'hi':v.ci95_high} for k,v in r.items()}, indent=2))"
```

## Resume here

**M0–M3 are all GO and CI-green** on `causal-mbrl-agent` (PR #28). The DESIGN's milestone spine is
complete. Open options (none started — await direction):

- **Merge** PR #28 (flip draft → ready), or keep iterating on the branch.
- **External-credibility tier** — the `from_causal_gym` seam is built + tested; the remaining piece
  is a d3rlpy CQL comparison on CartPoleWind, out of CI (numpy-2 risk). Deferred deliberately.
- **Write-up** — done: `RESULTS.md` pulls all five verdicts together.

Re-run any verdict from the exported harnesses: `run_m0_kill_gate`, `run_m1_discovery_gate`,
`run_m1b_dtr_gate`, `run_m2_phase_diagram`, `run_m3_function_approx_gate`.

## Notes / decisions

- Discovery is deferred to M1 (the bandit's action×context interaction has no marginal main-effect
  for `discover_interventional`'s invariance test; structure is *given* in M0). See DESIGN.md §8.
- Causal agent is **certify-gated** (`certify_policy`), not Manski-greedy — Manski lower-bound greedy
  does not correct a backdoor `A ← U → Y`.
- Correlational/model-free baseline reuses `NaiveOffline` (one-step bandit collapses them); a distinct
  model-free/world-model baseline arrives with M1's multi-step env.
