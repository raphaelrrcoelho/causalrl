# Causal MBRL Agent — Design Spec

- **Date:** 2026-07-19
- **Status:** Approved design (brainstorming). Next step: implementation plan (writing-plans).
- **Branch:** `causal-mbrl-agent`

## 1. Motivation

causalrl ships causal *analysis* (certificates, identification, partial-ID bounds) and tabular/demo
agents, but is not used as a *functioning agent* library the way SB3 / d3rlpy are. This program adds
a **general, domain-neutral causal model-based RL agent** that discovers a structural causal model
(SCM) of its environment and plans a robust policy under it — the capability that lets a causal agent
succeed where correlational RL hits its limit. The certificate is not a substitute for the agent; it
is the trust artifact the agent emits.

### The dividing line (when causal structure adds value over correlational learning)

Causal machinery carries more information than correlational learning in exactly these regimes:

- data is **confounded** and/or you **cannot freely intervene** (offline / observational), **or**
- the objective requires **transfer** across environments, **extrapolation to interventions not yet
  tried**, or **counterfactual planning**.

It is redundant only for **single-task reward-maximization under unlimited intervention**
(self-play / simulator / verifier — the "DeepMind pantheon": Go, AlphaFold, AlphaProof, Atari from a
sim). This program targets the first set and explicitly does **not** claim the second.

### Testbed domains (illustrative — NOT the library's identity)

Dynamic treatment regimes (medicine / biology / pharma — the founding causal-RL application), confounded
finance / off-policy decisions, structured control. The agent is **general**; these are instances used
to measure it. The library stays domain-neutral (organized by the Bareinboim 9-task taxonomy). No
finance- or LLM-specific code or branding enters the library.

## 2. Goals / Non-goals

**Goals**
- A general `CausalMBRLAgent` in causalrl: **discover → robustly plan → certify → transport**, over an
  SCM belief, reusing existing library primitives (dogfooding; extend the library in-scope rather than
  inlining).
- A **falsifiable demonstration** that a causal agent transfers where a correlational one cannot, in
  controlled oracle worlds, with an early kill gate.

**Non-goals**
- Beating SOTA on clean single-task benchmarks (clean Atari / MuJoCo / D4RL) — out of scope by the
  dividing line.
- Domain-specific (finance / LLM) code or positioning.
- Causal **representation** learning from raw perception (pixels → causal variables). We assume a
  **known variable set**; perception→variables is the acknowledged open problem, deferred.

## 3. The agent architecture

A causal model-based RL loop; every box maps to an existing causalrl primitive.

```
        ┌───────────────────────────────────────────────────────────┐
        │  interact  (online interventions, or offline logs)         │
        ▼                                                            │
  discover SCM ──► SCM belief ──► robust plan ──► act ───────────────┘
  (discover_        (structure +   (causal_q_bounds:
   interventional/   mechanisms +   worst-case Q over SCM
   discover_latent)  confounding)   uncertainty + MSM Γ)
        │                                  │
        │                                  ▼
        │                            certify_policy  ── DecisionCertificate (trust layer, free)
        ▼                                  │
  on env shift ──► identify_transport reuses invariant mechanisms ◄──┘
```

Units (each has a single purpose, an interface, and a dependency):

- **Discovery unit** — infers/refines the SCM over a known variable set from interaction data.
  Interface: `data → SCM belief (graph + mechanisms + confounding)`. Depends on `discover_interventional`
  / `discover_latent` (FCI/PAG for latent-aware). Actively intervenes to orient edges (observation alone
  gives only the Markov-equivalence class).
- **Robust planning unit** — chooses actions maximizing the confounding-/uncertainty-robust value.
  Interface: `SCM belief → policy`. Depends on `causal_q_bounds` (worst-case lower-bound Q in the
  planning/Bellman step), tabular value iteration, and `msm_*` bounds for the confounded case.
- **Certification unit** — emits a machine-checkable certificate on the shipped policy. Interface:
  `(policy, confounded dataset) → DecisionCertificate`. Depends on `certify_policy` (Tan MSM).
- **Transport unit** — on environment shift, reuses invariant mechanisms instead of relearning.
  Interface: `(source SCM, shift) → transported policy/estimand`. Depends on `identify_transport`.

`CausalMBRLAgent` composes these behind an `act` / `update` (and offline `fit`) surface, domain-neutral.

## 4. Crux falsification experiment

A novel **oracle world**: known variables, *unknown* graph + a hidden confounder, and a controllable
**shift**. This one experiment exercises the whole loop.

- **Phase A (discover):** the agent actively intervenes to recover the SCM.
- **Phase B (plan):** it computes a robust policy under its inferred SCM.
- **Shift** the world (change mechanisms/reward) and measure transfer.
- **Baselines:** correlational model-based (predictive world-model, no causal graph); model-free
  (fitted-Q / Dreamer-lite); **ablation** (our agent fed a wrong/correlational graph).
- **Verdict 1 — discovery** (independently scored): active discovery drives structural error (SHD)→0,
  materially faster/more reliably than observational-only discovery, and yields
  interventional/counterfactual predictions the correlational model gets wrong.
- **Verdict 2 — policy/transfer** (independently scored): the causal agent earns higher **post-shift**
  return than the correlational and model-free baselines (causal mechanisms transfer; correlations
  don't), and its certificate correctly flags unsafe transfers.
- **Kill criterion:** if the causal agent does **not** beat the *correlational world-model* on transfer
  even in this controlled oracle world, the thesis is the IRM outcome — **stop.**

The two verdicts are scored and reported **separately**, so a miss on the harder policy-value half
cannot contaminate the certificate read (or vice versa).

## 5. Pass / fail criteria (falsifiable, per verdict)

- **Verdict 1 (discovery) — PASS:** under active interventional discovery, median SHD to the true graph
  reaches ≤1 within the interventional budget and is strictly lower than observational-only discovery at
  matched budget (95% CI over seeds excludes 0 gap); the recovered SCM's interventional/counterfactual
  predictions have materially lower error than the correlational model's on held-out do-queries.
- **Verdict 2 (policy/transfer) — PASS:** post-shift return of the causal agent exceeds both the
  correlational-MBRL and model-free baselines, with a per-seed gap whose 95% CI excludes 0 at high
  confounding/shift, **and** the gap is monotone-increasing in Γ and shift magnitude (the "confounding
  bites where theory predicts" signature); certificate abstain precision/recall (or ROC-AUC) on
  unsafe-transfer detection beats naive confounding-blind OPE.
- **Overall kill:** Verdict 2's transfer gap vs correlational-MBRL is ≤0 (CI includes 0) across the
  controlled worlds → stop the program; fall back to the certification-layer-only positioning.

## 6. Substrate & baselines

**Tiered substrate** (must have oracle ground truth to score both verdicts):
- **Primary (tabular oracle):** `envs/suite/` confounded envs — confounded-chain / `scbandit`,
  `dtr` / `seq_dtr` (the medicine/DTR instance), `mabuc`, `gridworld`. Full SCM oracle → exact true
  policy value via intervention; controllable confounding strength Γ and shift magnitude.
- **External-credibility tier:** one CausalGym env (`CartPoleWind`) via
  `causalrl.interop.from_causal_gym` (function-approx scale; exercises the interop seam).
- Heavy confounded-Atari **deferred** (needs GPU + teacher backends; out of window).

**Baselines:** correlational model-based (predictive world-model); model-free (tabular fitted-Q, and a
Dreamer-lite on the CartPoleWind tier); ablation (agent given a wrong/correlational graph); and
**d3rlpy CQL/IQL** on the CartPoleWind tier (via `scale.d3rlpy`), **isolated and kept out of CI** per the
known numpy-2 risk, so it can strengthen the story without breaking the core go/no-go.

## 7. Interop seam (verified feasible)

```python
causalrl.interop.from_causal_gym(env, behavior_policy, n_episodes) -> ConfoundedTrajectoryDataset
```
Mirrors the existing `pettingzoo_to_trajectory_log` adapter. **Data-level** (collect CausalGym `see()`
behavior-policy rollouts, whose `info` carries the realized natural action *and the hidden variables* for
oracle checks). No full SCM bridge required; their env's `SCM` subclasses `gym.Env` and differs from
causalrl's torch `StructuralCausalModel`, but the probe consumes datasets, not live SCM surgery. Their
`ctf_do` is `NotImplementedError` — irrelevant, since this is an L2 / partial-ID probe.

## 8. Milestone spine (kill-gate first)

| Milestone | Scope | Verdict | ~Time |
|---|---|---|---|
| **M0 — gate** | 1 small oracle world, 1 shift level, discovery-on vs correlational-MBRL | Does causal transfer beat correlational *at all*? Go/no-go. | Wk 1 |
| **M1** | Full discover→plan→certify loop on tabular oracle worlds (incl. DTR/medicine instance); few seeds | Both verdicts, single-axis | Wk 2–4 |
| **M2** | 2-D phase diagram (Γ × shift) × ~10 seeds; correlational-MBRL baseline; harden `CausalMBRLAgent` API | Both verdicts, joint | Wk 4–7 |
| **M3** | CartPoleWind function-approx tier + d3rlpy CQL; write-up | External-credibility tier | Wk 7–10 |

The program hangs on **M0**: one cheap week that either shows a causal agent transferring where a
correlational one can't, or saves the other nine.

**Implementation staging:** the first implementation plan covers **M0 plus the shared scaffolding** it
needs — the `CausalMBRLAgent` skeleton, one oracle world with controllable Γ/shift, the discovery +
robust-planning units wired end-to-end, and the correlational-MBRL baseline. M1–M3 each get their own
plan once M0 clears.

## 9. Metrics & experimental shape

2-D phase diagram over **Γ (confounding strength) × shift magnitude**, ~10 seeds. Plot both verdicts
across both axes; expect the causal advantage to grow with Γ and shift. Any coverage bound (top-N,
sampled cells, dropped seeds) is **logged explicitly** — no silent truncation.

## 10. Risks & open questions

- **IRM fragility:** causal transfer may fail to beat correlational even in controlled worlds. That is
  precisely the M0 kill signal — cheaply learned.
- **Identifiability:** an SCM is not recoverable from observation alone (MEC only); the agent must
  actively intervene to orient edges. Latent confounders cap what is identifiable.
- **Perception → variables:** deferred. Known variable set assumed; note the ceiling.
- **Discovery scaling:** keep variable sets small; document where discovery reliability degrades.
- **d3rlpy dependency:** isolated from CI (numpy-2 risk); optional tier only.
- **Dogfooding:** route all causal ops through causalrl; extend the library in-scope rather than inlining.

## 11. Attribution

Conceptual antecedents: Causal-DQN (worst-case partially-identified bound in the Bellman update — Li,
Zhang, Bareinboim, NeurIPS 2025) and Causal-Flow-Q (robust offline objective, ICML 2026). External env
substrate: CausalGym (CausalAILab). Cite papers; port no external code without attribution + license.
