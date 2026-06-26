# Causal LLM — the research program

> **Thesis.** Causal competence in a language model must be **installed** as explicit structure,
> **routed through** (not merely encoded), and **decoupled** from perception in training. It does
> *not* emerge from correlational next-token learning — and naive end-to-end training fails to learn
> it even when every part is present.

**Start here.** This is the single source of truth for the causal-LLM side track (the `causal_*` and
`rlvr_*` scripts in this folder). It tells the discovery as one arc, then maps every script to its
place in that arc and its honest status. Detailed numbers live in [`PHASE01_RESULTS.md`](PHASE01_RESULTS.md);
the adversarial audit in [`AUDIT.md`](AUDIT.md); the architecture in [`ARCHITECTURE.md`](ARCHITECTURE.md).
The older framing docs (`CAUSAL_LLM_RESEARCH.md`, `FRONTIER_PROPOSAL.md`, `FRONTIER_PROPOSAL_v2.md`)
are **archived** — superseded by this file.

Scope guard: this is a controlled, CPU-sized, synthetic proof of concept (graphs of 2–7 variables,
tiny GPT-2s, labels generated from `causalrl` so every answer has ground truth). It is **not** a
frontier-scale result. The point is to isolate *mechanism*, not to win a benchmark.

---

## The arc

### Act 0 — The problem
Standard LLMs are *causal parrots*: they recite correlational regularities. By the Causal Hierarchy
Theorem (Bareinboim et al.), interventions (L2) and counterfactuals (L3) are **not** recoverable from
observational text alone without structural assumptions. The bet of this program: install the
structure, route the computation through it, and verify against a known-truth world (`causalrl`).

### Act 1 — Diagnosis: correlation isn't enough
- **Correlational training can't learn causal direction** — reproduced the Corr2Cause phenomenon
  (stuck ≈ train ≈ val, below a correlation heuristic). `causal_transfer_corr2cause.py`,
  `causal_reasoning_scaffold.py`.
- **Structure is the missing ingredient** — handed the graph, accuracy jumps to the
  Markov-equivalence ceiling (0.66 → 0.82). `causal_reasoning_scaffold.py`.
- **Presence ≠ mediation** — a structure that is 100% *decodable* doesn't change answers; you must
  *route the computation through* it (0.66 → 0.81). `causal_grounding_install.py`.
- **Honest negative:** a d-separation transformer learns shortcuts (adjacent ⇒ connected), not
  d-separation — 0.96 on balanced data but 0.745 on the natural distribution.
  `causal_graph_transformer.py` (+ `_diagnose`). Curriculum doesn't fix size-extrapolation
  (`causal_reasoning_curriculum.py`).

### Act 2 — Mechanism: install an explicit causal core
An embedded core that perceives a soft adjacency, routes through it, and applies the causal
computation: `causal_core_architecture.py` (propagation), `causal_core_do.py` (`do()` / back-door),
`causal_core_perception.py` (relational perception). **Interventions break the observational ceiling**
(0.82 → 0.99, `causal_beyond_correlation.py`); an NCM recovers `P(Y|do X)` from observation via graph
surgery (`causal_ncm_reasoning.py`). Grounding the latent variable inside a *real* model's activations
via DAS + interchange/IIT: `causal_grounding_das_iit.py` (+ `phase2/3/3b`).

> **Honest caveat (the audit, Act 3):** in these core scripts the reachability/`do()`/back-door step
> is a **hand-coded differentiable formula** — only *perception* is learned, so the size-generalization
> is *by construction*. The grounding phases are **oracle-fed** (the true regime variable is known),
> and `phase3b` shows the gain **collapses** under co-training.

### Act 3 — Learnability: is the reasoning learned, or wired in?
The audit ([`AUDIT.md`](AUDIT.md)) is the pivot. Making the reasoning *itself* learned:
`causal_core_learned_reasoning.py` — a GNN matches the hand-coded core in-distribution (1.0, incl.
confounding) but only **partially** size-extrapolates (0.8–0.9 at size 5). Discovery from raw data:
interventional works (edge 0.92) but **observational collapses to the MEC limit** (confounded 0.28),
and it's fragile (`causal_data_discovery.py`); `causal_active_discovery.py` and
`causal_core_discovery.py` are **oracle-fed** (use the true adjacency).

### Act 4 — Coupling to a real LM
The spine of the program, in the order it was discovered:
1. **Bespoke coupling** — prose → hand-coded core → answer (`causal_lm_coupling.py`).
2. **From-scratch LMs** — a 6-token toy (`causal_lm_from_scratch.py`), then a real GPT-2 trained from
   scratch with a BPE tokenizer (`causal_lm_real_from_scratch.py`).
3. **Hybrid** — real GPT-2 perception + the (hand-coded) core (`causal_hybrid_lm.py`).
4. **Pure path — negative.** A real GPT-2 does **not** internalise the computation in its weights
   (~0.15–0.5 vs hybrid ~1.0). `causal_pure_lm.py`.
5. **Fully-learned hybrid — the apparent wall.** GPT-2 perception + a *learned* GNN reasoner learns
   the general `cause` query (1.0) but **fails the confounding trap (~0.43)**. `causal_hybrid_learned.py`.
6. **Diagnosis.** The 0.43 is an end-to-end **joint-training** artifact, not perception (edge F1 0.86;
   the perceived graph drives the exact algorithm to ~1.0) and not capacity. `causal_perception_bottleneck.py`.
7. **The fix.** Decoupled **two-stage** training (perception on the edge loss; a structure-only GNN
   reasoner on clean structure; composed) → **1.000 ± 0.000** confounded in-dist / **0.933 ± 0.003**
   held-out, from prose, nothing hand-coded. `causal_hybrid_twostage.py`.

L3 in a learned head: counterfactuals via twin-network abduction-action-prediction, climbing a
crutch-removal ladder — fixed SCM (`causal_counterfactual_twin.py`, by-construction) → random
parameters (`causal_counterfactual_general.py`, genuine generalization) → random topologies
(`causal_counterfactual_topo.py`, latest step). Axiomatic-training reasoner (d-separation rule from
traces): `causal_reasoner_prototype.py` → hardened trainer `causal_reasoner_train.py`.

### Act 5 — RL-adjacent thread
`rlvr_causal_verifier.py`: RL (GRPO) with a *verifiable causal reward* (`causalrl`'s identifiability
oracle) + an honesty penalty → the model learns to **abstain** on non-identifiable estimands. Today
this is **orthogonal** to the LM arc (not wired into it) — see the roadmap for the obvious bridge.

### Act 6 — Frontier: real-benchmark contact (Phase 1 + Phase 2 done)
The real benchmark (Corr2Cause) — long cited, never run — has now been tackled. Its premises are
complete d-separation patterns and its hypotheses are causal claims, mapping onto our pipeline
exactly. **Phase 1:** an exact parse→MEC→necessity solver (`causal_corr2cause_solver.py`, dogfooding
`causalrl`'s d-separation) scores **F1 0.923 on the full 1162-example test set** vs lexical 0.365 and
GPT-4 ~0.29 — the *symbolic ceiling*. **Phase 2** turned the thesis into a learned test:

- A decoupled **parse→GNN reasoner** (`causal_corr2cause_learned.py`) reaches **F1 0.927** on the full
  test — *matching the symbolic oracle* (0.923), ~3× GPT-4; it is **exactly relabel-invariant**, but
  the regex front-end collapses under paraphrase (→0).
- A **learned perception** (`causal_corr2cause_perception.py`) feeding the *same* reasoner **recovers
  paraphrase (0 → 0.73)**, and with **relabel augmentation (B3) is robust on BOTH OOD axes** (clean
  0.68 / relabel 0.64 / paraphrase 0.66) — decoupling localizes robustness to a cheap, retrainable
  front-end you augment per-axis.
- On a controlled `causalrl`-generated benchmark (`causal_mec_scaling.py`) the size-agnostic GNN
  **extrapolates** to graph sizes never trained on (N4–5 → N9: **0.93**), beating a **fair**
  size-agnostic graph transformer (0.86, B2) and crushing the fixed-size MLP strawman (0.41) — the
  message-passing bias helps even versus a model that *can* handle any N.

A live local **Mistral-7B** (prompted, 2-shot) manages only **0.35** — ~lexical, ≈ the paper's GPT-4
(0.29) — i.e. LLM scale doesn't help; the structure-routing does. **B1 (now measured, not assumed):** a
*converged* distilbert end-to-end LM reaches only **F1 0.523 i.i.d.** (not the ~0.8–0.95 a far larger
RoBERTa-large would, per Jin et al.), so the decoupled GNN wins **in-distribution too** (0.927 vs 0.523)
— *not* OOD-only as earlier feared — and the LM **collapses on relabel** (0.523→0.154), worsening with
training (it learns lexical/letter shortcuts, not structure). "Decouple to generalize on Corr2Cause" is
already published as a *prompted* method (arXiv 2505.18034); the open, differentiated claim is the
**training-schedule mechanism** + a *trained* reasoner that wins both i.i.d. and OOD. Evidence (data, not
just prose): `examples/results/b1_distilbert_*` + `causal_corr2cause_b1_lm.py`. **Workshop-grade** today;
see **Roadmap**.

---

## Where it's strong / weak

**Strong (genuinely learned, multi-seed, defensible):**
- **Real-benchmark keystone** — exact structure-routing solves Corr2Cause at **F1 0.92** (full test)
  vs GPT-4 0.29; the thesis transfers off synthetic data (`causal_corr2cause_solver.py`).
- **Phase-2 learned decoupling (real data)** — a trained parse→GNN reasoner *matches the symbolic
  oracle* (**0.927** vs 0.923); the size-agnostic reasoner **extrapolates N4→N9 (0.93)**, beating a
  fair size-agnostic graph transformer (0.86, B2) and the fixed-size MLP (0.41); a learned perception
  is **robust on both OOD axes** (B3:
  relabel+paraphrase aug). All dogfood `causalrl` (`causal_corr2cause_learned.py`, `_perception.py`,
  `causal_mec_scaling.py`).
- **Strong-LM baseline beaten on both axes (B1)** — a *converged* distilbert reaches only **0.523
  i.i.d.** (vs GNN 0.927) and **collapses on relabel** (0.154, worsening with training) — it learns
  lexical shortcuts, not structure. Backed by data, not prose: `examples/results/b1_distilbert_*`
  (run log + JSON + checkpoint sha256), trainer `causal_corr2cause_b1_lm.py`.
- **Real, non-circular OOD (C)** — on Jin et al.'s *published* `perturbation_by_refactorization` split,
  the decoupled reasoner is **refactor-invariant** (0.923 → 0.920, full coverage) while distilbert
  **collapses** (0.523 → 0.195) — replicating Jin et al.'s headline finding on our model and validating
  the synthetic relabel as a faithful proxy (`causal_corr2cause_realood.py`, evidence in `results/`).
- The **two-stage fix** — fully-learned, 1.0 / 0.93 confounded, stable (`causal_hybrid_twostage.py`).
- **Learned reasoning is real** in-distribution (`causal_core_learned_reasoning.py`).
- The **pure-path negative** — clean, multi-seed (`causal_pure_lm.py`).
- **Counterfactual (L3) generalization** over random parameters (`causal_counterfactual_general.py`).
- **Intellectual honesty as infrastructure** — an adversarial audit, negatives kept, multi-seed on
  load-bearing claims.

**Weak / fragile / by-construction:**
- Much of the "core" reasoning is **hand-coded**; the headline size-generalization was the algorithm,
  not learning (the audit's decisive finding).
- **Learned** size-extrapolation is unsolved (0.8–0.9, not 1.0).
- Grounding (DAS/IIT) is **oracle-fed**; `phase3b` collapses.
- Active / structured discovery is **oracle-fed**; raw-data observational discovery is **fragile**.
- The win is **not OOD-only** anymore (B1 retired that fear — see Strong), but real caveats remain: the
  regex perception is paraphrase-fragile; the learned perception trades peak clean accuracy for
  cross-axis robustness (B3 made it relabel-invariant too: 0.68/0.64/0.66); the size-extrapolation
  now has a fair baseline (B2: GNN 0.93 > graph transformer 0.86 > MLP strawman 0.41), retiring the
  strawman criticism; and "decouple to
  generalize on Corr2Cause" is partly **occupied** by prompted prior work (arXiv 2505.18034) — our open
  angle is the *training-schedule mechanism* + a trained reasoner, not "structure helps". **Scale
  caveat:** a much larger fine-tuned LM (RoBERTa-large ~0.8 i.i.d., Jin et al.) would narrow the i.i.d.
  gap, but it's untrainable on this box and shows the same OOD collapse.
- Tiny models, synthetic prose, CPU; held-out numbers drift run-to-run. GPU works for short bursts but
  a *sustained* training run wedged the WSL2 driver — the LM trainer now checkpoints/resumes.

---

## Roadmap (the compass)
1. **Real benchmark — Corr2Cause.** ✅ *Phase 1 + 2 done.* Symbolic ceiling 0.92; the learned parse→GNN
   reasoner matches it (0.927); OOD robustness + learned perception + size extrapolation measured. The
   conference path is the dedicated plan below.
2. **Close learned size-extrapolation** (0.8–0.9 → ~1.0): scratchpad/recurrence, scheduled sampling
   true→perceived, stronger algorithmic alignment. (Item-2c already shows clean N4→N9 extrapolation on
   the *definite-ancestor* query.)
3. **Real-data perception** — discovery from messy text, not SVO templates (the recurring bottleneck;
   the Phase-2b learned perception is the first step).
4. **Paper spine — chosen:** the **training-schedule decoupling** mechanism ("causal reasoning in LMs
   is gated by training schedule, not capacity or perception") — synthetic two-stage + the Phase-2
   real-data echo. The plan below hardens it.
5. **Unify causal-RL × LLM:** use the RLVR causal verifier as the **reward signal** to train the LM's
   causal reasoning/honesty (today they are separate toys).

### Phase 2 → conference-grade (the plan)
Sequenced by leverage, with a **decision gate** so we don't over-invest before knowing it can clear the
bar. (Today's Phase-2 result is a clean **workshop** contribution; the gate decides if conference is
realistic.)
- **A — Land what we have.** ✅ scripts + results + docs committed (workshop-ready artifact exists).
- **B — Fair baselines (days; cheapest, highest leverage).**
  - **B1** ✅ *done* — converged distilbert: clean **0.523** / relabel **0.154** / paraphrase 0.546
    (`causal_corr2cause_b1_lm.py`; evidence in `results/`). It does **not** tie i.i.d. (GNN 0.927 wins)
    *and* collapses OOD → the decoupling win is i.i.d. **and** OOD at this scale; the earlier "OOD-only"
    concession is retired. (A larger LM would narrow the i.i.d. gap per Jin et al., but is untrainable
    here — GPU thermally wedges, CPU OOMs; trained CPU-side via gradient-checkpointing + accumulation,
    learning-exact at ~3 GB.)
  - **B2** ✅ *done* — added a fair size-agnostic **graph-transformer** baseline: at N9, GNN **0.93** >
    transformer **0.86** > MLP strawman **0.41** (gap widens out-of-distribution), so message-passing
    helps even vs a model that *can* handle any N. (`causal_mec_scaling.py`; evidence
    `results/b2_size_extrapolation_run.log`.)
  - **B3** ✅ *done* — relabel-augmented the learned perception (relabel the premise + re-parse its
    target): relabel **0.328 → 0.637**, now robust on both axes (clean 0.68 / relabel 0.64 /
    paraphrase 0.66). Decoupling targets *all* OOD axes from the cheap front-end alone.
    (`causal_corr2cause_perception.py`; evidence `results/b3_perception_run.log`.)
- **C — Real OOD, de-circularized.** ✅ *refactor axis done* — evaluated on **Jin et al.'s published
  `perturbation_by_refactorization`** (variables renamed to arbitrary letters): the decoupled symbolic
  reasoner is **refactor-invariant** (0.923 → 0.920, full coverage) while distilbert **collapses**
  (0.523 → 0.195), replicating Jin et al. on our model AND validating the synthetic relabel as a proxy
  (synthetic 0.154 ≈ real 0.195). (`causal_corr2cause_realood.py`; evidence `results/c_realood_run.log`.)
  *Remaining:* the **paraphrase** axis — Jin's `perturbation_by_paraphrasing` is 3-class NLI + paraphrases
  the *hypothesis*, so the regex query-parser can't consume it; needs a learned query parser /
  premise-only LLM paraphrases.
- **🚪 GATE (B done; C refactor ✅, paraphrase pending):** on REAL data the structure reasoner already
  **beats the LM on the refactor axis** (invariant vs collapse) and wins i.i.d. at distilbert scale
  (0.93 vs 0.52). Two open items before a confident conference call: a **non-circular paraphrase axis**
  (the 2nd real OOD axis) and a **strong-LM i.i.d. point** (RoBERTa-large — does it tie i.i.d.?).
  **Leaning workshop-strong / conference-plausible**; don't spend D–E before closing those two.
- **D — Mechanism + positioning (days).** Lead with the training-schedule ablation (joint vs decoupled,
  synthetic + real); run the *prompted* method (arXiv 2505.18034; Mistral/Qwen → JSON graph) as a
  head-to-head baseline, positioning us as the *trained/mechanistic* counterpart.
- **E — Scale & breadth (1–2 wk, compute).** Bigger models, multiple seeds + error bars, a 2nd real
  benchmark.
- **F — Write-up.** Workshop after A–B; conference draft after the gate passes.

---

## The map — every side-track script, by act and status

Status legend: **canonical** (current, defensible) · **superseded→** (kept for lineage) ·
**by-construction** (the result is the hand-coded algorithm, not learning) · **oracle-fed** (given the
true structure) · **honest-negative** (a kept negative result) · **fragile** (seed/budget-sensitive) ·
**support/infra** (harness, not a claim) · **research** (latest step, results unverified) ·
**exploratory** (orthogonal probe).

### Act 1 — Diagnosis
| Script | Shows | Status |
|---|---|---|
| `causal_transfer_corr2cause.py` | correlational training can't learn direction | canonical |
| `causal_reasoning_scaffold.py` | structure is the missing ingredient; CoT doesn't help | canonical |
| `causal_grounding_install.py` | presence ≠ mediation; routing (0.66→0.81) | canonical |
| `causal_graph_transformer.py` | learns shortcuts, not d-sep (0.745 nat-dist) | honest-negative |
| `causal_graph_transformer_diagnose.py` | diagnostic for the above | support |
| `causal_reasoning_curriculum.py` | curriculum doesn't fix size-extrapolation | honest-negative |
| `run_dsep_multiseed.py` | multi-seed d-separation harness | support/infra |
| `causal_robustness.py` | multi-seed robustness of load-bearing claims | support |

### Act 2 — Mechanism
| Script | Shows | Status |
|---|---|---|
| `causal_beyond_correlation.py` | interventions break the obs ceiling (0.82→0.99) | canonical |
| `causal_ncm_reasoning.py` | NCM recovers P(Y\|do X) from observation | canonical |
| `causal_core_architecture.py` | propagation / size-gen | by-construction |
| `causal_core_do.py` | `do()` / back-door | by-construction |
| `causal_core_perception.py` | relational perception (given structure) | by-construction · oracle-fed |
| `causal_grounding_das_iit.py` | DAS+IIT grounding in real activations | oracle-fed |
| `causal_grounding_phase2.py` | orthonormal DAS disentangles carriers | oracle-fed · support |
| `causal_grounding_phase3.py` | IIT on frozen carriers (delicate) | oracle-fed · fragile |
| `causal_grounding_phase3b.py` | co-training reintroduces collapse | refuted |

### Act 3 — Learnability (the audit pivot)
| Script | Shows | Status |
|---|---|---|
| `causal_core_learned_reasoning.py` | learned reasoner: 1.0 in-dist, partial size-extrap | canonical |
| `causal_data_discovery.py` | discovery from raw data; obs → MEC limit (0.28) | fragile |
| `causal_core_discovery.py` | embedded discovery (interventional facts = true edges) | oracle-fed |
| `causal_active_discovery.py` | choosing interventions (uses true adjacency) | oracle-fed |

### Act 4 — Coupling to a real LM
| Script | Shows | Status |
|---|---|---|
| `causal_lm_coupling.py` | first bespoke prose→core→answer | superseded→ `causal_hybrid_lm.py` |
| `causal_lm_from_scratch.py` | 6-token toy: do vs see | superseded→ `causal_lm_real_from_scratch.py` |
| `causal_lm_real_from_scratch.py` | real GPT-2 from scratch (BPE) — from-scratch branch | canonical (sub-thread) |
| `causal_hybrid_lm.py` | GPT-2 + hand-coded core; data/utils others reuse | foundational (imported) · by-construction core |
| `causal_pure_lm.py` | a real GPT-2 does NOT internalise it | canonical-negative |
| `causal_hybrid_learned.py` | fully-learned hybrid FAILS confounding (0.43) | canonical (end-state superseded by two-stage) |
| `causal_perception_bottleneck.py` | the 0.43 is joint TRAINING, not perception | canonical |
| `causal_hybrid_twostage.py` | the FIX: decoupled two-stage → 1.0 in-dist | canonical |
| `causal_reasoner_prototype.py` | axiomatic d-sep rule from traces (didactic) | superseded→ `causal_reasoner_train.py` |
| `causal_reasoner_train.py` | hardened, device-agnostic reasoner trainer | support/infra |

### L3 — Counterfactual (capability within Acts 2/4)
| Script | Shows | Status |
|---|---|---|
| `causal_counterfactual_twin.py` | fixed-SCM twin network (8-state lookup) | by-construction |
| `causal_counterfactual_general.py` | random parameters — genuine generalization | canonical |
| `causal_counterfactual_topo.py` | random topologies — latest step | research |

### Act 5 — RL-adjacent
| Script | Shows | Status |
|---|---|---|
| `rlvr_causal_verifier.py` | RL + causal-verifier reward → honest abstention | exploratory (orthogonal) |

### Act 6 — Frontier (real benchmark)
| Script | Shows | Status |
|---|---|---|
| `causal_corr2cause_solver.py` | exact structure solver on REAL Corr2Cause: F1 0.92 (full test) vs GPT-4 0.29; parser [A-Z]-generalized so it handles Jin's refactorization | canonical · keystone (symbolic ceiling) |
| `causal_corr2cause_learned.py` | Phase 2: end-to-end LM (2a) vs decoupled parse→GNN (2b, F1 0.927=oracle) + OOD relabel/paraphrase; GPU-checkpointed LM | canonical · Phase-2 main |
| `causal_corr2cause_perception.py` | Phase 2b/B3: learned text→structure perception, relabel+para aug → robust on BOTH OOD axes (clean 0.68/relabel 0.64/para 0.66) into the same reasoner; evidence in `results/` | canonical · learned-perception |
| `causal_mec_scaling.py` | Phase 2c/B2: size extrapolation N4→N9 — GNN 0.93 > fair graph transformer 0.86 (B2) > MLP strawman 0.41; dogfoods causalrl Meek + d-sep | canonical · size leg |
| `causal_corr2cause_b1_lm.py` | Phase B1: CONVERGED strong distilbert end-to-end LM (clean 0.523 / relabel 0.154 / paraphrase 0.546) — GNN wins i.i.d. too; mem-minimal learning-exact (grad-ckpt + accum). Evidence in `results/` | canonical · fair-baseline |
| `causal_corr2cause_realood.py` | Phase C: REAL de-circularized OOD on Jin et al.'s published perturbation_by_refactorization — decoupled symbolic refactor-INVARIANT (0.92→0.92) vs distilbert COLLAPSE (0.52→0.20); validates the synthetic relabel proxy. Evidence in `results/` | canonical · real-OOD |

---

## Running things
All scripts are CPU-sized and self-contained. The torch-dependent ones run via the `torch` extra:

```
uv run --extra torch python examples/<script>.py
```

Labels are generated from `causalrl`, so every run is checked against ground truth. Multi-seed scripts
take a `SEEDS` env var (e.g. `SEEDS=0` for a fast smoke). Headlines reproduce in-distribution;
held-out numbers drift run-to-run (CPU nondeterminism).

## Document index
- **This file** — canonical program (thesis, arc, map, roadmap).
- [`PHASE01_RESULTS.md`](PHASE01_RESULTS.md) — detailed, reproducible results log.
- [`AUDIT.md`](AUDIT.md) — independent adversarial audit.
- [`ARCHITECTURE.md`](ARCHITECTURE.md) — the embedded-core design blueprint.
- `CAUSAL_LLM_RESEARCH.md`, `FRONTIER_PROPOSAL.md`, `FRONTIER_PROPOSAL_v2.md` — **archived** framing
  docs (history; superseded by this file).
