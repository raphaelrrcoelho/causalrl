# It's the Schedule, Not the Size: The Training-Schedule Bottleneck in Causal Reasoning

*Decoupling perception from reasoning makes correlation→causation learnable for tiny models on Corr2Cause.*

**Status:** working draft (workshop-targeted). Source of truth for the narrative; compiles to LaTeX for
submission and distills to a technical post. Every number is backed by a committed artifact under
`examples/results/` (pointers in the Evidence appendix).

---

## Abstract

Large language models infer causation from correlation poorly: on the Corr2Cause benchmark (Jin et al.,
ICLR 2024) GPT-4 scores F1 0.29, near the lexical baseline. The prevailing reads are that this is a
matter of *scale/capacity* or that it can be patched by *prompting* a model to "think structurally." We
argue it is neither: the bottleneck is the **training schedule**. When perception (text → causal
structure) is *decoupled* from reasoning (structure → answer) and the intermediate structure is
supervised, a 4M-parameter encoder feeding a small graph reasoner matches an exact symbolic solver
(F1 0.927 vs 0.923) — ~3× GPT-4 — and beats a *converged* 66M distilbert both in-distribution (0.523)
and out-of-distribution. We isolate the cause with a controlled ablation: holding architecture, data, and
query-extraction fixed and varying *only* whether the structure is supervised, decoupled training beats
joint end-to-end training by a wide margin at equal capacity, and joint training *converges yet plateaus*
— so the limiting factor is the supervision schedule, not capacity. The decoupled reasoner is robust on
two **de-circularized** out-of-distribution axes (variable renaming on Jin et al.'s published split;
held-out LLM paraphrases) where the end-to-end LM collapses, and a size-agnostic version extrapolates to
causal graphs larger than any seen in training. We position the result against prompted structured
reasoning and trained-axiom approaches, and argue the differentiator is the *mechanism*: it is the
training signal, not the model size or the prompt.

---

## 1. Introduction

Corr2Cause asks: given the complete set of statistical (in)dependencies among a set of variables, does a
stated causal claim hold in *every* DAG consistent with those facts? It is a clean test of formal causal
reasoning, and LLMs are bad at it — GPT-4 F1 ≈ 0.29, fine-tuned encoders generalize poorly, and the
benchmark's own perturbations (renaming variables) collapse fine-tuned models (Jin et al., 2024).

Two explanations dominate. **Scale**: bigger models will fix it (a larger RoBERTa reaches ~0.8 in-dist
but still collapses OOD). **Prompting**: ask the model to externalize a structured intermediate before
answering (Structured Thinking Matters, arXiv:2505.18034; PC-SubQ, arXiv:2507.23488).

We make a third, mechanistic claim: **the bottleneck is the training schedule.** The task factorizes into
*perception* (read the premise → the causal structure: skeleton + v-structures) and *reasoning* (decide
necessity over the Markov-equivalence class). End-to-end ("joint") training optimizes the final label and
hopes the intermediate structure emerges; it does not. **Decoupling** the two stages and supervising the
structure makes the task learnable for a *tiny* model — and the gap between the two schedules, at equal
capacity, is the evidence.

**Contributions.**
1. **A controlled mechanism ablation** (§4): one architecture, same data and query extraction, varying
   *only* structure supervision → decoupled ≫ joint at equal capacity; joint converges yet plateaus.
   *This is the differentiating result*: it isolates the schedule from capacity, architecture, and data.
2. **A trained decoupled reasoner** (§5) that matches the exact symbolic ceiling (0.927 vs 0.923) and
   beats a converged distilbert **in-distribution** (0.523), retiring the "decoupling only helps OOD"
   worry.
3. **Two de-circularized OOD axes** (§5): variable renaming (Jin et al.'s *published* refactorization
   split) and held-out LLM paraphrases, where the LM collapses and the structure reasoner does not.
4. **Size extrapolation** (§6): a size-agnostic reasoner generalizes to graphs larger than any trained
   on, beating a fair size-agnostic transformer baseline.
5. **A prompted-vs-trained head-to-head** (§5): the published *prompted* decoupling regime is far weaker
   than a *trained* one — consistent with the mechanism.

Honest scope: small models, a single benchmark family, CPU-scale. The claim is about the *mechanism* and
the *gap*, not a leaderboard number.

---

## 2. Related work

**Corr2Cause.** Jin et al. (2024) introduce the benchmark and report that LLMs barely beat trivial and
that fine-tuned models collapse under variable renaming. We replicate the collapse on our own LM and use
their published perturbation split for a non-circular OOD test.

**Prompted structured reasoning.** Structured Thinking Matters (arXiv:2505.18034) prompts an LLM to build
a knowledge graph then answer (Qwen3-32B 32.7→48.3 F1). PC-SubQ (arXiv:2507.23488) decomposes into
PC-algorithm sub-questions with reasoning-specialist models (o-series, DeepSeek-R). These *decouple by
prompting*; we show a *trained* decoupling is far stronger and isolate *why* via the schedule ablation.
They report no controlled training ablation and (for Structured Thinking) no OOD test.

**Trained causal reasoning.** Axiomatic Training (arXiv:2407.07612) fine-tunes transformers on
demonstrations of causal axioms (transitivity, d-separation), reaching SOTA on Corr2Cause with
Llama-3-8B. Their lever is *demonstration* (CoT-style) training; ours is *explicit decoupled structure
supervision*, and we isolate that it is the structure signal — not training in general — that matters
(§4).

**Small models via training.** VibeThinker-3B (arXiv:2606.16140) shows a 3B model reaching frontier
*general* reasoning via a training paradigm (curriculum SFT + RL + distillation), arguing reasoning is
"parameter-dense" (compressible into a small core). Our result is a controlled, causal-specific instance
of the same meta-claim — capacity is not the bottleneck — with a different, isolated lever (structure
supervision rather than outcome-reward RL).

---

## 3. Setup

**Benchmark.** Corr2Cause (HF `causalnlp/corr2cause`): premise = full (in)dependence pattern over 2–6
variables; hypothesis = a causal claim (direct cause, confounder, collider, (non-)ancestor). Binary F1
on the positive class (~15.5% positive). Standard train/test/validation splits; we additionally use the
published `perturbation_by_refactorization` split (variables renamed to arbitrary letters) for OOD.

**Decoupled architecture.** *Perception*: a `bert-tiny` encoder (4M params) maps the premise text to a
soft skeleton **S** and v-structure evidence **D**. *Reasoning*: a small GNN consumes S/D as a
differentiable adjacency (`bmm`), plus a one-hot query (template, X, Y) parsed by regex, and predicts the
label. The two are trained separately (perception on structure targets; GNN on structure→label) and
composed. The query extraction is shared across all conditions.

**Symbolic ceiling.** An exact parse→MEC-enumeration→necessity solver (dogfooding the `causalrl`
library's d-separation) gives the *structure-decidable upper bound*: F1 **0.923** on the full 1162-example
test (P 0.918 / R 0.928), vs lexical TF-IDF 0.365 and GPT-4 0.29. It is an oracle, not a learned LM; it
sets the ceiling and validates that the benchmark is structure-decidable from the premise.

---

## 4. The mechanism: schedule, not capacity (main result)

Every cross-system comparison (a decoupled GNN vs an end-to-end LM) confounds the *schedule* with the
*architecture*. We remove the confound: **one architecture** — the `bert-tiny` perception → GNN of §3 —
trained under **two schedules**, holding *everything else fixed* (same N=6000 premises, same query
extraction, same epoch budget). The only variable is whether the intermediate structure is supervised.

- **Decoupled**: perception trained on structure targets, GNN trained on structure→label, composed.
- **Joint**: the *same* perception+GNN trained end-to-end on text→label only, no structure supervision.

**Result (clean test F1, mean ± std over 5 seeds):**

| schedule (identical model / data / query extraction) | clean test F1 |
|---|---|
| reasoner given the regex structure (upper reference) | 0.843 ± 0.012 |
| **Decoupled** (structure-supervised, two-stage) | **0.666 ± 0.022** |
| **Joint** (end-to-end, label-only) | 0.469 ± 0.008 |

*(Per-seed values: decoupled [0.659, 0.626, 0.683, 0.688, 0.673], joint [0.474, 0.474, 0.476, 0.453,
0.467]; `examples/results/d_mechanism_seed{0..4}.log`, aggregate in `d_mechanism_multiseed.log`. The
earlier single-seed reference run is `d_mechanism_run.log`: 0.833 / 0.692 / 0.474.)*

Same capacity, same data, same architecture — only the schedule differs — and **decoupled beats joint by
+0.20 F1 (mean over 5 seeds; the gap is 9× the larger std)**. Critically, joint training *converges*
(train loss 1.06 → 0.81 over 5 epochs) yet plateaus
far below decoupled: the end-to-end label gradient does not induce the intermediate structure on its own.
This is the real-data echo of the synthetic two-stage demonstration (`causal_hybrid_twostage.py`: joint
~0.43 → decoupled 1.0 in-distribution). The bottleneck is the **training schedule (structure
supervision), not capacity, architecture, or data.** It also coheres with §5: the joint number (~0.47)
sits beside a converged distilbert (0.523), i.e. end-to-end LMs of this class plateau near ~0.5.

*Scope:* N=6000 / single architecture / no augmentation, so the absolute ceiling here (~0.83) is below
the full-data §5 headline (0.927); the controlled **gap at equal everything** is the claim, not the level.

---

## 5. Trained decoupling: in-distribution and OOD

**In-distribution (full 1162-example test, F1):**

| system | F1 |
|---|---|
| majority / lexical TF-IDF | 0.00 / 0.365 |
| GPT-4 (Jin et al.) | 0.29 |
| Mistral-7B, prompted (local, 2-shot) | 0.35 |
| converged distilbert-base (66M, end-to-end LM) | 0.523 |
| **trained decoupled parse→GNN (ours)** | **0.927** |
| symbolic ceiling (exact solver) | 0.923 |

The trained decoupled reasoner *matches the oracle* and beats a converged distilbert **in-distribution**
(0.927 vs 0.523) — not merely OOD. The distilbert is also **not relabel-invariant**: it collapses under
consistent variable renaming and *worsens* with training (epoch-1 0.292 → epoch-2 0.154), i.e. it learns
lexical/letter shortcuts rather than structure.

**Out-of-distribution, de-circularized.** Self-made OOD splits invite a circularity objection, so we use
two disjoint, non-circular axes:

| axis | structure reasoner | end-to-end distilbert |
|---|---|---|
| variable renaming — Jin et al.'s *published* refactorization | 0.920 (refactor-invariant) | 0.195 (collapse) |
| held-out LLM paraphrases (Mistral-7B rewrites, disjoint from training) | 0.06 → **0.48** (with diverse aug) | regex front-end 0.00 |

The renaming axis replicates Jin et al.'s headline collapse on our own LM and validates the synthetic
relabel as a faithful proxy (synthetic 0.154 ≈ real 0.195). The paraphrase axis is an honest arc: narrow
connective-swap training was *circular* (held-out 0.06); retraining on diverse full-rewrites recovers
genuine generalization to unseen paraphrases (0.48), with a residual gap to clean (0.61) we attribute to
bert-tiny-scale perception.

**Prompted vs trained head-to-head (Corr2Cause, fixed N=150 sample, F1 on the positive class;
`causal_corr2cause_prompted.py`, single cached run):**

| system | F1 |
|---|---|
| Mistral-7B — direct (2-shot) | 0.354 |
| Mistral-7B — structured-thinking prompting (arXiv:2505.18034 reimpl) | 0.393 |
| Llama-3.2-3B — direct (2-shot) | 0.000 |
| Llama-3.2-3B — structured-thinking prompting | 0.185 |
| symbolic ceiling (same sample, 100% coverage) | 0.893 |
| trained decoupled GNN (full test, ref) | 0.927 |

Measured and consistent with the mechanism: structured prompting helps over direct (+0.04 for
Mistral-7B; +0.19 for Llama-3.2-3B, whose direct prompting is degenerate — zero true positives), but
the prompted decoupling regime stays far below a *trained* one on the same sample — it is the training
signal, not the prompt.

---

## 6. Size extrapolation

Corr2Cause stops at 6 variables. To test breadth we generate random DAGs at N=4..9 with `causalrl`, feed
the reasoner the PC-style structure, and label a Markov-equivalence-invariant query ("is X a definite
ancestor of Y?") via the library's Meek orientation. Trained only on N∈{4,5}; the rest is extrapolation.

| reasoner | N=4 | N=5 | N=6* | N=7* | N=8* | N=9* |
|---|---|---|---|---|---|---|
| size-agnostic GNN | 1.00 | 1.00 | 0.99 | 0.98 | 0.95 | **0.93** |
| graph transformer (fair size-agnostic baseline) | 1.00 | 0.98 | 0.96 | 0.94 | 0.90 | **0.86** |
| fixed-size MLP (strawman) | 0.99 | 0.94 | 0.72 | 0.55 | 0.46 | **0.41** |

(* = unseen graph size.) Message-passing extrapolates beyond a model that *can* handle any N (the
transformer), and the edge widens out of distribution — the inductive bias helps, not merely
size-agnosticism.

---

## 7. Limitations

- **Scale.** Tiny models, CPU. A much larger fine-tuned LM (RoBERTa-large, ~0.8 in-dist per Jin et al.)
  would narrow the *in-distribution* gap (untrainable on our hardware), but shows the same OOD collapse —
  so the OOD finding is the scale-robust one, and the i.i.d. tie vs a *big* LM is untested.
- **Oracle ceiling.** The symbolic solver is an exact algorithm, not a learned model; it bounds the task,
  it is not a baseline we claim to have learned end-to-end.
- **Perception residue.** The learned paraphrase robustness is real but imperfect (held-out 0.48 < clean
  0.61) at bert-tiny scale, and requires diverse augmentation.
- **Single benchmark family.** Corr2Cause (+ `causalrl`-generated graphs for §6). A second real benchmark
  is future work.
- **Query extraction.** The (template, X, Y) query is regex-parsed; perception of the *premise* structure
  is the learned, stressed component.

---

## 8. Conclusion

Causal reasoning in language models, on a benchmark where LLMs fail, is gated by the **training schedule**
— whether the intermediate causal structure is supervised — not by capacity, architecture, or the prompt.
Decoupling perception from reasoning makes the task learnable for a 4M-parameter model, matches an exact
solver, beats a converged LM in- and out-of-distribution, and extrapolates across graph sizes. The
mechanism is the message: a controlled ablation that isolates the schedule is what distinguishes this from
prompted and demonstration-trained prior work.

---

## Evidence appendix (committed artifacts)

- §4 mechanism: `examples/causal_corr2cause_mechanism.py`; `examples/results/d_mechanism_run.log`
  (single seed); `examples/results/d_mechanism_seed{0..4}.log` + aggregate (multi-seed).
- §5 i.i.d./OOD: `examples/causal_corr2cause_learned.py`, `_perception.py`, `_realood.py`,
  `causal_corr2cause_b1_lm.py`; `examples/results/{b1_distilbert_*,c_realood,c_paraphrase_*}_run.log`.
- §5 prompted head-to-head: `examples/causal_corr2cause_prompted.py`; `examples/results/d_prompted_run.log`.
- §6 size extrapolation: `examples/causal_mec_scaling.py`; `examples/results/b2_size_extrapolation_run.log`.
- Symbolic ceiling: `examples/causal_corr2cause_solver.py`.
- Synthetic two-stage: `examples/causal_hybrid_twostage.py`.
- Full program narrative + per-script status: `examples/CAUSAL_LLM.md`, `examples/PHASE01_RESULTS.md`.
