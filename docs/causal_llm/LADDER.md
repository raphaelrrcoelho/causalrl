# The internalization ladder — where must causal computation live?

**Status:** design doc, approved 2026-08-08. The unifying spine for the next phase of the causal-LLM
program. Companion to [`PAPER.md`](PAPER.md) (Paper 1: the schedule mechanism); this ladder is the
Paper-2 arc. Source of truth for what gets built and in what order; per-run evidence lands in
`examples/results/` as always.

---

## 1. The question

The program's results to date are all points on one axis: **where the causal computation lives**.

| rung | computation lives in… | evidence | verdict |
|---|---|---|---|
| R0 | exact external solver | `causal_corr2cause_solver.py` | F1 0.923 (full test) — the ceiling |
| R1 | learned external module (GNN) | `causal_hybrid_twostage.py`, `causal_corr2cause_learned.py` | 1.000 synth / 0.927 real — works |
| R2 | **tokens** — a trained trace the LM writes | *this phase* | open on causal semantics |
| R3 | **latent iteration** — looped weights | *this phase* | open on causal semantics |
| R4 | one-shot fixed-depth weights | `causal_pure_twostage.py`, multi-seed | ~0.73 / 0.19 vs GNN 1.000/1.000 — the wall |

The R4 wall is the theory's prediction, not an anomaly: fixed-depth transformers learn connectivity
heuristics that collapse OOD; chain-of-thought strictly increases expressivity; depth scaling with
input size makes connectivity expressible (arXiv:2509.22343, arXiv:2510.19753, arXiv:2412.04703).
R2 and R3 are the two principled ways to buy iteration — **token tape vs latent tape** — and there
is now a formal memory-budget separation between them (arXiv:2605.30757). Nobody has run that
comparison on causal semantics. That comparison is the centerpiece.

**The invariant lever is the decoupled schedule** (Paper 1): supervise the intermediate. R1
supervised the intermediate *structure*; R2/R3 extend the same principle one level down, to the
intermediate *computation*.

## 2. What the literature already owns (checked 2026-08-08)

- **Traces solve reachability** — a DAG-reachability scratchpad of O(|E|·log|V|) suffices
  (arXiv:2502.02393); Searchformer/Stream-of-Search train full search traces (arXiv:2402.14083).
  *Not a contribution.* What is open: (i) causal semantics beyond connectivity — d-separation and
  common-cause logic are not transitive closure; (ii) **which trace design size-extrapolates** —
  the globality-barrier theory predicts state-only ("inductive") scratchpads generalize where
  history-dependent ones don't (Abbe et al., NeurIPS 2024).
- **Looped transformers length-generalize on algorithmic tasks** and can scale reasoning depth at
  test time (arXiv:2604.07822, arXiv:2409.15647). *Not a contribution.* Open: loops × causal
  queries, and the controlled R2-vs-R3 comparison on one substrate.
- **Outcome-only RLVR does not induce causally faithful reasoning** (arXiv:2604.22074; causal
  reasoning as RLVR testbed: arXiv:2512.20760). This is Phase D's finding in RL clothing — and it
  sets up **decoupled RLVR** (§5).
- **"LLM calls a causal library" agents are an occupied space** (CausalAgent, causalNLP's CAIS,
  CausalDS). We do **not** build another wrapper; our contribution there is the boundary
  measurement + a *trained* deferral/abstention policy.

## 3. R2 — the TRACE arms (`causal_pure_twostage.py`, arms `trace` / `tracev`)

Substrate = STRUCTONLY (prose deleted, true graph given as tokens), so the trace isolates the
*reasoning* rung exactly where R4's wall was measured. Sequence:

```
query <g> true graph </g> <t> trace </t> yes|no
```

with loss on the trace tokens and the answer (the `<g>`/`<t>` markers and graph unsupervised).

**Trace semantics — backward closures.** For query "does X cause Y?": `cause = X ∈ anc*(Y)`, one
backward frontier expansion from Y. For the correlation query: with `A = anc*(X)`, `B = anc*(Y)`,
`corr = (X ∈ B) ∨ (Y ∈ A) ∨ (A ∩ B ∖ {X,Y} ≠ ∅)` — two closures plus an intersection line. The
trace is those frontier steps written out; the answer is *derived from the written trace*, and at
corpus build every trace-derived answer is asserted equal to the substrate's label (audit rule:
no silent divergence between trace logic and ground truth).

**Two designs, per the globality-barrier prediction:**
- `trace` (inductive): each step prints only the current frontier set — next step depends only on
  state. Predicted to extrapolate.
- `tracev` (verbose): each step reprints the adjacency before the frontier — history/global
  dependent, ~3× longer. Predicted to fail extrapolation. This is the ablation that makes a
  positive interpretable.

**Metrics:** teacher-free (model writes its own trace): `cause` and `confounded` accuracy at s3
(in-dist) and **s4 (headline — the lit says in-dist success is expected)**, plus final-frontier F1
vs the true closure (localizes failures to trace-following vs answer-reading). Multi-seed
(SEEDS=0,1,2) from day one; CPU numbers are a ±0.1 band (see `CONTINUE_HERE.md` traps).

**Outcome branches (pre-registered):** `trace` passes s4 → the wall was one-shot answering, and
"supervise the computation" completes the schedule thesis downward. Both pass → tape location
doesn't matter at this scale. Both fail s4 → the globality barrier survives the token tape on
causal semantics — a keepable negative that sharpens R1's module as load-bearing.

### R2 RESULTS (2026-08-08, 3 seeds, fixed harness — see the method note below)

| metric | TRACE | TRACEV | R4 no-trace | GNN (R1 function) |
|---|---|---|---|---|
| `cause` s3 | **0.926 ± 0.009** | 0.930 ± 0.041 | 0.731 ± 0.094 | 1.000 |
| `cause` s4 | 0.681 ± 0.014 | 0.723 ± 0.051 | 0.581 ± 0.084 | 0.952 |
| `conf` s3 | 0.421 ± 0.018 | 0.377 ± 0.045 | 0.190 ± 0.150 | **1.000** |
| `conf` s4 | 0.190 ± 0.031 | 0.342 ± 0.039 | 0.160 ± 0.133 | 0.893 |

Decomposition (TRACE; frontier-F1 = writing / tans = answer implied by own writes / tftr =
teacher-forced read of the TRUE trace / cons = follows own writes):

| set | fset s3→s4 | tans s3→s4 | tftr s3→s4 | cons s3→s4 |
|---|---|---|---|---|
| `cause` | 0.982→0.777 | 0.979→0.720 | 0.951→0.703 | 0.937→0.594 |
| `confounded` | 0.804→0.746 | **0.562**→0.700 | 0.860→**0.416** | 0.844→0.472 |

**Verdict — three findings, none of which was the pre-registered guess:**
1. **The tape substantially works in-distribution for reachability**: `cause` 0.731 → 0.926 in the
   LM's own weights. Much of R4's wall was *one-shot answering*, and trace format did not matter
   (inductive ≈ verbose — so the globality prediction went untested at these sizes).
2. **Extrapolation improves (+0.10/+0.14) but does not close** (0.68–0.72 vs GNN 0.952): at unseen
   size BOTH stages decay — writing (0.982→0.777) and clean reading (0.951→0.703) — compounding.
3. **The confounding trap survives the tape, and the decomposition shows why, in two channels**:
   (a) *the correlational shortcut invades the computation itself* — on confounded pairs the
   model's own written closure implies the correct "no" only **0.562 ± 0.034** of the time (vs
   0.979 on `cause`): it hallucinates X into anc*(Y) precisely when a confounder makes the pair
   correlated; (b) *at unseen size it overrides even the true trace* — teacher-forced read on
   confounded s4 is **0.416 ± 0.071** on an all-"no" set: the exonerating closure is in front of
   it and it answers "yes" anyway. The bias lives in the weights at both the writing and the
   reading stage — precisely the two failure modes a hard-coded message-passing architecture (R1)
   cannot have.

**Method note (kept — the branch's recurring lesson, now in eval form):** the first R2 runs read
the answer off the incremental generation buffer, where early-finishing rows accumulate repeated
`</t>` padding — a suffix never seen in training. The decomposition exposed it (model writes 0.982,
reads clean traces 0.951, yet "agreed with itself" only 0.651) and the harness now re-encodes
`prompt + own trace + single </t>` before reading. Teacher-free accuracy moved ~+0.25 on `cause`.
Evidence: `results/pure_twostage_trace{_s*,diag_s*,fix_s*,conf_s*}.log`; superseded logs kept.

**Implication for the ladder:** R2 sits strictly between R4 and R1 — reachability competence is
buyable in-weights with a token tape; *robust confounding discipline and size-invariance are not,
at this scale*. The R3 question sharpens accordingly: does latent iteration share the tape's two
weaknesses, or is the symbolic tape itself the problem?

## 4. R3 — the LOOPED arm

Weight-tied GPT-2-style block iterated T times (`causal_looped_lm.py`, 219K params — 3.7× FEWER
than the 809K one-shot baseline), same STRUCTONLY substrate, loss on the answer token only, no
trace. Falsifiable mechanistic check: required T should track graph diameter; test-time T-scaling
on s4 is the extrapolation probe. Known failure mode from the lit: overthinking past the fixpoint —
report the full T-curve, not the best T.

### R3 RESULTS (2026-08-08, 3 seeds × {fixed T=8, train-time T~U(4,12)})

At eval T = 8 (train budget):

| | cause s3 | cause s4 | conf s3 | conf s4 | params |
|---|---|---|---|---|---|
| R4 one-shot 4L | 0.731 ± 0.094 | 0.581 ± 0.084 | 0.190 ± 0.150 | 0.160 ± 0.133 | 809K |
| R2 token tape | **0.926 ± 0.009** | 0.681 ± 0.014 | **0.421 ± 0.018** | 0.190 ± 0.031 | 809K |
| **R3 latent tape (fixed)** | 0.862 ± 0.016 | 0.664 ± 0.006 | 0.340 ± 0.022 | **0.421 ± 0.132** | **219K** |
| R3 latent tape (jitter) | 0.836 ± 0.075 | 0.629 ± 0.051 | 0.296 ± 0.063 | 0.409 ± 0.172 | 219K |
| GNN (R1) | 1.000 | 0.952 | 1.000 | 0.893 | 11.9K |

T-sweep: fixed-T climbs T2→T8 (0.820→0.862 s3 — iteration does real work up to the train budget)
then plateaus with mild decay; jitter is flat everywhere (T-robust, as the recipe promises, but no
gain from extra loops). **Test-time compute does NOT extrapolate**: s4 never improves with more
iterations — the "more loops at eval = bigger graphs solved" transfer reported for hop-style QA
(arXiv:2604.07822) does not appear on this causal substrate, and the diameter-tracking prediction
is not supported. A FAST-budget model was fully fixpoint-flat (identical outputs ∀T≥4, smoke log) —
the sweep only comes alive at full budget.

**Verdict — the two tapes CONVERGE, and that is the finding:**
1. Latent iteration reproduces most of the token tape's in-dist gain (0.862 vs 0.926, both ≫
   0.731) with 3.7× fewer parameters and zero trace tokens — *iteration itself*, not where the
   state lives, is the active ingredient behind R2's lift.
2. Both tapes hit the SAME extrapolation plateau: s4 ≈ 0.63–0.68 (R2 0.681 ± 0.014, R3 0.664 ±
   0.006) vs GNN 0.952. The OOD wall is **tape-independent**.
3. Both leave the confounding trap far below the module (best in-dist 0.421; R3's conf-s4 0.421 ±
   0.132 is the best OOD trap number any in-weights variant has posted, but high-variance and
   still half the GNN's 0.893).
4. Loop-count jitter buys T-robustness, not accuracy (means ≈ fixed, higher seed variance).

**Implication for the ladder:** R2 and R3 are one rung, not two — "iterated computation in
weights," token or latent, buys in-dist reachability and stalls in the same two places (size
invariance, confounding discipline). What R1's explicit module still uniquely holds: exact
size-generalization, immunity of the computation to the correlational shortcut, and 18×–68× fewer
parameters. The decoupled-RLVR leg (§5) and the scale cell (§6) are the remaining ways the
in-weights story could still move.

## 5. Decoupled RLVR (the RL instantiation of the schedule thesis)

Outcome-only RLVR is known not to produce causally important reasoning. Our fix mirrors Phase D:
**verifiable process rewards on the emitted structure** — reward the trace the model writes (final-
set F1 via the oracle) plus the answer, GRPO from a shared supervised-trace state
(`causal_rlvr_trace.py`). Prediction: structure-level rewards repair the confounded trace
corruption that teacher forcing can never see (it only shows TRUE traces, never the model's own).
Abstention/identifiability-gating deferred to the Corr2Cause side (everything here is decidable).

### RLVR RESULTS (2026-08-08, seed 0, three arms — an instrumented negative)

| condition (300 GRPO steps) | conf s3 acc | conf s3 own-trace | cause s3 acc |
|---|---|---|---|
| supervised baseline (R2) | 0.404 | 0.556 | 0.932 |
| OUTCOME reward, LR 5e-5 | 0.387 | 0.514 | 0.952 |
| STRUCT reward, LR 5e-5 | 0.379 | 0.506 | **0.965** |
| OUTCOME + CONFBOOST (½ batches confounded-pattern) | 0.404 | 0.514 | 0.937 |
| STRUCT + CONFBOOST | 0.410 | 0.520 | 0.941 |

(LR 1e-5: everything flat — under-powered; LR sweep was run to rule that out.)

**Verdict: the hypothesis is refuted at this scale, and the instrumentation says precisely why.**
Structure rewards polish `cause` slightly (0.932→0.965) but do NOT repair the confounded
corruption — even when half of every RL batch is confounded-pattern (the exposure control).
The tell is **reward saturation**: STRUCT's mean reward reaches **1.99 / 2.0** on the RL pool —
the policy already writes near-perfect traces on its *training* confounded graphs, so group
advantages vanish and there is no gradient. The corruption R2 measured lives in the
**train→fresh-instance generalization gap** of a thin confounded slice (724 prompts at n=8000),
not in on-policy behavior: on-policy RL cannot observe — let alone fix — a failure that never
appears as a reward difference on its own distribution. This *sharpens* rather than contradicts
the RLVR literature: outcome rewards don't induce faithful intermediate computation
(arXiv:2604.22074), and process rewards can't either when the process is already reward-perfect
in-distribution. What the diagnosis predicts would help is **data diversity on the confounded
pattern** (a schedule/data fix, again — not a reward fix): more varied confounder configurations
at train time. Evidence: `results/rlvr_trace_s0{,_lr5,_confboost}.log`.

## 6. The scale cell (GPU-gated)

The only cell that could move the R4 wall: the decoupled schedule at 1–7B. Parked with the existing
gate blockers (paraphrase axis, RoBERTa-large i.i.d. point) until GPU time exists. GPT-4's 0.29 on
Corr2Cause says scale *alone* fails; scale × schedule is unrun.

## 7. Sequencing

1. R2 `trace` — implement + smoke + 3 seeds (CPU, ~1 day wall).
2. R2 `tracev` ablation — 3 seeds.
3. R3 looped arm + R2-vs-R3 head-to-head (the centerpiece).
4. Decoupled RLVR on the winner.
5. Scale cell when GPU materializes.

Discipline throughout: multi-seed bands, `confounded` read only beside balanced `cause`, s4 as the
headline, negatives kept, every claim's evidence in `results/`, STATUS headers + `CAUSAL_LLM.md`
map kept in sync.

## 8. Literature anchors

arXiv:2509.22343 · arXiv:2510.19753 · arXiv:2412.04703 · arXiv:2502.02393 · arXiv:2402.14083 ·
Abbe et al. NeurIPS 2024 (globality barrier / inductive scratchpad) · arXiv:2605.30757 ·
arXiv:2604.07822 · arXiv:2409.15647 · arXiv:2604.22074 · arXiv:2512.20760 · CausalAgent /
causalNLP CAIS / CausalDS (occupied agent space).
