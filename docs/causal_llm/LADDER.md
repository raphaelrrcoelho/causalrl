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

**Outcome branches:** `trace` passes s4 → the wall was one-shot answering, and "supervise the
computation" completes the schedule thesis downward. Both pass → tape location doesn't matter at
this scale. Both fail s4 → the globality barrier survives the token tape on causal semantics — a
keepable negative that sharpens R1's module as load-bearing.

## 4. R3 — the LOOPED arm

Weight-tied GPT-2 (shared block iterated T times) at parameters matched to the 4L baseline, same
STRUCTONLY substrate, same schedule. Falsifiable mechanistic check: required T should track graph
diameter; test-time T-scaling on s4 is the extrapolation probe. Known failure mode from the lit:
overthinking past the fixpoint — report the full T-curve, not the best T. Build after R2 lands
(shares eval harness).

## 5. Decoupled RLVR (the RL instantiation of the schedule thesis)

Outcome-only RLVR is known not to produce causally important reasoning. Our fix mirrors Phase D:
**verifiable process rewards on the emitted structure** — reward the graph the model writes (edge
F1 via the `causalrl` oracle), the answer, and honesty (identifiability-gated abstention, extending
`rlvr_causal_verifier.py` from its orphaned Act-5 toy onto the LM arc). Prediction: structure-level
rewards close what outcome-level rewards can't. Builds on the R2 harness (the trace/graph is the
rewardable intermediate). Sequenced after R2/R3.

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
