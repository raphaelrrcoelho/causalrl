# Independent adversarial audit of the causal-LM experiments

An independent, skeptical auditor (separate agent, clean context, instructed to distrust the framing)
reviewed every headline script: code audit, independent re-derivation of all labels (its own BFS
reachability / d-connection, not the scripts' `reachable()`), fast reruns, a decisive
"hardcoded-formula on true adjacency" probe, class-balance analysis, and an ablation break-test. This
file records its findings verbatim-in-substance, so they are not buried.

## What is clean

- **No answer-path label leakage.** The answer head in every model receives only token ids, the queried
  pair `xs/ys`, the query *type* `is_causal`, and `present`/`entw`. It never receives `adj`, `cause`,
  `corr`, or `label`. `adj` is used **only** in the train-time auxiliary edge loss
  (`causal_hybrid_lm.py` `HybridLM.forward` has no `adj` argument).
- **Labels are correct.** 0 mismatches on `cause` and `corr` over 400 examples per generator, re-derived
  independently.
- **The "confounded" subset is degenerate (100% label "no") but cross-checked.** A constant-"no"
  predictor scores 1.000 there, so that number alone is meaningless — but the scripts always report it
  next to the *balanced* corr/cause columns (where a constant predictor scores 0.5). Reporting is honest.

## The decisive caveat — the reasoning is hand-coded, not learned

In `causal_core_do.py`, `causal_core_perception.py`, `causal_data_discovery.py`, `causal_hybrid_lm.py`
the answer is `read(score)` where `score` is a **fixed formula** of reachability `r`
(`score = is_causal*fwd + (1-is_causal)*(1-(1-fwd)(1-bwd)(1-common))`) and `read = nn.Linear(1,1)` is a
2-parameter scalar. Feeding the **true adjacency** into this formula with **zero learning** already
yields all-query and confounded accuracy **1.000 at every size**. So the only learned thing that affects
accuracy is **edge perception**; reachability + do() + back-door are a **hardwired differentiable
inductive bias**, not learned. The size-generalization is therefore largely **by construction**.

Additionally, several scripts are **fed the true structure**: `causal_core_perception.py` builds the
input edge-set directly from `e["adj"]` (no text to parse → edge-recovery ~1.0 is near-trivial), and
`causal_core_discovery.py` sets the interventional "facts" to the true oriented edges. The genuinely
learned, hard task in those scripts is the **observational** regime only.

## Per-claim verdicts

| # | Claim | Verdict |
|---|---|---|
| 1 | Hybrid GPT-2 + core ≫ vanilla | **PARTIALLY SUPPORTED.** No leakage; direction plausible; but the auditor could not reproduce hybrid > vanilla within the CPU budget (reduced repro under-trained to chance). Rests on the documented, fragile full-budget runs. |
| 2 | Pure GPT-2 does NOT internalise | **SUPPORTED** (direction/mechanism). Negative claim, most credible; not re-run (expensive). |
| 3 | Active discovery ≫ random | **SUPPORTED & reproduced** (0.815 / 0.991 / 0.926). But it is a combinatorial/oracle result (uses true `adj`); the "learned" policy trivially imitates the greedy oracle. |
| 4 | Data discovery: int > obs | **NOT VERIFIED / FRAGILE.** 20-epoch config timed out; a reduced run plateaued at chance. May be real at full budget but is slow/seed-fragile (published std 0.86 ± 0.22). Weakest claim. |
| 5 | Embedded cores size-generalize + distinguish corr/causation | **SUPPORTED mechanically, inflated framing.** The model is handed the true edge set and the computation is hand-coded, so size-generalization is by construction. |

## Bottom line

No fraud, no answer-label leakage, correct labels, honest cross-checking. The real issues are
**framing and fragility**, not cheating:

1. the causal "reasoning" is a hardwired formula — the models only learn edge perception (decisive);
2. "perception"/"interventional discovery" are partly fed the ground-truth structure;
3. the expensive claims (hybrid, data-discovery) are seed/budget-fragile and were not reproducible
   within a CPU time budget; the data-discovery claim actively plateaued at chance at reduced budget.

Claims 3 and 5 reproduce exactly but are, respectively, a combinatorial fact and a by-construction
result. This audit motivated `causal_core_learned_reasoning.py` (making the *reasoning* itself learned,
to test honestly what the hand-coded core only asserted by construction).
