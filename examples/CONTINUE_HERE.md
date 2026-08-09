# Continuation briefing — the causal-LLM side track

> Feed this file to a fresh agent. It carries intent, state, constraints, and the traps that have
> already cost us once each. Read it, then read [`CAUSAL_LLM.md`](CAUSAL_LLM.md) (the source of
> truth) and [`AUDIT.md`](AUDIT.md) before touching anything.

---

## 1. The intent (why this exists)

Build toward a **large causal language model**: an LM whose reasoning goes *beyond correlations*
because causal machinery is embedded in it, not prompted into it. The long-standing goal, in the
user's words, was always to **embed the causal in the model**.

This is a controlled, CPU-sized, synthetic-first research program. It is **not** chasing a benchmark
number. The point is to isolate **mechanism**: what has to be true for causal competence to appear,
and what is merely by-construction. Every label is generated from `causalrl`, so ground truth exists
for every answer, and the library gets dogfooded in the process.

## 2. Standing constraints (do not violate)

- **Develop, commit, and push on branch `claude/train-llm-custom-lib-7e47zm`.**
- **Never open a pull request.** The user asked explicitly: *"push num branch mas sem PR."*
- Push with `git push -u origin claude/train-llm-custom-lib-7e47zm`, retrying on network failure
  with exponential backoff (2s, 4s, 8s, 16s).
- The user is Brazilian and often writes in Portuguese; answer in the language they use.

## 3. Where the program stands

Read `CAUSAL_LLM.md` for the full arc (Acts 0–6). The short version:

**Established and defensible:**
- Correlational training cannot learn causal direction; **structure is the missing ingredient**
  (0.66 → 0.82 when handed the graph).
- **Presence ≠ mediation** — a 100%-decodable structure changes nothing unless the computation is
  *routed through* it.
- **The training schedule is the bottleneck, not capacity.** Decoupled (structure-supervised)
  training beats joint end-to-end at equal capacity/data/architecture: synthetic 0.43 → 1.000
  (`causal_hybrid_twostage.py`), real Corr2Cause **0.666 ± 0.022 vs 0.469 ± 0.008 (5 seeds)**
  (`causal_corr2cause_mechanism.py`, Phase D — the differentiating result, now with error bars).
- **Prompted vs trained decoupling measured** (`causal_corr2cause_prompted.py`): structured
  prompting lifts Mistral-7B only 0.354 → 0.393 (Llama-3.2-3B 0.000 → 0.185) vs the trained
  decoupled GNN's 0.927 and the same-sample symbolic ceiling 0.893 — the training signal, not the
  prompt.
- **Real benchmark contact:** exact structure solver F1 **0.923** on the full Corr2Cause test vs
  GPT-4 ~0.29; a learned parse→GNN reasoner **matches the oracle at 0.927**; a converged distilbert
  reaches only 0.523 i.i.d. and collapses to 0.154/0.195 under relabel/refactor.
- **The routing must be an explicit computational module** — the pure-weights cell of the 2×2 is a
  **multi-seed localized negative** (see §4).
- A workshop paper draft exists: [`../docs/causal_llm/PAPER.md`](../docs/causal_llm/PAPER.md)
  ("It's the Schedule, Not the Size"), every table filled from committed artifacts.

**Weak / by-construction / owed:** much of the early "core" reasoning was hand-coded (the audit's
decisive finding); learned size-extrapolation is unsolved (0.8–0.9, not 1.0); grounding (DAS/IIT)
and active discovery are **oracle-fed**; observational discovery from raw data is fragile.

## 4. The pure-weights negative — now multi-seed (this session's close)

`causal_pure_twostage.py` runs the decoupled schedule on the **pure weights** (one GPT-2, answer
always its own next token, nothing hand-coded). The negative **replicates over SEEDS=0,1,2**, at
both capacities — four probes, each closing one explanation:

| probe | mean ± std (3 seeds) | rules out |
|---|---|---|
| self-generated edge F1 | **0.942 ± 0.023** | perception |
| teacher-forced ceiling | **0.594 ± 0.012** (2× supervision: 0.596 → 0.596 at s0) | undertraining |
| prose deleted, true graph given (`STRUCTONLY`) | **0.731 ± 0.094** cause / **0.190 ± 0.150** conf | the prose shortcut |
| 4.4× bigger model (8L/192d) | **0.753 ± 0.018** cause / **0.196 ± 0.094** conf | capacity |

Headline comparator — same function, same data:

| system | params | cause s3 | conf s3 |
|---|---|---|---|
| GNN reasoner on clean structure | **11,857** | **1.000** | **1.000** |
| GPT-2 given the same structure as tokens | **809,344** | 0.731 ± 0.094 | 0.190 ± 0.150 |
| GPT-2 capacity control | **3,583,296** | 0.753 ± 0.018 | 0.196 ± 0.094 |

Across all six trainings the best single run on `cause` is 0.828 and `conf` never exceeds 0.28 —
the GNN gap holds in every run. Two seed-0 sub-claims were **retired by replication**: the
"< scaffold's 0.818" comparison (a *same-seed* rerun drifted 0.723 → 0.828) and the
"train loss falls while generalization worsens" nuance (4L/8L loss ranges overlap). Evidence:
`results/pure_twostage_SUMMARY.md`.

**Conclusion (now solid):** decoupling fixes the *external-module* route but does **not** transfer
into LM weights. Multi-hop reachability over a serialized graph is the wall — the same thing
`causal_graph_transformer.py` found from the other side.

## 5. What to do next (priority order)

**The spine is now the internalization ladder — read
[`../docs/causal_llm/LADDER.md`](../docs/causal_llm/LADDER.md) (approved 2026-08-08)**: where must
causal computation live (external solver → external GNN → token trace → looped weights → one-shot
weights), with the decoupled schedule as the invariant lever. In order:

1. ✅ **R2 done (2026-08-08, 3 seeds, both formats)** — the token tape lifts `cause` in-dist
   0.731 → **0.926 ± 0.009** (much of R4's wall was one-shot answering; format irrelevant) but s4
   stays ~0.7 (writing AND clean reading decay OOD) and the **confounded trap survives (0.421)**:
   the shortcut corrupts the written computation itself (own-trace implies "no" only 0.562 on
   confounded pairs) and at s4 overrides even the TRUE trace (0.416 teacher-forced on an all-"no"
   set). Results + method note in `LADDER.md` §R2-RESULTS and `results/pure_twostage_SUMMARY.md`.
2. ✅ **R3 done (2026-08-08, 3 seeds × {fixed-T, T-jitter}) — the tapes CONVERGE.** A weight-tied
   looped block (219K params, no trace) reaches cause s3 0.862 ± 0.016 and the SAME s4 plateau as
   the token tape (0.664 vs 0.681); test-time T-scaling does not extrapolate; conf ≤ 0.42. So the
   OOD and confounding walls are **tape-independent** — iteration is the active ingredient, and
   what the R1 module uniquely keeps is size-invariance + shortcut-immune computation. Tables in
   `LADDER.md` §R3-RESULTS.
3. ✅ **Decoupled RLVR done (2026-08-08, seed 0, 3 arms) — instrumented negative.** Structure
   rewards polish `cause` (0.932→0.965) but cannot repair the confounded corruption even with
   confounded-pattern prompts in half of every batch: STRUCT's reward SATURATES at 1.99/2.0 on
   the RL pool — the policy is already trace-perfect on its training confounded graphs, so
   advantages vanish. The corruption is a train→fresh-instance generalization gap on a thin
   confounded slice, invisible to on-policy reward differences. Predicted fix = confounded-pattern
   DATA DIVERSITY at train time (a schedule/data lever, not a reward lever) — that is the natural
   next experiment. `LADDER.md` §RLVR-RESULTS; `results/rlvr_trace_s0*.log`.
4. **NEXT front options:** (a) the confounded-diversity training experiment the RLVR diagnosis
   predicts (cheap, CPU, closes the loop on the trap); (b) the GPU-gated cells; (c) the paper —
   the ladder now has a complete R2/R3/RLVR story to fold into `PAPER.md` or a Paper-2 skeleton.
3. **Decoupled RLVR** — verifiable *structure* rewards via the `causalrl` oracle + identifiability-
   gated abstention (outcome-only RLVR is known insufficient, arXiv:2604.22074); unifies the
   orphaned Act-5 thread with the LM arc.
4. **GPU-gated cell** (when hardware exists): decoupled schedule at 1–7B, plus the two old gate
   blockers (paraphrase 0.48 < 0.61; RoBERTa-large i.i.d. point).
5. **Paper track:** `docs/causal_llm/PAPER.md` (Paper 1, schedule) is draft-complete — LaTeX when
   the user wants venue; the ladder results accumulate into Paper 2.

Explicitly deprioritized: building another "LLM calls a causal library" agent (space occupied —
CausalAgent / causalNLP CAIS / CausalDS); re-running the HF Act-6 leg for its own sake (it runs
fine on this local box; it was only ever blocked on the cloud box).

## 6. Traps — each of these has already cost us once

- **Hand-coded ≠ learned.** The independent audit's decisive finding: the causal "reasoning" was a
  differentiable *formula* (`score = is_causal*fwd + (1-is_causal)*(1-(1-fwd)(1-bwd)(1-common))`)
  with a 2-parameter readout. Feeding it the true adjacency scores 1.000 with **zero learning**, so
  the celebrated size-generalization was **by construction**. If a result looks too clean, check
  whether an algorithm — not a model — produced it. Tag scripts `by-construction` / `oracle-fed`.
- **Always print trivial baselines.** The `confounded` set is **all-negative**: a constant-"no"
  model scores **1.000** on it. It is only meaningful read beside the balanced `cause` query. A
  degenerate 0.99 once looked like a triumph.
- **Fairness unit = epochs per objective, not gradient steps.** JOINT carries both losses on every
  item, so step-matching silently hands the decoupled arm *half* the answer supervision. This nearly
  manufactured a false positive; the discarded run is kept in
  `results/pure_twostage_stepmatched_s0.log`.
- **A teacher-forced ceiling using the TRUE graph proves nothing about graph-reading.** The true
  graph agrees with the prose, so the model may be reading either. JOINT's 0.871 "ceiling" was
  mostly prose (DIRECT alone gets 0.795). Delete the prose to isolate reasoning.
- **Keep negatives; do not tune them away.** `phase3b` (co-training collapses), the curriculum
  failure, the pure-path negative, and §4 are all *kept*. Freezing being load-bearing was learned
  from a kept negative.
- **Multi-seed anything load-bearing — and treat CPU numbers as a band, not a point.** A *same-seed*
  rerun of `causal_pure_twostage.py` moved STRUCTONLY `cause` 0.723 → 0.828 (thread-order
  nondeterminism). Sub-claims that lean on a single run's second decimal will not survive; two of
  ours didn't (see §4).
- **Don't route around blocked egress.** If `huggingface.co` 403s through the agent proxy, report it;
  don't disable TLS or unset `HTTPS_PROXY`.
- **The venv can be silently corrupted.** A killed/concurrent sync once left transformers 5.9.0
  missing 276 files vs its own wheel RECORD (`from transformers import GPT2Config` died deep in a
  GGUF lazy import). Diagnose by diffing RECORD vs disk; fix with
  `uv sync --extra torch --reinstall-package transformers`.
- **The working directory is SHARED across sessions.** Mid-run, another session checked this repo
  out to a different branch — tracked files vanished from the tree under our feet (untracked result
  logs and running jobs with absolute paths survived). If the tree is contested: park your branch in
  a git worktree (`git worktree add .claude/worktrees/<name> <branch>`) and commit/push from there;
  never `git checkout` over someone else's live session.
- **A generative eval must re-encode before it reads.** Reading the answer off the incremental
  generation buffer exposed the model to repeated-closer padding it never saw in training and
  silently depressed R2's teacher-free accuracy by ~0.25. The decomposition metrics (teacher-forced
  read + self-consistency) are what caught it — build those diagnostics BEFORE trusting any
  teacher-free number, and always re-encode `prompt + generation + single closer` for the read.

## 7. Environment

```bash
uv sync --extra torch
uv run --extra torch python examples/<script>.py
```

Knobs on `causal_pure_twostage.py`: `SEEDS`, `ARMS` (`direct,joint,joint2x,decoupled,structonly`),
`LAYERS`/`EMBD`/`HEADS`, `FAST=1` (~3 min smoke).

Ruff line-length 100; `src`/`tests` are the actual quality gate, `examples/` is held looser (long
`# STATUS:` headers are house style). Last verified stack: torch 2.12, transformers 5.9.

Every script carries a `# STATUS:` header giving its act, claim, and honesty tag — keep that up to
date, and keep `CAUSAL_LLM.md`'s script map in sync when adding one.
