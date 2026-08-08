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
  (`causal_hybrid_twostage.py`), real Corr2Cause 0.474 → 0.692 (`causal_corr2cause_mechanism.py`,
  Phase D — the differentiating result).
- **Real benchmark contact:** exact structure solver F1 **0.923** on the full Corr2Cause test vs
  GPT-4 ~0.29; a learned parse→GNN reasoner **matches the oracle at 0.927**; a converged distilbert
  reaches only 0.523 i.i.d. and collapses to 0.154/0.195 under relabel/refactor.
- **The routing must be an explicit computational module** — see §4, this session's addition.

**Weak / by-construction / owed:** much of the early "core" reasoning was hand-coded (the audit's
decisive finding); learned size-extrapolation is unsolved (0.8–0.9, not 1.0); grounding (DAS/IIT)
and active discovery are **oracle-fed**; observational discovery from raw data is fragile.

## 4. What the most recent session added

**The missing cell of the 2×2.** Every "an LM can't internalise causal reasoning" result had been
measured under the **joint** schedule — the one Phase D proved deficient — while the fix that worked
routed the answer through an **external GNN**. So the negative was confounded with its schedule.

`causal_pure_twostage.py` runs the decoupled schedule on the **pure weights** (one GPT-2, answer
always its own next token, nothing hand-coded). The negative **survives**, and is now localized —
four probes, each closing one explanation:

| probe | result | rules out |
|---|---|---|
| self-generated edge F1 | **0.940** | perception |
| 2× answer supervision | ceiling **0.596 → 0.596** | undertraining |
| prose deleted, true graph given (`STRUCTONLY`) | **0.723** cause / 0.186 conf | the prose shortcut |
| 4.4× bigger model (8L/192d) | **0.745**, worse elsewhere, **lower** train loss | capacity |

Headline comparator — same function, same data, same seed:

| system | params | cause s3 | conf s3 |
|---|---|---|---|
| GNN reasoner on clean structure | **11,857** | **1.000** | **1.000** |
| GPT-2 given the same structure as tokens | **809,344** | 0.723 | 0.186 |
| GPT-2 capacity control | **3,583,296** | 0.745 | 0.088 |

At **302×** the GNN's parameters the transformer still cannot compute reachability over a serialized
graph — while fitting the training set *better* (loss 0.365 → 0.320). Memorization rising as
generalization falls is what an architectural mismatch looks like.

**Conclusion:** the thesis tightens rather than breaks. Decoupling fixes the *external-module* route
but does **not** transfer into LM weights. Multi-hop reachability over a serialized graph is the
wall — the same thing `causal_graph_transformer.py` found from the other side (shortcuts, not
d-separation).

**Owed:** all of §4 is **seed 0 only**. Multi-seed replication is the one open gap.

## 5. What to do next (priority order)

1. **Multi-seed `causal_pure_twostage.py`** (`SEEDS=0,1,2`). Cheap, and it is the only thing standing
   between the §4 negative and a solid claim. If the 0.723-vs-1.000 gap holds, it's real; if it
   moves a lot, weaken the conclusion in `CAUSAL_LLM.md` accordingly.
2. **The HF-dependent Act-6 leg** — `causal_corr2cause_{solver,learned,perception,realood,mechanism,b1_lm}.py`.
   These need `huggingface.co` (dataset `causalnlp/corr2cause` + pretrained models) and were blocked
   by egress policy on the cloud box. They ran fine previously.
3. **The two named gate blockers** (from `CAUSAL_LLM.md`'s roadmap): tighten the paraphrase axis
   (held-out 0.48 vs clean 0.61) and add a **RoBERTa-large i.i.d. point** (Jin et al. report ~0.8;
   ours is distilbert 0.523). Both need a real GPU.
4. **Phase D's remaining item** — run the *prompted* decoupling method (arXiv 2505.18034;
   Mistral/Qwen → JSON graph) head-to-head, positioning us as the *trained/mechanistic* counterpart.
   Needs a local 7B.
5. **Roadmap items 2/3/5** — close learned size-extrapolation (0.8–0.9 → 1.0); perception from messy
   text rather than SVO templates; unify the RLVR causal verifier as a *reward signal* for the LM
   (today `rlvr_causal_verifier.py` is orthogonal to the LM arc).

An honest strategic note the user should weigh: the differentiated, publishable claim is the
**training-schedule mechanism** ("causal reasoning in LMs is gated by training schedule, not capacity
or perception"), because "decouple to generalize on Corr2Cause" is already occupied by prompted prior
work. §4 adds a real boundary condition to that claim: the decoupling has to terminate in a
computational module.

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
- **Multi-seed anything load-bearing.** Held-out numbers drift run-to-run on CPU.
- **Don't route around blocked egress.** If `huggingface.co` 403s through the agent proxy, report it;
  don't disable TLS or unset `HTTPS_PROXY`.

## 7. Environment

```bash
uv sync --extra torch
uv run --extra torch python examples/<script>.py
```

Knobs on the newest script: `SEEDS`, `ARMS` (`direct,joint,joint2x,decoupled,structonly`),
`LAYERS`/`EMBD`/`HEADS`, `FAST=1` (~3 min smoke).

Ruff line-length 100; `src`/`tests` are the actual quality gate, `examples/` is held looser (long
`# STATUS:` headers are house style). Last verified stack: torch 2.12, transformers 5.9.

Every script carries a `# STATUS:` header giving its act, claim, and honesty tag — keep that up to
date, and keep `CAUSAL_LLM.md`'s script map in sync when adding one.
