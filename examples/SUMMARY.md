# Toward a causal language model — summary of the arc

A consolidated, honest account of the experiments in this folder building toward a language model that
reasons **beyond correlations** by embedding causal machinery. `PHASE01_RESULTS.md` is the detailed,
reproducible log; `ARCHITECTURE.md` is the design; this file is the narrative and the honest status.

All experiments are CPU-sized, on synthetic tasks generated from causalrl (so every label is ground
truth), with multi-seed robustness on the load-bearing claims. This is a controlled proof of concept,
not a frontier-scale result.

## The thesis

Causal reasoning does **not** emerge from correlational next-token training; it must be **identified,
installed, and routed through**, and going *beyond correlation* requires an explicit **do()** operator
and **interventional** evidence — which an agent must learn to **choose**. A vanilla transformer lacks
the parts; they are specific and addable.

## The arc, in one line each

**Diagnosis**
- Correlational training fails to learn causal direction — we reproduced the Corr2Cause phenomenon in a
  controlled setting (train ≈ val ≈ 0.65, below a trivial correlation heuristic).
- The causal structure is the *missing ingredient* and is usable: given it, accuracy jumps to the
  Markov-equivalence ceiling (struct-only 0.82 vs direct 0.66).
- **Presence ≠ mediation**: grounding the structure in the representation (decodable at 100%) does not
  change the answer; routing the computation *through* it does (pipeline 0.66 → 0.81).
- Interventions break the observational ceiling: same queries, 0.82 (obs) → 0.99 (interventional).

**The embedded causal core** (the parts a vanilla decoder lacks; each tied to a finding above)
- explicit adjacency · routing · **iterative propagation** (size-general: held-out 0.55→0.88) ·
  **do() operator** (correlation vs causation, confounded held-out ~0.9) · **relational perception**
  (size-robust, all 1.0 on held-out 4/5) · **amortized discovery** (evidence → structure).

**Beyond correlation, completed**
- do() embedded; and **active discovery** — *choosing* interventions reaches ~0.99 in 2 experiments
  where random needs ~5 (observation floor 0.81).

**Coupling to language**
- A tiny bespoke interface, then the real step: a **hybrid GPT-2 + embedded core** that beats vanilla
  GPT-2 decisively (confounded 0.99 vs 0.21 in-dist; 0.87 vs 0.24 held-out).

**Discovery from raw data**
- The model infers structure from **raw SCM samples** (no oracle facts). Judged against trivial
  baselines (non-degenerate): interventional data → edge 0.92, answer 0.87, confounded 0.85;
  observational → reads dependence (answer 0.72 > majority) but cannot orient → confounded 0.28 (the
  MEC limit, now *from raw data*). It needed a non-tiny encoder; a first 1K-param attempt collapsed.

## Robustness (multi-seed)

The gaps hold across seeds at full training budget: hybrid confounded 1.000 ± 0.000 (in-dist) /
0.914 ± 0.071 (held-out) vs vanilla ~0.15; active 0.991 ± 0.002 vs random 0.923 ± 0.003; interventional
discovery 0.859 ± 0.222 vs observational 0.113 ± 0.097. Honest caveat surfaced *by* multi-seed:
under-training makes the embedded-core models **seed-fragile** (a reduced sweep was bimodal
`[1,0,1]`); they are robust only with adequate budget.

## What is solved vs open

**Solved (small scale, validated, multi-seed):** the causal *computation* — explicit structure,
routing, size-general iterative reasoning, the do() operator, relational perception, amortized discovery
(from facts and from raw samples), active discovery, and coupling to a real GPT-2 as a hybrid.

**Open (the path to a *large* causal LM):**
1. **Real natural language** — rich phrasing, multi-sentence, coreference, implicit causality — and a
   **generative verbaliser** (here: simple SVO prose, parsed questions, yes/no readout).
2. **Discovery from messy real data/text** at the perception front-end (the recurring bottleneck; the
   hybrid's residual held-out drop is its GPT-2 perception, not the core's reasoning).
3. **The "pure" path** — whether a real LM can *internalise* the causal computation in its own weights
   (vs the hybrid module). Our diagnostics suggest this is hard (emergence fails, presence ≠ mediation);
   it is the next thing to test with robustness.
4. **Scale & faithfulness** — whether the exact, size-general behaviour survives being *learned* at
   frontier scale on real distributions. This — with a real pretrained LM and a real benchmark
   (Corr2Cause) — is the only test that would show whether this is a genuine advance; it needs a
   different environment (network / GPU) than this one.

## Honest bottom line

The thesis is demonstrated **end to end at small scale and with multi-seed robustness**: embedding
causal structure + routing + do() + (active) discovery into / alongside a real LM yields reasoning that
goes beyond correlation and generalizes in graph size, where a vanilla LM fails. It is a strong,
falsifiable proof of concept — **not** a frontier result. The wall between the two is scale, real
language, and real-data discovery, not the causal mechanism, which is now a working embedded
architecture.
