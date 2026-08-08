# `causal_pure_twostage.py` — the missing 2x2 cell (evidence index)

Multi-seed (SEEDS=0,1,2), `causal_hybrid_lm` substrate (word-level prose, train sizes 2/3, size 4
held out). `confounded` is **all-negative** (constant-"no" scores 1.000), so it is only meaningful
read together with the balanced `cause` query (constant answer ~0.500).

## The question

Every "a real LM does not internalise causal reasoning" result on this branch was measured under the
**joint** schedule, and the fix that worked routed the answer through an **external GNN**:

|                     | joint schedule            | decoupled schedule          |
|---------------------|---------------------------|-----------------------------|
| external GNN module | 0.43 (`hybrid_learned`)   | **1.000** (`hybrid_twostage`) |
| pure LM weights     | 0.37-0.53 / CoT 0.655     | **this script**             |

## Result — a localized negative, now replicated over 3 seeds

| probe | mean ± std (3 seeds) | per-seed [s0, s1, s2] | rules out |
|---|---|---|---|
| self-generated edge F1 (s3) | **0.942 ± 0.023** | [0.917, 0.962, 0.947] | perception |
| teacher-forced ceiling on `cause` (s3) | **0.594 ± 0.012** | [0.591, 0.583, 0.607] | undertraining (2x supervision moved it 0.596→0.596 at s0) |
| STRUCTONLY `cause` s3 (prose deleted, true graph given) | **0.731 ± 0.094** | [0.828, 0.725, 0.640] | the prose shortcut |
| STRUCTONLY `conf` s3 | **0.190 ± 0.150** | [0.280, 0.272, 0.017] | (read beside `cause`) |
| STRUCTONLY 8L/192d `cause` s3 (4.4x params) | **0.753 ± 0.018** | [0.745, 0.773, 0.740] | capacity |
| STRUCTONLY 8L/192d `conf` s3 | **0.196 ± 0.094** | [0.088, 0.261, 0.239] | capacity |

Same structure→answer function, same data:

| system | params | cause s3 | conf s3 |
|---|---|---|---|
| GNN reasoner on clean structure | **11,857** | **1.000** | **1.000** |
| GPT-2 (4L/128d), same structure as tokens | **809,344** | 0.731 ± 0.094 | 0.190 ± 0.150 |
| GPT-2 (8L/192d), capacity control | **3,583,296** | 0.753 ± 0.018 | 0.196 ± 0.094 |

(The GNN row is the seed-0 stage-B reasoner of `causal_hybrid_twostage.py`, whose composed system is
itself multi-seed: 1.000 ± 0.000 confounded in-dist.) Every individual run — six trainings across two
capacities and three seeds — stays far below the 11.9K-parameter GNN on both queries: the worst gap
on `cause` is 0.17 (best run, 0.828), the typical gap ~0.27, and `conf` never exceeds 0.28 vs 1.000.
Capacity moves `cause` by ~0.02 (within seed noise) and leaves `conf` broken. The wall is **multi-hop
reachability over a serialized graph** in transformer weights — the same thing
`causal_graph_transformer.py` found from the other side (shortcuts, not d-separation). This is why
decoupling transfers to an external module but not into the weights.

## Method notes (kept — the first two nearly produced a false positive each)

* **Step-matching was the wrong fairness unit.** JOINT carries both losses on every item, so equal
  gradient steps handed DECOUPLED half the answer supervision. Fixed to epochs *per objective*.
  Discarded run: `pure_twostage_stepmatched_s0.log`.
* **JOINT's teacher-forced ceiling (0.871) does not measure graph-reading.** JOINT only ever sees the
  TRUE graph, which agrees with the prose, and DIRECT already gets 0.795 from prose alone. This is
  why STRUCTONLY (prose deleted) is the load-bearing probe, not the ceiling.
* **Same-seed reruns drift on CPU.** Seed 0 rerun in a fresh process: STRUCTONLY `cause` s3 moved
  0.723 → 0.828 (thread-order nondeterminism). Single-run numbers on this substrate are a band, not
  a point — which is exactly why this file reports mean ± std. One casualty: the earlier "STRUCTONLY
  (0.723) sits below the scaffold's struct-only 0.818" sub-claim does not survive (the band touches
  0.83); the claim that survives is the gap to the GNN's 1.000, which every run shows with margin.
  A second casualty: the seed-0 "bigger model fits train better while generalizing worse" loss
  nuance (0.365→0.320) — multi-seed the loss ranges overlap (4L [0.287, 0.365, 0.405] vs 8L
  [0.320, 0.345, 0.352]); the accuracy conclusion (capacity doesn't fix it) is what replicates.

## Logs

| file | what |
|---|---|
| `pure_twostage_stepmatched_s0.log` | first run, step-matched (discarded, kept for lineage) |
| `pure_twostage_decoupled_fair_s0.log` | DECOUPLED with full per-objective supervision — refutes undertraining |
| `pure_twostage_structonly_s0.log` | STRUCTONLY (4L/128d) seed 0, original |
| `pure_twostage_multiseed_s{0,1,2}.log` | the multi-seed wave: DECOUPLED + STRUCTONLY per seed (s0 is a same-seed rerun — see drift note) |
| `pure_twostage_structonly_8L192d_s{0,1,2}.log` | STRUCTONLY at 4.4x capacity, 3 seeds — rules out "too small" |
