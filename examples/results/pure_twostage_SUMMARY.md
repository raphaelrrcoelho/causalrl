# `causal_pure_twostage.py` — the missing 2x2 cell (evidence index)

All numbers seed 0, `causal_hybrid_lm` substrate (word-level prose, train sizes 2/3, size 4 held
out). `confounded` is **all-negative** (constant-"no" scores 1.000), so it is only meaningful read
together with the balanced `cause` query (constant answer ~0.500).

## The question

Every "a real LM does not internalise causal reasoning" result on this branch was measured under the
**joint** schedule, and the fix that worked routed the answer through an **external GNN**:

|                     | joint schedule            | decoupled schedule          |
|---------------------|---------------------------|-----------------------------|
| external GNN module | 0.43 (`hybrid_learned`)   | **1.000** (`hybrid_twostage`) |
| pure LM weights     | 0.37-0.53 / CoT 0.655     | **this script**             |

## Result — a localized negative

| probe | number | rules out |
|---|---|---|
| self-generated edge F1 | **0.940** | perception |
| 2x answer supervision | ceiling **0.596 -> 0.596** | undertraining / budget |
| STRUCTONLY (prose deleted, true graph given) | cause **0.723**, conf **0.186** | the prose shortcut *and* the contradictory-prose design |

Same structure->answer function, same data and seed:

| system | params | cause s3 | cause s4 | conf s3 | conf s4 | final train loss |
|---|---|---|---|---|---|---|
| GNN reasoner on clean structure | **11,857** | **1.000** | 0.952 | **1.000** | 0.893 | - |
| GPT-2 (4L/128d), same structure as tokens | **809,344** | 0.723 | 0.579 | 0.186 | 0.249 | 0.365 |
| GPT-2 (8L/192d), capacity control | **3,583,296** | 0.745 | 0.563 | 0.088 | 0.020 | **0.320** |

A 68x parameter advantage, losing badly -- and **capacity is not the explanation**: 4.4x more
parameters bought +0.02 on `cause` and got *worse* on everything else, while the training loss
*fell* (0.365 -> 0.320). The bigger model memorizes harder without learning the algorithm. At 302x
the GNN's parameter count it still cannot do reachability. The wall is **multi-hop reachability over a serialized
graph** in transformer weights — the same thing `causal_graph_transformer.py` found from the other
side (shortcuts, not d-separation). This is why decoupling transfers to an external module but not
into the weights.

## Method notes (kept — the first nearly produced a false positive)

* **Step-matching was the wrong fairness unit.** JOINT carries both losses on every item, so equal
  gradient steps handed DECOUPLED half the answer supervision. Fixed to epochs *per objective*.
  Discarded run: `pure_twostage_stepmatched_s0.log`.
* **JOINT's teacher-forced ceiling (0.871) does not measure graph-reading.** JOINT only ever sees the
  TRUE graph, which agrees with the prose, and DIRECT already gets 0.795 from prose alone. This is
  why STRUCTONLY (prose deleted) is the load-bearing probe, not the ceiling.

## Logs

| file | what |
|---|---|
| `pure_twostage_stepmatched_s0.log` | first run, step-matched (discarded, kept for lineage) |
| `pure_twostage_decoupled_fair_s0.log` | DECOUPLED with full per-objective supervision — refutes undertraining |
| `pure_twostage_structonly_s0.log` | STRUCTONLY (4L/128d) — isolates the reasoning step |
| `pure_twostage_structonly_8L192d_s0.log` | STRUCTONLY at 4.4x capacity — rules out "too small" |

## Owed

Multi-seed replication (all numbers here are seed 0). The capacity control is **done** and closes
"architectural, not capacity": `pure_twostage_structonly_8L192d_s0.log`.
