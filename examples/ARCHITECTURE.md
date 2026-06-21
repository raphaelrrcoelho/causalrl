# Architecture of an embedded Causal Language Model

A blueprint for a language model that reasons **beyond correlations** by embedding causal machinery in
its weights — not by calling an external engine. Every component below is here because a *measured
deficit* of a vanilla transformer demanded it; each was prototyped and validated on causalrl
ground-truth tasks (see `PHASE01_RESULTS.md`). This document is the design that ties those bricks
together.

> Thesis: causal reasoning does not emerge from correlational next-token training (we reproduced the
> Corr2Cause failure). It must be **identified, installed, and routed through** — and going beyond
> correlation requires an explicit **do()** operator and **interventional** evidence. The pieces that
> a generic decoder lacks are specific and addable.

---

## 1. The deficits → the components

| Measured finding (this repo) | Deficit in a vanilla transformer | Component added |
|---|---|---|
| Correlational training doesn't learn causal direction (Corr2Cause transfer ≈ 0.64, train≈val) | reasoning is not induced by next-token loss | **explicit causal structure** as an internal object |
| Grounded structure decodable at 100% but answer unchanged (presence ≠ mediation) | nothing forces the output to depend on the latent | **routing**: the answer is a function *only* of the structure |
| Reachability collapses on graphs larger than trained (≈ chance) | fixed-depth attention can't run a variable-length algorithm | **iterative propagation** (size-invariant) |
| Observation is capped at the Markov-equivalence ceiling; interventions resolve it | no `do()` operator; can't mutilate and re-read | **embedded do()** operator |
| Text/position encoder doesn't length-generalize the parse (edge recovery 1.0→0.78) | absolute-position sequence encoding | **relational, permutation/count-invariant perception** |
| Structure must come from data, not be handed over | no discovery front-end | **amortized discovery** (evidence → structure) |

---

## 2. The architecture

```
            raw evidence (text / data / experiments)
                          │
            ┌─────────────▼──────────────┐
            │  PERCEPTION (relational)    │   permutation/count-invariant over facts
            │  facts → embeddings         │   → size-general by construction
            └─────────────┬──────────────┘
                          │
            ┌─────────────▼──────────────┐
            │  DISCOVERY (amortized)      │   cross-attention: each variable pair
            │  evidence → soft adjacency A│   queries the evidence set → edge logits
            └─────────────┬──────────────┘   (observational ⇒ MEC; interventional ⇒ full DAG)
                          │
            ┌─────────────▼──────────────┐
            │  LATENT SCM:  A  (+ do())   │   explicit, inspectable, intervenable
            │  do(Z): zero column Z of A  │   ← the operator that goes beyond correlation
            └─────────────┬──────────────┘
                          │
            ┌─────────────▼──────────────┐
            │  REASONING (iterative)      │   R ← clamp(A + R·A), K steps
            │  directed reachability R    │   = the causal algorithm, size-invariant
            └─────────────┬──────────────┘
                          │
            ┌─────────────▼──────────────┐
            │  READOUT (do()-routed)      │   causal:  R[x,y]
            │  answer is a fn of R only   │   observational: + back-door (common-cause) term
            └─────────────┬──────────────┘
                          │
                       answer
```

**The five embedded components**

1. **Explicit causal structure `A`.** A soft adjacency matrix is materialised as an internal object,
   not left implicit in the residual stream. This is what makes mediation possible and `do()` definable.
2. **Routing.** The answer is computed *only* from quantities derived from `A` (reachability `R`). A
   feature that is merely decodable does not change behaviour (we measured this); routing makes the
   structure causally load-bearing — the mechanistic lesson of Phases 0–1.
3. **Iterative propagation.** Reachability/identification is computed by an algorithm that iterates
   (`R ← clamp(A + R·A)`), so it is invariant to the number of variables and **extrapolates to graph
   sizes never trained on** (0.55 → 0.88 on held-out size 4 vs a vanilla transformer at chance).
4. **`do()` operator.** Intervening is graph surgery on `A` (zero a column / drop the back-door term).
   The same `A` answers "are X,Y correlated?" (with the back-door) and "does X cause Y?" (directed only)
   — distinguishing correlation from causation, including confounded pairs, where a correlation-only
   reasoner is wrong by definition.
5. **Relational perception + amortized discovery.** Evidence is read as a *set of relational facts* and
   mapped to `A` by a permutation/count-invariant encoder (cross-attention from variable-pair queries to
   the fact set). This is the AVICI/CSIvA-style amortized-discovery idea, embedded. Observational
   evidence yields the structure only up to the Markov-equivalence class; interventional evidence yields
   the full DAG — the identifiability ceiling shows up *inside* the model.

---

## 3. What each prototype validated

| File | What it shows |
|---|---|
| `causal_reasoning_scaffold.py` | the causal structure is the missing ingredient (struct-only 0.82 vs direct 0.66) |
| `causal_grounding_install.py` | presence ≠ mediation; routing through the structure (pipeline) → 0.81 |
| `causal_core_architecture.py` | iterative propagation → size-generalization (held-out 0.88) |
| `causal_beyond_correlation.py` | interventions break the MEC ceiling (0.82 → 0.99) |
| `causal_core_do.py` | embedded `do()`: one core answers obs + int; confounded 0.94–1.0 held-out |
| `causal_core_perception.py` | relational perception → fully size-robust (all 1.0 on held-out 4/5) |
| `causal_core_discovery.py` | amortized discovery: evidence → structure → reasoning, end-to-end |

---

## 4. What is solved vs open

**Solved (embedded, validated at small scale):** the *causal computation* — explicit structure,
routing, size-general reasoning, the `do()` operator, relational perception, and amortized discovery
from idealised evidence. Together these are a working embedded causal core that reasons beyond
correlation and generalizes in graph size.

**Open (the path to a true causal LM):**

1. **Discovery from raw data / natural text**, not idealised CI/intervention facts — the hardest real
   step, and information-theoretically capped without interventions (so the model must also *seek*
   interventions: active causal discovery).
2. **Coupling to a language model.** The core is a reasoning module; a causal LM needs the LM to (a)
   extract the relational facts from natural language and (b) verbalise the core's answers — a
   bidirectional NL ↔ latent-SCM interface, trained jointly.
3. **Scale & faithfulness.** Whether the core's exact, size-general behaviour survives being learned
   (rather than hand-structured) at frontier scale, and whether the discovered `A` stays faithful on
   messy, real distributions — the central empirical risk.

**One-line summary.** A vanilla decoder lacks an explicit causal structure, output routing through it,
an iterative reasoning core, a `do()` operator, and a discovery front-end. Add those five — as we have,
in miniature — and you get causal reasoning that is mediated, goes beyond correlation, and generalizes
in size. Turning that core into a *large* causal language model is the work of coupling it to language
and learning discovery from raw data at scale.
