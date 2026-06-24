# Phases 0-1 results — Causal Grounding (DAS + interchange + IIT)

Runnable demonstration for `FRONTIER_PROPOSAL_v2.md`. Code: `causal_grounding_das_iit.py`.
A 0.84M-param GPT-2 is trained from scratch on a causalrl see/do task whose **true causal variable —
the observational-vs-interventional regime — is known**, so every mechanistic claim is checked against
ground truth. CPU-sized, didactic; not a performance claim. Numbers below are from a single seed
(`torch.manual_seed(0)`); they are stable across reruns because data generation and training are seeded.

## Setup

Confounded SCM (hidden severity `U` → drug `X`, recovery `Y`); see and do genuinely disagree:

```
causalrl SCM truth:  P(rec|see drug)=0.860   P(rec|do drug)=0.650   ground-truth see/do gap=+0.209
```

The model reads each scenario as text tagged `<see>`/`<do>` and we read `P(recover)` off a single
reserved outcome token. "Domain" = surface vocabulary: train on A (+ a held-in domain C used only to
make the latent invariant); **domain B is held out as the OOD test** (same SCM, unseen words).

## Headline numbers

| see/do gap in P(recover) | in-dist (A) | OOD (B) | reading |
|---|---|---|---|
| **SCM truth** | +0.209 | +0.209 | the target |
| **base model** | +0.239 | **+0.019** | learns the regime in-dist; **collapses OOD** (causal parrot) |
| **Phase 1a — repair by intervention** | — | **+0.285** | intervening on the located direction restores OOD |
| **Phase 1b — IIT-installed latent** | +0.213 | **+0.190** | grounding makes the regime domain-invariant |

## Phase 0 — locate + diagnose (model frozen)

```
k=1:  IE(learned)=+1.00   IE(random)=+0.00   gap base=+0.239 -> ablated=+0.000   attribution=100%
k=4:  IE(learned)=+1.00   IE(random)=+0.01   gap base=+0.239 -> ablated=-0.000   attribution=100%
```

- **The regime is a single linear direction.** A 1-D subspace found by Distributed Alignment Search
  carries the *entire* see/do behaviour: swapping only that direction between a see- and a do-prompt
  transfers 100% of the behavioural gap (Interchange Effect IE≈1.0), while a random direction transfers
  nothing (≈0.0) — a causal-mediation diagnosis that the gap is mediated by *this* direction.
- **It is causally load-bearing.** Mean-ablating that one direction collapses the +0.239 see/do gap to
  0.000 → 100% attribution.
- **The OOD failure is diagnosed:** the same model that shows +0.239 in-dist shows only +0.019 on
  domain B. It *has* the regime feature but does not compute it from unseen surface words.

## Phase 1a — repair OOD by intervention (inference-time, no retraining)

```
OOD (domain B) see/do gap:  base=+0.019  ->  repaired=+0.285   (truth +0.209;  control gap=+0.004 ~ 0)
```

Overwriting *only* the regime coordinate of domain-B prompts with its in-distribution value (interchange
along the Phase-0 direction) **restores the see/do gap** from +0.019 to +0.285 — i.e. the OOD failure is
a *regime mis-encoding*, repairable by intervening on the identified variable. Control: injecting the
*same* regime into both see and do prompts collapses the gap to ~0, confirming the regime is what does
the work. This is the proposal's thesis in miniature: identify the internal causal variable, intervene
on it to overcome the limitation.

## Phase 1b — install a domain-invariant regime latent (IIT)

```
installed 1-D regime subspace:  IE=+1.03   gap intact=+0.213 -> ablated=-0.006   attribution=103%
iit  in-dist (domain A)         gap = +0.213
iit  OOD     (domain B)         gap = +0.190     (was +0.019 for the base model; B never trained on)
```

Interchange Intervention Training **on the located direction** (frozen `D`, model trained around it,
fixed SCM-truth targets, cross-domain swaps A↔C) installs the regime as a domain-invariant 1-D latent:
the OOD gap rises from +0.019 to **+0.190**, nearly matching in-dist (+0.213) and truth (+0.209), with
the behaviour still 100%-attributable to the single installed direction.

### A real negative result that shaped the method

Naive joint IIT (learning `D` from scratch together with the model) **collapsed and sign-flipped** the
gap. Diagnosis: with a random `D` the interchange swap transmits nothing, so the interchange loss
degenerates into a second behavioural anchor with *swapped* targets and overpowers the correct one.
Fix, used above: **warm-start `D` to the Phase-0 direction and freeze it** so the swap transmits the
regime from step one. (Also fixed in Phase 0: an off-by-one readout — the outcome token must be the
*immediate* successor of the prompt — which a naive behavioural test would have silently hidden, but
the interchange check surfaced.)

## How to reproduce

```
uv run --extra torch python examples/causal_grounding_das_iit.py     # full run, retrains the base
CG_CACHE=1 uv run --extra torch python examples/causal_grounding_das_iit.py   # cache base for fast iteration
```

## What this is and is not

- **Is:** a controlled existence proof that an LLM's observe-vs-intervene distinction is a linear,
  localizable, causally load-bearing latent that can be (a) located, (b) repaired OOD by intervention,
  and (c) installed as a domain-invariant feature — each step verified against causalrl ground truth.
- **Is not:** a frontier-scale or natural-language result. The central open risk (proposal Phase 3) is
  the synthetic→natural and small→large transfer; this demonstrates the method, not its transfer.

---

# Phase 2 results — grounding *multiple* causal primitives

Code: `causal_grounding_phase2.py` (reuses the Phase 0-1 machinery and the same cached base). The task
now has **two** causal variables that both move P(recover): the **regime** R∈{see,do} and the
**treatment** T∈{drug,nodrug}. Four-cell SCM truth: see/drug 0.86, see/nodrug 0.14, do/drug 0.65,
do/nodrug 0.35 (R-gap +0.21, T-gap +0.30). The question: are the two primitives grounded as
*separate, independently-controllable* latents?

### 1. Naive per-variable DAS entangles the two

Locating each variable on its own gives overlapping directions and off-diagonal leakage:

```
|cos(D_R, D_T)| = 0.749
                swap regime     swap treatment
  patch D_R :     +1.00            +0.66        <- swapping regime also moves treatment
  patch D_T :     +0.71            +1.00
  composition MAE = 0.131
```

### 2. Joint orthonormal DAS disentangles the directions

Locating both *jointly* in an orthonormal 2-frame, optimised so the interchange matrix is the
identity, cuts the off-diagonal leakage ~4× (at a modest cost to the on-diagonal):

```
|cos(D_R, D_T)| = 0.000
                swap regime     swap treatment
  patch D_R :     +0.78            +0.16        <- off-diagonal 0.66 -> 0.16
  patch D_T :     +0.22            +0.87        <- off-diagonal 0.71 -> 0.22
```

### 3. Composition exposes a real limit — variable interaction

Setting (R,T) from a single prompt by intervening on both subspaces reconstructs the low-interaction
cells but **fails on the high-interaction cell**:

```
  cell          composed   truth    |err|
  see_drug      0.908     0.860    0.048
  do_drug       0.743     0.650    0.092
  see_nodrug    0.672     0.141    0.531   <- the high-interaction cell, badly off
  do_nodrug     0.372     0.350    0.022
  MAE = 0.173   (naive DAS: 0.131)
```

The reason is structural, not a tuning artifact: the **treatment's effect is regime-dependent** —
under see the drug→nodrug gap is 0.72, under do it is 0.30. A *single linear direction* per variable,
calibrated at one regime, cannot carry an effect whose magnitude depends on the other variable.

**Honest finding.** Disentangling the *directions* succeeds (off-diagonal cut ~4×), but it is
**necessary, not sufficient**: when causal variables interact, the interaction itself must be grounded
(a higher-dim subspace, or installed via IIT so the downstream computes the interaction from clean
disentangled carriers — the constructive Phase 3 step). This is exactly the kind of boundary a
ground-truth (causalrl) testbed is meant to surface.

### Reproduce

```
CG_CACHE=1 uv run --extra torch python examples/causal_grounding_phase2.py
```

---

# Phase 3 results — install disentangled carriers *and* the grounded interaction

Code: `causal_grounding_phase3.py`. The constructive answer to Phase 2's limit: using two frozen
orthonormal carriers (e_R for regime, e_T for treatment), run IIT so the model (a) routes each
variable through its carrier (disentanglement specs) and (b) computes the **interaction downstream**
(composition specs: setting (R,T) via the carriers must hit the SCM-truth cell, including see/nodrug).
All targets are fixed SCM truth.

### Before → after (composition error, lower is better)

| | off-diagonal IE | composition MAE | high-interaction cell see/nodrug |
|---|---|---|---|
| base + Phase-2 carriers, in-dist A | 0.19 | 0.173 | err 0.531 |
| **after Phase 3 IIT, in-dist A** | **0.12** | **0.104** | **err 0.228** |
| base + Phase-2 carriers, OOD B | 0.32 | 0.150 | err 0.311 |
| **after Phase 3 IIT, OOD B** | **—** | **0.104** | **err 0.113** |

### Honest read — a *partial* success

IIT **partially grounds the interaction**: composition MAE drops ~40% (0.173 → 0.104), the
high-interaction cell see/nodrug roughly halves (0.531 → 0.228 in-dist; 0.311 → 0.113 OOD), and the
improvement **transfers to the held-out OOD domain B**. So the downstream did learn to compute part of
the interaction from the two clean carriers — something Phase 2's frozen-base linear control could not.

It does **not** fully close, and the script says so: see/drug regresses (0.86 → ~0.72) and training is
delicate (a late-epoch instability spike). Fully grounding an interaction with **frozen 1-D** carriers
in a 0.84M model is hard — the carriers have no room to also encode the interaction, and the single
downstream block must do all the work. The clear next step (Phase 3+/scaling) is **co-trained or
higher-dimensional carriers** so the representation has capacity for the interaction itself.

This is the honest shape of the result: grounding *non-interacting* primitives (Phases 0-1) is clean
and strong; grounding *interacting* primitives is partially achievable and the residual difficulty is
real — exactly the kind of boundary a ground-truth (causalrl) testbed exists to expose.

### Reproduce

```
CG_CACHE=1 uv run --extra torch python examples/causal_grounding_phase3.py
```

---

# Phase 3b — co-training the carriers (a negative result)

Code: `causal_grounding_phase3b.py`. Hypothesis: the partial Phase 3 was limited because the carriers
were *frozen*; **co-training** them (warm-started from the Phase-2 frame, kept orthonormal) should give
the optimiser freedom to pick axes that disentangle tightly *and* support the interaction, closing
composition. **The hypothesis was wrong.**

```
epoch 1..6:  losses freeze (beh 0.143, disent 0.514, comp 0.692) — stuck after epoch 1
after:  every cell composes to ~0.604 (constant)   off-diagonal IE = 0.00   composition MAE = 0.255
```

Co-training **collapses** the model into a degenerate flat minimum: it outputs a near-constant
P(rec)≈0.60 for every cell, the carriers carry nothing (IE=0), and composition MAE *rises* to 0.255 —
worse than frozen Phase 3 (0.104) and even worse than the base (0.173). The same OOD-B numbers (0.255)
confirm it is a genuine collapse, not overfitting.

**Why this matters.** This is the *same* degenerate-collapse failure mode that the naive joint DAS/IIT
hit in Phases 1-2, and the reason Phases 1b/3 **froze** their carriers. Making the carriers trainable
reintroduces it. So: **freezing the located direction is load-bearing**, not a convenience — it is what
keeps interchange training out of the trivial "make the output constant, carry nothing" basin. The
frozen-carrier Phase 3 (partial, MAE 0.104) remains the best result; closing the interaction needs a
different lever that preserves freezing (higher-dim *frozen* carriers, training only the downstream
block, or scale) — candidates this run did not test.

### Reproduce

```
CG_CACHE=1 uv run --extra torch python examples/causal_grounding_phase3b.py
```

---

# Transfer step 1 — a Corr2Cause-style task, generated from causalrl

Code: `causal_transfer_corr2cause.py`. The central open risk is transfer to *natural* causal reasoning.
The real Corr2Cause benchmark needs a large pretrained LLM + a dataset download (both out of scope in
this offline container), so this is the honest first step: the **same skill** (infer a causal relation
from correlation / conditional-independence facts) rendered in language, but **generated from causalrl**
so every label is ground truth — random DAGs → the d-separation facts in words → "does X cause Y?"
(truth = Y is a descendant of X). Labels are balanced 50/50; we score against a correlation heuristic
(predict "cause" iff X,Y marginally correlated) and a brute-force **Markov-equivalence-class oracle**
(the best any premise-based reasoner can do — direction is only identifiable up to the MEC).

### Result (2.78M GPT-2, trained from scratch, converged)

| accuracy | model | majority | corr-heuristic | MEC-ceiling |
|---|---|---|---|---|
| **train (3 vars, seen)** | 0.682 | 0.510 | 0.799 | 0.821 |
| **in-dist (3 vars)** | 0.641 | 0.500 | 0.787 | 0.838 |
| **OOD (4 vars, unseen)** | 0.572 | 0.500 | 0.732 | 0.801 |

### Honest read — a negative transfer result that *confirms the thesis*

The model **does not learn the Corr2Cause skill**. It plateaus at ~0.65–0.68 — *below the trivial
correlation heuristic* (~0.79) and well below the MEC ceiling (~0.82) — and degrades OOD (~0.57). The
decisive diagnostic is the train row: **train accuracy ≈ val accuracy** (0.68 ≈ 0.64), so this is *not*
overfitting. Under standard LM training the model cannot even **fit** the conditional-independence →
direction mapping on seen data, and a bigger model + more data + more epochs did not move it (the 0.86M
4-layer model gave the same ~0.66).

This **reproduces the real Corr2Cause finding** (LLMs barely beat trivial baselines at inferring
causation from correlation) — now in a fully controlled, ground-truth setting where the data is clean
and provably sufficient (MEC ceiling 0.82). So the failure is the model's *reasoning*, not the data.

And that is exactly the point of the whole program: **causal direction does not emerge from
correlational next-token training — it has to be identified and installed** (Phases 0–3), not hoped for.
The negative transfer result and the positive grounding results are two sides of the same thesis.

### Reproduce

```
uv run --extra torch python examples/causal_transfer_corr2cause.py
```

---

# Does causal reasoning improve the model's reasoning?  (the on-thesis result)

Code: `causal_reasoning_scaffold.py`. The most direct test of the program's claim. On the same
Corr2Cause-style task where direct answering is stuck at ~0.66, three conditions (same tiny model):

1. **DIRECT** — premises → answer (extract causal direction from raw correlations).
2. **STRUCT-ONLY** — the **CPDAG alone** → answer (reason over a *given* causal structure; no premises,
   so structure-reasoning is the only path — avoids the redundancy of showing both).
3. **CoT** — premises → *model derives the CPDAG* → answer (derive the structure, then use it).

The CPDAG (skeleton + Markov-equivalence-invariant orientations) is the exact identifiable causal
structure, computed from causalrl ground truth.

### Result

| accuracy | direct | **struct-only** | CoT (self-derived) | corr-heuristic | MEC-ceiling |
|---|---|---|---|---|---|
| **in-dist (3 vars)** | 0.659 | **0.818** | 0.655 | 0.787 | 0.838 |
| OOD (4 vars) | 0.573 | 0.565 | 0.138 | 0.732 | 0.801 |

### Read — the clearest "causal reasoning improves reasoning" signal in this work

In-distribution, **handing the model the causal structure lifts accuracy from 0.659 to 0.818 — to the
information-theoretic MEC ceiling (0.838)**, and past the correlation heuristic (0.787). So:

- the model **can reason over a causal model** — near-optimally;
- it **cannot extract** that model from raw correlations (direct stuck at 0.66);
- it **cannot derive** it itself either (CoT ≈ direct).

That triangulates the thesis precisely: **the causal structure is the missing ingredient, and the
bottleneck is extraction/grounding — exactly what Phases 0–3 target.** "Causal reasoning improves
reasoning" here is not a slogan: providing a causal model moves a stuck reasoner to the ceiling.

**Honest caveat.** It does **not** generalize OOD: on 4-variable graphs struct-only (0.565) collapses
back to direct, and CoT generation breaks (0.138). The *structure-reasoning skill learned on 3-var
graphs is not size-general*, and self-derivation fails. So this is a controlled existence proof — the
causal model is usable and is what's missing — not a scalable solution. Closing the extraction gap and
making structure-reasoning size-general is the open work.

### Reproduce

```
uv run --extra torch python examples/causal_reasoning_scaffold.py
```

---

# Installing the causal structure + extract-then-reason pipeline (the positive result)

Code: `causal_grounding_install.py`. The scaffold localized the bottleneck (reason-over-structure works,
extract-from-correlations doesn't). This tests the program's core move — *install* the causal structure
and route the answer through it.

| in-dist (3 vars) | plain-direct | grounded-direct | **pipeline** | corr | MEC-ceiling |
|---|---|---|---|---|---|
| answer accuracy | 0.659 | 0.654 | **0.812** | 0.787 | 0.838 |
| aux CPDAG accuracy | — | **1.000 / pair** | — | — | — |

Two findings, both important:

1. **Representational grounding alone is necessary but not sufficient.** Giving the *same hidden state
   that produces the answer* a dense auxiliary target (predict the CPDAG) makes the representation
   encode the structure **perfectly (aux accuracy 1.000/pair)** — yet the answer is **unchanged**
   (0.654 ≈ plain 0.659). The structure is *present* but the answer head does not *route through it*:
   **presence ≠ mediation** — exactly the Phase-0 lesson (a feature must be causally load-bearing, not
   just decodable).

2. **Routing the answer through the structure closes the gap.** The extract-then-reason **pipeline**
   (the grounded model extracts the CPDAG — which it does at 100% — then a struct model reasons over
   it) lifts accuracy **0.659 → 0.812**, to the MEC ceiling (0.838) and matching struct-only.

So, on this task, **explicit causal reasoning — ground the causal structure *and* route the
computation through it — turns a stuck reasoner (0.66) into a near-optimal one (0.81).** And the
surprise from the aux head (1.000): *extraction was never the wall* — with decomposed per-edge
supervision the model extracts the full structure; the wall was **mediation/routing**.

Caveat: in-distribution (3-var). The aux extractor is size-specific, so the pipeline is not yet
size-general — which motivates the curriculum experiment below.

### Reproduce

```
uv run --extra torch python examples/causal_grounding_install.py
```

---

# (2) Can a size curriculum make structure-reasoning size-general?  (negative)

Code: `causal_reasoning_curriculum.py`. The pipeline above works in-distribution but not across graph
sizes. This tests whether training the reason-over-structure model on a **curriculum of sizes** makes
it extrapolate. Two struct-only models (CPDAG → answer, size-agnostic format), both evaluated on
**4-variable graphs neither saw** (held-out, larger). Variable names are randomized over A–E so every
symbol appears at every size — isolating *pure size-extrapolation* (no new-token confound).

| answer accuracy | single (sizes {3}) | curric (sizes {2,3}) | corr | MEC-ceiling |
|---|---|---|---|---|
| size 3 (in-dist) | 0.827 | 0.721 | 0.811 | 0.829 |
| **size 4 (held-out)** | **0.559** | **0.551** | 0.766 | 0.855 |

**It does not work.** On the held-out size 4, both models sit at ~chance (0.55), far below the MEC
ceiling (0.855) and below even the correlation heuristic — and the curriculum is no better than the
single-size model. Training on multiple sizes did **not** induce a reasoning procedure that
extrapolates to a larger unseen graph. (An earlier run with `LETTERS[:k]` gave the same collapse but
was confounded by a never-seen variable token at test; randomizing names removes that confound and the
failure persists — so it is genuine **size/length-extrapolation failure**, the well-known wall for
algorithmic generalization in transformers, not symbol novelty.)

**Where this leaves the arc.** The method works *at a fixed scale* — ground the causal structure and
route through it, and a stuck reasoner reaches the ceiling (the install/pipeline result). It does
**not** yet generalize in size. So the honest boundary is exactly the one flagged from the start:
**scale/extrapolation, not the in-distribution mechanism, is the wall.** Closing it likely needs
ideas beyond a plain curriculum — algorithmic/length-generalization techniques (e.g. positional
schemes, recurrence/scratchpad iteration, or explicit graph-traversal inductive biases).

### Reproduce

```
uv run --extra torch python examples/causal_reasoning_curriculum.py
```

---

# Beyond correlations: interventions break the observational (MEC) ceiling

Code: `causal_beyond_correlation.py`. Everything above was observational, capped at the MEC ceiling.
The defining move of a *causal* model is using interventions. Same queries ("does X cause Y?"), two
evidence types, same tiny model: OBSERVATIONAL (the CPDAG, only MEC-invariant orientations) vs
INTERVENTIONAL (the fully-oriented DAG that do()-experiments reveal).

| answer accuracy | observational | interventional |
|---|---|---|
| all queries | 0.817 | **0.994** |
| **MEC-ambiguous (observation cannot decide)** | 0.703 | **1.000** |
| MEC-determined | 1.000 | 0.984 |
| ceiling | MEC = 0.835 | full orientation = 1.000 |

The observational model sits exactly at its MEC ceiling, and on the **MEC-ambiguous** queries
(direction unidentifiable from correlation) it cannot break ~0.70 — *correlation literally cannot
orient those edges*. The interventional model resolves them (1.000), exceeding the observational
ceiling. **This is "beyond correlations" made concrete**, and the architectural mandate for a causal
LM: it must ingest/seek interventional evidence, not reason from correlation alone.

### Reproduce

```
uv run --extra torch python examples/causal_beyond_correlation.py
```

---

# An embedded causal core — the first architectural brick (size-generalization solved)

Code: `causal_core_architecture.py`. The size wall (curriculum experiment: vanilla transformer ~chance
on held-out larger graphs) is **architectural**. This embeds the causal algorithm in the weights:

```
text → [transformer encoder] → per-variable reps → [edge MLP] → explicit soft adjacency A
     → [K-step iterative propagation: R ← clamp(A + R·A)] → reachability → answer (routed through R)
```

The answer is a function only of the propagated reachability `R` (so the structure is *mediating*, not
merely present), and reachability is computed by an **iterative algorithm that is size-invariant**.
Trained on 2- and 3-variable graphs (random variable names), tested on **4- and 5-variable graphs held
out by size**. 79K parameters.

| graph size | answer acc | edge recovery | in training? |
|---|---|---|---|
| 2 | 1.000 | 1.000 | trained |
| 3 | 1.000 | 1.000 | trained |
| **4** | **0.878** | 0.893 | **held-out** |
| **5** | **0.729** | 0.776 | **held-out** |
| vanilla transformer (struct-only) | ~0.55 @ size 4 | — | held-out (≈ chance) |

**The embedded core size-generalizes where the vanilla transformer collapsed** — 0.55 → 0.878 on the
held-out size 4. And the diagnosis is clean: the **propagation is exact by construction** (it never
fails to generalize); the residual drop at size 5 is entirely in the **perception/parse front-end**
(edge recovery 1.0 → 0.89 → 0.78) — i.e. the *discovery* half, not the *reasoning* half.

So the architectural answer is constructive: a vanilla decoder lacks size-general causal reasoning, and
**embedding an explicit causal structure + an iterative propagation core + routing the answer through
it supplies exactly that.** This is the first brick of an embedded causal LM. The next brick is a
size-robust discovery front-end (the remaining bottleneck) and an explicit do()-operator on `A` (which
this architecture already admits — zero a column to intervene).

### Reproduce

```
uv run --extra torch python examples/causal_core_architecture.py
```

---

# Embedded do() — observational + interventional reasoning in one core

Code: `causal_core_do.py`. Adds the do() operator to the embedded core, unifying the two prior results.
One learned adjacency A; the answer is read two ways and **do() is the architectural switch**:
"are X,Y correlated?" reads reachability *with* the back-door (common-cause) term; "does X cause Y?"
reads *directed* reachability only (intervention removes the back-door). 79K params, trained on
2/3-variable graphs, tested on 4/5 held out by size.

| size | corr (observational) | cause (interventional) | in training? |
|---|---|---|---|
| 2 / 3 | 1.000 | 1.000 | trained |
| 4 | 0.741 | 0.890 | held-out |
| 5 | 0.600 | 0.771 | held-out |

**The decisive test — confounded pairs (correlated but X does NOT cause Y), asked as "does X cause Y?":**

| size | embedded do()-core | a correlation-only reasoner |
|---|---|---|
| 3 | **1.000** | 0.000 |
| 4 (held-out) | **0.973** | 0.000 |
| 5 (held-out) | **0.939** | 0.000 |

One architecture answers both query types; on the correlation≠causation trap the do()-routed core is
0.94–1.0 even on graph sizes never trained on, while a correlation-only reasoner is 0.0 by definition.
That is causal reasoning **beyond correlation, embedded in the weights** — and it size-generalizes.

Honest limit: the *observational* read degrades faster at size 5 (0.60) than the causal read (0.77),
because the correlation read aggregates a common-cause term over *all* Z and is thus more sensitive to
edge-perception errors at larger sizes. The remaining bottleneck is the **discovery/perception
front-end**, not the causal reasoning — the next brick.

### Reproduce

```
uv run --extra torch python examples/causal_core_do.py
```

---

# Size-robust perception — the last wall closed

Code: `causal_core_perception.py`. The do()-core's only remaining weakness was *perception*: the
position-indexed text encoder did not length-generalize (edge recovery 1.0→0.78). The fix is also
architectural — represent the evidence as a **set of relational facts** (directed edges) and build the
adjacency with a **permutation/count-invariant relational encoder** (each edge an item, each variable
pair a query matched against the set; no absolute positions). Same propagation + do() core.

| size | corr (obs) | cause (int) | edge recovery | confounded | in training? |
|---|---|---|---|---|---|
| 2 / 3 | 1.000 | 1.000 | 1.000 | 1.000 | trained |
| 4 | **1.000** | **1.000** | **1.000** | **1.000** | held-out |
| 5 | **1.000** | **1.000** | **1.000** | **1.000** | held-out |

Everything is 1.000 — both query types, edge recovery, and the correlation≠causation trap — on graph
sizes never trained on. **The embedded causal core is now fully size-robust: perception, reasoning, and
do() all generalize.**

**Honest scope.** This closes the size wall *given the structure as relational facts* (edges). The
relational matcher is content-based and size-invariant by construction, so the perception task (bind
variables, match the edge set) is exactly what it is built for. The remaining real-world gap is the one
deferred earlier: obtaining those facts from **raw data / correlations** (discovery), and coupling the
core to a language model. Those are the next bricks toward a large causal language model — but the
*causal computation* itself (structure + reachability + do, beyond correlation, size-general) is now an
embedded, working architecture.

### Reproduce

```
uv run --extra torch python examples/causal_core_perception.py
```

---

# Closing the loop: embedded discovery (evidence → structure → reasoning)

Code: `causal_core_discovery.py`. The last brick: instead of being handed the edges, the model
**discovers** the adjacency from *statistical evidence*, in the weights, then reasons + do() over it.
Discovery is a relational cross-attention (each variable pair queries the evidence-fact set → edge
logits). Two evidence regimes feed the same core; supervised toward the true DAG. Trained on 2/3-var
graphs, tested on 4 held out.

| size | evidence | edge recovery | answer acc | confounded-cause |
|---|---|---|---|---|
| 3 | interventional | 1.000 | 1.000 | 1.000 |
| 3 | observational | 0.817 | 0.802 | 0.134 |
| 4 (held-out) | interventional | **1.000** | **1.000** | **1.000** |
| 4 (held-out) | observational | 0.793 | 0.663 | 0.070 |

**The whole pipeline now runs in one embedded model: evidence → structure → causal answer.** And the
identifiability ceiling emerges *inside the weights*:

- **Interventional evidence** (direct effects do(i)→j) recovers the oriented DAG perfectly and answers
  everything, including confounded pairs, even on the held-out size — the loop closes and generalizes.
- **Observational evidence** (conditional-independence facts) recovers the *skeleton* (edge ≈ 0.80) but
  not the unidentifiable orientations, so on **confounded** pairs (correlated but X does not cause Y) it
  fails (0.13 → 0.07): without interventions it literally cannot decide direction. That is the
  Markov-equivalence limit, learned and exhibited end-to-end.

So "beyond correlations" holds at the **discovery** level too: only interventional evidence lets the
embedded discoverer orient the graph and answer causal questions correctly. (Honest note: observational
discovery is the genuinely hard learned half — it degrades at size 4, where the orientation logic is
harder to generalize than the relational matching; this is the natural next target.)

See `ARCHITECTURE.md` for how all the bricks compose into an embedded causal LM.

### Reproduce

```
uv run --extra torch python examples/causal_core_discovery.py
```

---

# A tiny causal *language* model: prose → embedded core → answer

Code: `causal_lm_coupling.py`. Couples the embedded core to language. Input is now PROSE about named
entities; a language front-end binds entity **words** to the core's variables (content-addressed) and
reads causal **verbs** as edges; the core reasons (reachability + do()); the answer is read out.

```
input : "smoking causes tar . tar causes cancer . does smoking cause cancer ?"  -> yes
```

22K parameters. Trained on 2/3-entity prose, tested on 4-entity prose held out by size.

| size | observational | interventional | confounded-cause | in training? |
|---|---|---|---|---|
| 2 / 3 | 1.000 | 1.000 | 1.000 | trained |
| 4 | **1.000** | **1.000** | **1.000** | held-out |

Worked held-out examples (4 entities, the model reading sentences):

```
"slippery triggers smoking . tar triggers stress . are smoking and tar correlated?"  -> no   ✓
"... fitness causes stress . exercise increases tar . are exercise and stress correlated?" -> yes ✓ (common cause: fitness)
```

**Language in → embedded causal reasoning → answer out.** From prose it builds the structure, reasons
with the do() switch, and distinguishes correlation from causation — recognising common-cause
confounding in the text and answering "correlated but not causal" where a correlation reader cannot. It
holds on 4-entity prose never trained on. A causal language model, in miniature.

**Scope (honest).** This is simple subject-verb-object prose with a small entity vocabulary, and the
question's type and entities are parsed for the model; what is *learned* is reading the scenario prose
into causal structure and reasoning over it. Real natural language (rich phrasing, coreference,
multi-sentence), a generative verbaliser, and joint training with a full LM are the next steps — see
`ARCHITECTURE.md`.

### Reproduce

```
uv run --extra torch python examples/causal_lm_coupling.py
```

---

# Active discovery: choosing interventions to break the MEC ceiling efficiently

Code: `causal_active_discovery.py`. Beyond correlation needs interventions — but experiments are
expensive, so *which* you run matters. Observation gives the CPDAG (colliders oriented, the rest
undirected); each `do(v)` orients the still-undirected edges incident to v; a policy picks the
sequence. Causal-query accuracy vs intervention budget, on random 6-node DAGs (avg over 479 graphs
with ≥2 undirected edges):

| budget | random | active | learned |
|---|---|---|---|
| 0 (observation only) | 0.813 | 0.813 | 0.813 |
| 1 | 0.880 | **0.942** | 0.945 |
| 2 | 0.927 | **0.991** | 0.992 |
| 3 | 0.967 | 0.999 | 0.999 |
| 4 | 0.989 | 1.000 | 1.000 |

Budget 0 is the **observation-only MEC floor (0.81)** — correlation cannot orient the undirected edges.
**ACTIVE** (intervene where it resolves the most ambiguity) reaches ~0.99 in **2** interventions where
**RANDOM** needs ~4–5, and a small **LEARNED** policy (trained to imitate the greedy oracle) recovers
the active curve. Going beyond correlation *efficiently* is about **choosing** interventions — the
active-causal-discovery capability a causal LM needs to seek the evidence that breaks the MEC.

### Reproduce

```
uv run --extra torch python examples/causal_active_discovery.py
```

---

# Hybrid: a real GPT-2 + the embedded causal core

Code: `causal_hybrid_lm.py`. The original "plug it into a real LM" step. A genuine GPT-2 decoder
(`transformers.GPT2Model`) is the language backbone; the embedded causal core is a reasoning head on
its hidden states (gather a rep per entity → edge MLP → adjacency → reachability + do()). The
experiment is the same backbone, with vs without the core, on natural-language causal QA.

| | observational | interventional | confounded-cause | size |
|---|---|---|---|---|
| **vanilla GPT-2** | 0.642 | 0.767 | 0.214 | 3 (trained) |
| **hybrid** | **0.998** | **0.997** | **0.994** | 3 (trained) |
| **vanilla GPT-2** | 0.449 | 0.561 | 0.242 | 4 (held-out) |
| **hybrid** | **0.732** | **0.834** | **0.869** | 4 (held-out) |

Same 0.34M GPT-2 backbone. **The hybrid dominates vanilla everywhere**, and the confounded column is
decisive: a vanilla GPT-2 doing the reasoning in its own weights sits at **0.21–0.33** (it answers
"causes" whenever correlated — worse than chance on the trap), while the hybrid is **0.99 in-dist and
0.87 held-out**. Plugging the embedded core into a real LM is what delivers reasoning **beyond
correlation**; a vanilla LM internalising it is the harder, weaker path (exactly the diagnosis from the
start of this arc).

Honest note: the hybrid still drops at held-out size 4 (0.73–0.87, not 1.0). The limiter is the
*perception* front-end — GPT-2's positional entity-gathering doesn't fully size-generalize — not the
core's reasoning (the pure relational core hit 1.0). A relational/size-invariant reader is the next
refinement; see `ARCHITECTURE.md`.

### Reproduce

```
uv run --extra torch python examples/causal_hybrid_lm.py
```

---

# Robustness: multi-seed mean ± std for the load-bearing claims

Code: `causal_robustness.py`. The headlines above were single-seed. This re-runs the three most
load-bearing claims across 3 seeds (full training budget) and reports mean ± std.

| claim | metric | result (mean ± std over 3 seeds) |
|---|---|---|
| **(a) hybrid vs vanilla** | hybrid confounded (size 3) | **1.000 ± 0.000** |
| | hybrid confounded (size 4, held-out) | **0.914 ± 0.071** |
| | vanilla confounded | 0.15 ± 0.05 |
| **(b) active vs random** | active @ budget 2 | **0.991 ± 0.002** |
| | random @ budget 2 | 0.923 ± 0.003 |
| | observation floor | 0.814 ± 0.001 |
| **(c) discovery obs vs int** | interventional confounded | **0.859 ± 0.222** |
| | observational confounded | 0.113 ± 0.097 |

The gaps hold across seeds: hybrid ≫ vanilla, active ≫ random, interventional ≫ observational. Active
and the hybrid are tight; discovery-from-evidence (interventional) is robust in the mean but the most
seed-variable (one seed 0.60) — it is the hardest *learned* component.

**Honest caveat that multi-seed surfaced.** A reduced-budget sweep (8 epochs) made the hybrid **bimodal**
across seeds — `[1.0, 0.0, 1.0]` — i.e. it *collapsed* on some seeds. That was **under-training**: at
the full budget (12 epochs) it is `[1.0, 1.0, 1.0]` in-dist and `0.914 ± 0.071` held-out. So the results
are robust *with adequate training*; under-trained, the embedded-core models are seed-fragile. The
single-seed headlines were the good mode — now confirmed to be the typical mode at full budget.

### Reproduce

```
uv run --extra torch python examples/causal_robustness.py
```

---

# Discovery from RAW DATA — structure inferred from SCM samples, not oracle facts

Code: `causal_data_discovery.py`. Every prior discovery experiment fed pre-digested facts (edges, CI
results). Here the model sees only **raw samples** from a binary SCM and must infer the adjacency
(then reason). The front-end is permutation-invariant over samples: per ordered pair it embeds the raw
tuple `[x_u, x_v, do-on-u?, do-on-v?]` and averages over samples. Judged against trivial baselines so
the result is not a degenerate artefact (size-4 test graphs):

| data regime | edge-recovery (all-no-edge base) | answer-acc (majority base) | confounded-cause |
|---|---|---|---|
| observational | 0.794 (0.750) | 0.715 (0.509) | **0.283** |
| interventional | **0.924** (0.755) | **0.873** (0.509) | **0.845** |

Both beat the trivial baselines (it is learning structure from data, not collapsing). The pattern is
the expected one, now **from raw data**:

- **Interventional** samples → recovers the oriented DAG (edge 0.92) and resolves confounded pairs
  (0.85): beyond correlation, end-to-end from data.
- **Observational** samples → reads dependence (answer 0.72 > majority) but **cannot orient** → on
  confounded pairs it fails (0.28): the Markov-equivalence limit, exhibited from raw data.

Honest note: this is the hard regime (the AVICI/CSIvA amortized-discovery problem). A first attempt
with a 1K-param aggregator **collapsed** (answer at chance, a no-bias artefact that faked confounded
~1.0); a non-tiny encoder + baselines were needed to get a real, non-degenerate result. Strong
discovery from genuinely raw, messy data at scale remains open.

### Reproduce

```
uv run --extra torch python examples/causal_data_discovery.py
```

---

# The PURE path: can a real GPT-2 internalise causal reasoning? (multi-seed)

Code: `causal_pure_lm.py`. The hybrid bolts an explicit core onto GPT-2; the "pure" alternative is a
real LM doing the reasoning *in its own weights*. We give it its best shot — including grounding
*pressure* (an aux loss making its hidden states encode the true edges, while the answer still comes
only from the LM head) — and test multi-seed on confounded-cause (correlated but not causal).

| | confounded (mean ± std, 2 seeds) |
|---|---|
| pure-direct, in-dist (size 3) | 0.155 ± 0.056 |
| pure-direct, held-out (size 4) | 0.150 ± 0.095 |
| pure-grounded, in-dist (size 3) | 0.370 ± 0.064 |
| pure-grounded, held-out (size 4) | 0.529 ± 0.108 |
| **hybrid (explicit routed core), reference** | **~1.000 / ~0.914** |

**The pure path does not internalise causal reasoning.** Vanilla GPT-2 sits at ~0.15 (it answers
"causes" whenever correlated). Grounding *pressure* helps — 0.37–0.53 — i.e. forcing the structure into
the hidden states does make the LM head use it *somewhat*, but it stays far below the hybrid's routed
core (~1.0 / ~0.91). This is **presence ≠ mediation at LM scale**: pushing structure into the weights is
not enough; the answer must be *routed through* an explicit causal computation. So, of the fork
"internalise vs bolt-on", the **hybrid (bolt-on explicit core) is the path that works**; the pure
internalised path is the weaker one — exactly the diagnosis from the start of the arc, now confirmed
with a real LM and multiple seeds.

### Reproduce

```
uv run --extra torch python examples/causal_pure_lm.py
```

---

# Corrected (post-audit): is the causal REASONING learnable, or only hand-coded?

Code: `causal_core_learned_reasoning.py`. The independent audit (`AUDIT.md`) found the decisive caveat:
in the earlier "embedded core" scripts the reachability + do() + back-door computation is a **hard-coded
differentiable formula** — the models only learned edge perception, so the size-generalization was *by
construction*, not learned. This experiment fixes that: it gives the model the **true structure** (so
perception is not the variable) and makes the **reasoning itself learned**, then tests size-extrapolation
honestly (random variable slots; trained on 2/3-var, tested on 3/4/5).

| reasoner | size 3 (in-dist) | size 4 (held-out) | size 5 (held-out) |
|---|---|---|---|
| | corr / cause / conf | corr / cause / conf | corr / cause / conf |
| **HARDWIRED** (not learned — reference) | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 |
| **MLP** (learned) | 0.99 / 0.99 / 0.96 | 0.84 / 0.91 / 0.88 | 0.77 / 0.83 / 0.82 |
| **GNN** (learned message passing) | 1.00 / 1.00 / 1.00 | 0.88 / 0.97 / 0.92 | 0.80 / 0.91 / 0.80 |

Honest conclusions:

1. **Causal reasoning IS learnable in-distribution** — a learned GNN matches the hand-coded core at
   size 3 (1.0 incl. confounded), MLP ~0.99. So it is not *only* hand-codable.
2. **It does NOT fully size-extrapolate.** Both learned reasoners degrade out of size (down to 0.77–0.91
   at size 5), where the hardwired formula stays 1.0 *because it is the algorithm, not learning*.
3. **Architectural alignment helps:** the GNN (message passing, aligned to reachability) generalizes
   noticeably better than the MLP (cause 0.97 vs 0.91 at size 4; 0.91 vs 0.83 at size 5) — but does not
   close the gap.

**So the earlier "size-general embedded causal core" headline was the hand-coded formula doing the
work, not learning** (per the audit). The corrected, honest result: a *learned* causal reasoner does the
reasoning in-distribution and generalizes *partially* in size (better with algorithmic alignment),
consistent with the broader length/size-generalization literature — it does not match the by-construction
extrapolation of the hardwired algorithm.

### Reproduce

```
uv run --extra torch python examples/causal_core_learned_reasoning.py
```

---

# The fully-learned hybrid — the honest end-state (nothing hand-coded)

Code: `causal_hybrid_learned.py`. The final correction: a real GPT-2 reads the prose (learned
perception → entity reps → soft adjacency) **and** a **learned GNN reasoner** (init from the GPT-2 reps,
message-passing over the learned adjacency) produces the answer. The reachability / do() / back-door
logic is **not** wired in — both halves are learned. Multi-seed, confounded-cause vs vanilla, with the
hand-coded hybrid and given-structure GNN as cited references.

| metric (mean ± std, 2 seeds) | value |
|---|---|
| vanilla confounded (size 3 / 4) | 0.155 / 0.150 |
| **learned-hybrid confounded (size 3)** | **0.430 ± 0.024** |
| **learned-hybrid confounded (size 4, held-out)** | **0.390 ± 0.068** |
| learned-hybrid **cause** query, balanced (size 3) | **1.000 ± 0.000** |
| learned-hybrid **cause** query, balanced (size 4) | **0.901 ± 0.060** |
| *ref:* hand-coded hybrid confounded | ~1.000 / ~0.914 |
| *ref:* GNN reasoner given true structure, confounded | ~1.0 / ~0.8–0.9 |

**This deflates the headline, honestly:**

1. **The general causation query IS learned** — `cause` = 1.000 in-dist, 0.901 held-out, with nothing
   hand-coded, beating vanilla everywhere. (The balanced `cause` column rules out a constant-"no"
   artifact.)
2. **But it FAILS the confounding trap** — confounded 0.43 / 0.39, i.e. *below chance*: with nothing
   wired in, the end-to-end-learned system does **not** robustly distinguish correlation from causation
   on the hard (common-cause / reverse) cases. It beats vanilla (~0.15) but is far below the hand-coded
   hybrid (~1.0).
3. **Diagnosis:** the confounded distinction is the most structure-sensitive query; a GNN *given the
   true structure* gets it (~0.9), but with *learned* (imperfect) perception the signal collapses, and a
   learned reasoner is less robust than the wired-in back-door formula.

**So the "beyond correlation" results that looked strongest came from the hand-coded formula.** A
genuinely end-to-end-learned causal LM learns *general* causation but does **not** reliably go beyond
correlation on confounded cases — exactly the boundary the audit exposed. That is the honest conclusion
of the exercise: causal reasoning is partly learnable, the do()/back-door distinction is (so far) only
reliable when wired in or given clean structure.

> **[Superseded — see "Localizing the failure (and the two-stage fix)" below.]** This conclusion was
> correct for the *jointly-trained* hybrid, but premature as a statement about fully-learned systems.
> The ablation below shows the ~0.43 failure is an end-to-end **training** artifact (not perception,
> not the reasoner's capacity), and a **two-stage** recipe lifts a fully-learned, nothing-hand-coded
> system to **1.000 ± 0.000** confounded in-dist.

### Reproduce

```
uv run --extra torch python examples/causal_hybrid_learned.py
```

---

# Localizing the failure (and the two-stage fix)

Code: `causal_perception_bottleneck.py` (diagnosis), `causal_hybrid_twostage.py` (fix). The fully-learned
hybrid's ~0.43 confounded number above invites the conclusion "fully-learned systems can't go beyond
correlation." Two bracketing facts said otherwise: a GNN *given* clean structure reaches confounded
~1.0 in-dist (learned reasoning is not the blocker), yet the end-to-end hybrid fails. The only thing
that changed is perception — so we isolated it.

**Diagnosis — perception is NOT the bottleneck; end-to-end *training* is.** We take the fully-learned
hybrid's *perceived* soft adjacency, measure it, then feed TRUE vs PERCEIVED structure into reasoners
whose causal logic is known-good (the exact hand-coded formula; a GNN trained on clean structure).
Multi-seed, on confounded-cause:

| | edge F1 | confounded, TRUE adj | confounded, PERCEIVED adj |
|---|---|---|---|
| perceived adjacency quality | **0.863 ± 0.008** | — | — |
| HARDWIRED formula (exact algo) | — | 1.000 / 1.000 (s3/s4) | **1.000 / 0.976** |
| GNN trained on clean structure | — | 1.000 / 0.908 | **1.000 / 0.917** |
| end-to-end hybrid (joint-trained) | — | — | 0.43–0.51 |

The perceived graph (F1 0.86), fed to the *correct* algorithm — soft **or** thresholded — yields ~1.0
confounded. A GNN trained on clean structure transfers to the perceived graph at ~1.0. Only the
end-to-end jointly-trained reasoner fails. **Disambiguation:** feeding the hybrid's *own* trained
reasoner a thresholded graph leaves it unchanged (soft 0.506 → thresholded 0.506) — so it is not the
soft input, it is the *jointly-trained weights*. The `cause` query survives the same edge noise
(~0.89 held-out): confounding is the structure-sensitive query, and joint training simply never learns
its back-door computation.

**The fix — decoupled (two-stage) training.** Stage A trains GPT-2 + an edge MLP on the edge-recovery
loss only (→ a perceived adjacency). Stage B trains a structure-only GNN reasoner on clean
(teacher-forced) structure + the answer loss. At inference, prose → frozen Stage-A perception →
thresholded adjacency → Stage-B reasoner → answer. **Nothing is hand-coded**; the two halves are simply
trained apart. Same data/seeds, end-to-end from prose:

| system (fully learned, nothing hand-coded) | confounded s3 (in-dist) | confounded s4 (held-out) |
|---|---|---|
| JOINT hybrid (baseline) | 0.506 ± 0.084 | 0.586 ± 0.367 |
| **TWO-STAGE (fix)** | **1.000 ± 0.000** | **0.933 ± 0.003** |

Decoupling fixes both the **accuracy** (1.0 vs 0.51 in-dist) and the **fragility** (±0.003 vs ±0.367
held-out). The balanced `cause` query stays 1.000 / 0.862, ruling out a constant-"no" artifact.

**Corrected conclusion.** A fully-learned causal LM (learned perception + learned reasoner, nothing
wired in) **does** go beyond correlation on confounded cases — the earlier ~0.43 was an
optimization / credit-assignment artifact of naive joint training, removed by decoupling. Honest
scope: (1) Stage B teacher-forces the *true* structure at train time — a privileged-information
training device (like the edge aux-loss), not a hand-coded reasoner; at inference the reasoner sees
only the perceived graph. (2) The result is exact in-distribution (1.0) and strong but not perfect
out of size (0.93), consistent with the partial size-extrapolation of learned reasoners. (3) This is
still the *bolt-on* path (perception + a separate reasoner); the **pure** path — a single LM
internalising the computation in its own weights — remains negative (see above). (4) Small-scale,
synthetic, CPU. The wall to a *large* causal LM is still scale, real language, and real-data
discovery — not the causal mechanism, and now not naive trainability either.

### Reproduce

```
uv run --extra torch python examples/causal_perception_bottleneck.py   # diagnosis (the 2x2)
uv run --extra torch python examples/causal_hybrid_twostage.py         # the two-stage fix
```
