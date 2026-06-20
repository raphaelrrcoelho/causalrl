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
