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
