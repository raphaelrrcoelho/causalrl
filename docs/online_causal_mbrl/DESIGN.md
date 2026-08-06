# Online causal model-based RL: learning the SCM while acting

**Status:** design, 2026-08-04. Branch `online-causal-mbrl`.

## The gap this closes

`fit_scm` (merged in #34/#35) learns an SCM from a *static* table. The RL audit that shipped with it
recorded the honest verdict: a fitted SCM is an environment you can act in, but no agent in this
library *plans* in one, and nothing under `envs/` or `agents/` referenced the learned model at all.
Fitting mechanisms from a fixed dataset is supervised learning; it has no policy, no reward, no
exploration and no regret.

This sub-project closes that gap in the direction that actually makes it reinforcement learning: the
agent learns the SCM **along the way**, from its own on-policy interventional data and from
off-policy observational logs, and replans as the model sharpens.

## This is not novel, and the docs must not imply otherwise

The framework is published. What follows is an implementation on this library's primitives, not a
research contribution. Specifically:

- **The alternating loop** — intervene to learn structure during exploration, use the learned
  structure to guide policy during exploitation — is Sun et al., *Learning by Doing: an online causal
  reinforcement learning framework with causal-aware policy*, Science China Information Sciences
  (2024), [arXiv:2402.04869](https://arxiv.org/abs/2402.04869); code at
  [DMIRLAB-Group/FaultAlarmRL](https://github.com/DMIRLAB-Group/FaultAlarm). We do **not** port their
  code — it is built around a domain-specific fault-alarm environment, and this library already has
  the discovery and fitting primitives. We implement the same loop and cite them.
- **The observational + experimental fusion** is Bareinboim & Forney, *Bandits with Unobserved
  Confounders: A Causal Approach* (MABUC), and Forney, Pearl & Bareinboim, *Counterfactual
  Data-Fusion for Online Reinforcement Learners*. Their result is the reason this design exists at
  all: under unobserved confounding an agent needs **both** quantities, and averaging confounders out
  can incur unbounded regret. This library is already anchored to that taxonomy.
- **Thompson sampling over structure** appears in Ortega & Braun (2013), *Generalized Thompson
  Sampling for Sequential Decision-Making and Causal Inference*,
  [arXiv:1303.4431](https://arxiv.org/abs/1303.4431), and in later causal-bandit variants.
- **Choosing interventions by expected information gain** is standard active intervention design:
  Scherrer et al., *Learning Neural Causal Models with Active Interventions*
  ([arXiv:2109.02429](https://arxiv.org/abs/2109.02429)); Zhang et al., *Active learning for optimal
  intervention design in causal models*, Nature Machine Intelligence (2023); and the
  information-directed-sampling and adaptive-experimental-design lines
  ([arXiv:2405.11548](https://arxiv.org/abs/2405.11548),
  [arXiv:2510.08207](https://arxiv.org/abs/2510.08207)).
- **The object the agent maintains has a name**: the **interventional Markov equivalence class
  (I-MEC)** — the DAGs consistent with both the observational and the interventional distributions.
  Use that term, not an invented one.

## What is ours

A stance, not a result. Most implementations return *the* learned graph. This library already
refuses to guess: `orient` raises rather than picking an orientation the data does not identify, and
`fit_scm_mec` refuses to silently truncate an equivalence class. So this agent can report **when it
still does not know** — `belief_size()` and `structure_uncertain()` are first-class — and a caller can
route an ambiguous decision to the certificate layer instead of committing to an arbitrary member.

## Architecture

`src/causalrl/agents/online_causal_mbrl.py`, one class.

```python
OnlineCausalMBRL(
    variables, *, treatment, outcome, actions,
    policy="thompson",        # "thompson" | "average" | "robust"
    max_members=32, refit_every=8, n_rollout=512, seed=0,
)
```

**State.** An observational buffer (off-policy, possibly confounded); one interventional buffer per
intervention target; the current CPDAG; and the **belief** — the fitted I-MEC members.

**`ingest(data, source=...)`** takes a bulk off-policy log. **`observe(row, intervention=...)`**
appends one transition, routed to the observational buffer or to the buffer for its target.

**`refit()`** is the whole learning step, and it is entirely existing library calls:

```
discover_interventional(obs_buffer, int_buffers, variables)   ->  CPDAG   (the I-MEC)
fit_scm_mec(all_data, cpdag=cpdag, max_members=...)           ->  list[StructuralCausalModel]
```

Orientation comes from the do-data by the invariance principle; mechanisms are fitted from all data.
Called every `refit_every` steps, because rerunning PC per step is wasteful and makes the belief
flicker on sampling noise.

**`act()`** — Thompson sampling over structure by default: draw one member from the belief, return
`argmax_a member.do({treatment: a}).see(n_rollout)[outcome].mean()`. Structural uncertainty produces
exploration for free, and the policy sharpens exactly as the belief collapses. `"average"` (marginalise
over members) and `"robust"` (maximin over members) are selectable; `"robust"` is the one to reach for
when a wrong action is expensive.

**`probe()`** — return the intervention target whose interventional distribution the belief members
*disagree* about most, measured by mean pairwise total variation over each candidate's implied
`do()` distribution. That is expected information gain approximated by predictive disagreement, and
it concentrates experiments on edges still unoriented rather than ones already settled. Ties break
deterministically by name.

**`structure_uncertain()`** is `belief_size() > 1`. **`history()`** returns the per-refit record —
belief size, chosen actions, and (when ground truth is supplied by a benchmark) regret.

## The demonstration, and what it must not claim

A world where observational data alone leaves ≥2 I-MEC members that **disagree about the optimal
action** — so an observation-only agent cannot succeed at any sample size, which is the whole point of
Bareinboim's result. Three arms:

- **observational only** — plateaus at the wrong action;
- **interventional only** — correct but sample-hungry;
- **both** — the log buys mechanisms cheaply, the interventions buy orientation.

Reported alongside `mec_size_over_time()` so the belief collapse is visible, not asserted. This is a
synthetic world with exact ground truth, and the example must say so in its own output — the same
discipline `examples/learned_scm_policy.py` follows. No benchmark claims.

## Risks, stated rather than discovered later

- **PC is noisy at finite samples**, so the CPDAG can flicker between refits and the belief can grow
  as well as shrink. Batch the refits and record the trajectory rather than assuming monotone collapse.
- **MEC enumeration is exponential.** `fit_scm_mec` already refuses above `max_members` naming the true
  class size; surface that refusal rather than catching it.
- **`discover_interventional` requires a perfect intervention covering every variable** per target.
  The agent must therefore buffer full rows per intervention, not just the reward.
- **Rollout-based action values cost `n_rollout` samples per action per decision.** That is the price
  of planning in a model rather than reading a value table, and it should be documented, not hidden.
