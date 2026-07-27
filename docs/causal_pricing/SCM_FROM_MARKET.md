# Can we learn the market's SCM, and build on the causal market simulator?

> Short answer: **yes to building on the simulator, and a heavily qualified yes to learning the
> graph — but observational discovery is the wrong tool and the probe shows exactly why.**
> Evidence: `experiments/cpricing/poc_discovery.py`, output in `DISCOVERY_OUTPUT.txt`.

## Part 1 — Learning the SCM from market data

### The failure mode, measured

Constraint-based discovery (PC/FCI) assumes **faithfulness**: no conditional independence in the
data beyond those the graph implies. **Efficient markets violate this by construction**, because
arbitrage is a mechanism whose entire job is to cancel exploitable dependence.

The probe models this directly. A scheduled macro surprise `News` affects an implied-vol signal
`IV` two ways: directly, and through dealer hedging `Flow` that trades against it. When the
arbitrage is effective the paths cancel:

```
CMI(News ; IV)        = 0.00025      <- marginally independent
CMI(News ; IV | Flow) = 0.57082      <- the dependence is right there, once Flow is blocked
```

PC deletes the real `News -> IV` edge. Worse, the damage propagates: with that edge gone, the
remaining triple is read as a collider, and PC returns **`IV -> Flow`** — the causal arrow
*backwards*, since truly `Flow -> IV`. A faithfulness violation does not just lose an edge; it
manufactures a confidently wrong one.

And the punchline, from the contrast arm:

| Market | `News--IV` recovered? |
| --- | --- |
| Fully arbitraged (`arb_strength = 1.0`) | **No** |
| Half arbitraged (`arb_strength = 0.5`) | Yes |

**The better the market works, the worse discovery works.** Efficiency is adversarial to
constraint-based structure learning. This is not a tuning problem.

### Four more reasons the naive version fails

Beyond faithfulness, and all of them real:

1. **Non-stationarity.** PC assumes one fixed DAG; market graphs change by regime. (This is what
   the energy-market regime-detection SCM paper is about.)
2. **Pervasive latent confounding.** Order flow, positioning, macro. FCI handles latents but on
   market data returns a PAG that is mostly circles — technically correct, practically empty.
3. **Simultaneity.** Price ↔ vol ↔ flow are simultaneous at any sampling interval, so acyclicity
   fails. The repo's `experimental/cyclic` is the right home, not `discovery`.
4. **Temporal aggregation.** The causal timescale is milliseconds; the data is daily. A DAG is not
   preserved under subsampling.

Plus a repo-specific one: **`discover` is discrete-only** (CMI over integer columns). Market data
is continuous, so today it needs discretisation, which is lossy exactly in the tails that matter.

### What is actually defensible: the graph is mostly known already

The reframe that makes this tractable: **you do not need to discover the graph. A derivatives
market is a partially-known SCM.** Enormous structure is available a priori —

- payoff functions are deterministic and known (`payoff = f(underlying path)`);
- no-arbitrage identities (put–call parity, forward relations) are *constraints*, not edges to
  learn;
- temporal ordering gives strong tier constraints, and the repo already supports tiers
  (`discover_and_fit(data, tiers=...)`, the M1a slice that recovered `{Z}` on 10/10 seeds);
- the instrument hierarchy (underlying → derivative) is mechanically known.

So the real problem is: **given a large known backbone, test and orient the few genuinely uncertain
edges.** Those are few, contested, and economically meaningful — does hedging flow move the
underlying? does the surface lead or lag realised vol? does the options market lead spot? That is a
tractable discovery problem, and it is where discovery is reliable.

### The strongest asset: markets hand you real interventions

`discover_interventional` is already in the library (Hauser–Bühlmann I-essential graphs +
Peters–Bühlmann invariance), and markets supply genuinely exogenous interventions with known timing:

- **Scheduled macro announcements** (FOMC, CPI, NFP): the *timing* is exogenous and known in
  advance, the *content* is not. This underwrites the whole high-frequency-identification
  literature in macro-finance (Kuttner; Gürkaynak–Sack–Swanson; Nakamura–Steinsson).
- **Regulatory experiments**: the SEC Tick Size Pilot was a literal randomised controlled trial on
  market structure. Index rebalancing is quasi-random assignment.
- **Mechanical events**: option expiry (a scheduled shock to dealer gamma), circuit breakers,
  margin changes.

Invariance-based orientation is far more robust to the faithfulness problem than pure CI testing,
because it does not need the dependence to survive in the observational distribution.

### But the shipped tool does not close the loop — and this is the concrete gap

Tested, and it fails:

```
TEST 2  discover_interventional with do(Flow)
  News--IV edge recovered : False
  -> it ORIENTS existing undirected edges; it never re-tests the skeleton,
     so a faithfulness deletion survives the intervention.

TEST 3  is the information there at all?
  CMI(News ; IV) under do(Flow=1) = 0.95982
  CMI(News ; IV) observational    = 0.00025      (~3800x)
```

The dependence is *overwhelmingly* present in the interventional regime. The skeleton phase simply
never looks at it. **The library gap is small and precisely specified: let `discover` run its
independence tests per-regime (or pooled across regimes) rather than only orienting afterwards.**
That is the single highest-value addition this direction needs, it is domain-agnostic, and it
belongs in core.

Second addition, also small: a continuous CI test (partial correlation, or a k-NN CMI estimator) so
`discover` stops requiring discretisation.

---

## Part 2 — Building on the causal market simulator

[Towards Causal Market Simulators](https://arxiv.org/abs/2511.04469) (TNCM-VAE: VAE + SCM,
DAG-constrained decoder, causal Wasserstein training, public code at `github.com/thummd/tncm`) is
the right thing to build on. Four things the library adds, ranked by what I would actually do.

**1. Replace the inverse — the measurable win.** TNCM-VAE abducts with a learned VAE encoder: an
*approximate* inverse. [`GAINS.md`](GAINS.md) G1b measured what that costs — the pairing gain
collapses from 1850× to 0.5× as inversion error grows, and the estimate goes **silently biased**
(−3.43 against a true −0.080) while still reporting a confident standard error. Swap the encoder
for per-step invertible flow mechanisms (`ConditionalFlowMechanism`, already shipped with `invert`)
and abduction becomes exact. **This is directly testable on their own public code and their own
benchmark**, where they report counterfactual L1 distances of 0.03–0.10 against ground truth on
OU-inspired synthetic models. Prediction: exact inversion cuts that materially. Runnable,
falsifiable, against a published baseline. Best single next step.

**2. Bound-based validation — the conceptual contribution.** TNCM-VAE validated on *synthetic* models
where ground-truth counterfactuals exist. **On real markets there is never a ground-truth
counterfactual** — that is what makes it a counterfactual. So how does anyone validate a causal
market simulator on real data? Nobody has a good answer, and it is the field's central
methodological hole.

The library has one: **partial identification.** You cannot compare a counterfactual to truth, but
you *can* check whether it falls inside a certified bound (Manski, MSM, or the gain-loss interval
from T2). A simulator whose counterfactual lands outside the bound is **falsified** under stated
assumptions. That converts an unfalsifiable generative claim into a testable one, and it is a
general evaluation protocol for causal market simulators rather than a critique of one. If any
single idea in this whole line is the paper, it is this one.

**3. Certify their counterfactuals — cheap and honest.** An amortised/learned inverse is
`AmortizedGaussianAbduction` in the repo's taxonomy, which caps the claim at `kind=EMPIRICAL`, not
`IDENTIFIED`. Their pipeline has no such label. Adding it costs almost nothing and is exactly the
distinction their architecture currently elides.

**4. Supply and certify the graph.** Discovery (per Part 1, interventional) plus POMIS tells them
not just *what* the graph is but **which interventions are worth simulating at all** — POMIS returns
the minimal sufficient intervention sets. A simulator that can generate any intervention wastes
effort on ones that are not identified.

---

## What I would actually do

In order, and the first two are the whole thing:

1. **Ship the per-regime skeleton test** in `discover` (small, domain-agnostic, core). Without it
   the interventional-discovery story does not close, as measured above.
2. **Run the exact-inverse head-to-head against TNCM-VAE** on their public benchmark. It is a
   contained experiment with a published number to beat.
3. Bound-based validation as the methodological paper.
4. Only then attempt real market data — and with scheduled macro events as intervention targets,
   never with observational PC on a returns panel.

**What I would not do:** point PC at a panel of returns and report the graph. On an efficient
market that produces a confidently wrong answer, and the probe shows it producing a *reversed* edge,
not merely a missing one.
