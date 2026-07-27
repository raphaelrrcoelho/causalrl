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

**2. Bound-based validation — ~~the conceptual contribution~~ RETRACTED as a novelty claim.**

> **Retraction (July 2026).** An earlier revision of this document claimed that validating a causal
> simulator against partial-identification bounds was novel, that "nobody has a good answer" to
> validating counterfactuals without ground truth, and that "if any single idea in this whole line
> is the paper, it is this one." **All three claims are false.** The correction is kept visible.

The general method — *derive an inequality constraint from the causal model, check the data or the
claim against it, falsify if violated* — is the *standard* falsification paradigm in causal
inference and roughly thirty years old:

- **Pearl's instrumental inequality** (1995) is a testable necessary condition whose violation
  falsifies an IV model. **Balke & Pearl (1997)** give necessary *and sufficient* conditions for an
  observed distribution to be compatible with a binary IV model — literally "outside the bound ⇒
  falsified."
- **An empty identified set falsifies the model** is textbook partial identification.
- The general machinery for deriving such constraints is an active line:
  [Deriving Bounds and Inequality Constraints Using Logical Relations Among Counterfactuals](https://arxiv.org/pdf/2007.00628),
  [testability in continuous IV models](https://economics.sas.upenn.edu/system/files/2019-01/Gunsilius_JMP.pdf),
  [Salvaging Falsified Instrumental Variable Models](https://arxiv.org/pdf/1812.11598),
  [Falsification of Unconfoundedness by Testing Independence of Causal Mechanisms](https://arxiv.org/html/2502.06231v2).
  **Bell's inequality is the same structure**, which is a good hint at how old the idea is.

It is also **already productised**, which is the part that most clearly kills a novelty claim:
**DoWhy ships `gcm.falsify_graph`**, plus a suite of refutation tests (placebo treatment, random
common cause, data subset), documented under "graph refutations"; see also
[Toward Falsifying Causal Graphs Using a Permutation-Based Test](https://arxiv.org/pdf/2305.09565)
and [DoWhy-GCM](https://arxiv.org/pdf/2206.06821).

And "nobody has a good answer to evaluating counterfactuals without ground truth" is simply wrong —
the counterfactual-generation community has **several**, and treats it as a named open problem with
an active benchmark literature: [Benchmarking Counterfactual Image Generation](https://arxiv.org/html/2403.20287)
defines metrics for *composition, effectiveness, minimality and realism*;
[Synthetic Ground Truth Counterfactuals](https://papers.miccai.org/miccai-2025/0894-Paper2090.html)
(MICCAI 2025) builds synthetic ground truth precisely to evaluate causal generative models;
[The Causal Round Trip](https://arxiv.org/pdf/2511.05236) attacks information loss in the abduction
round-trip.

That last one stings usefully: **"composition"** — apply a null intervention and check you recover
the original — is exactly the round-trip test in `poc_ladder.py` C1 (`roundtrip_err = 0.0`). I
presented a standard evaluation metric as though it were an argument.

**What actually survives, stated narrowly.** Not "a new validation paradigm." At most: the
*specific constraint set*. Financial no-arbitrage relations are unusually strong, economically
meaningful, and cheap to check, and pairing them with **counterfactual (L3) partial-ID bounds** —
rather than the observational no-arbitrage penalties already standard in this literature — is a
plausible small delta. Even there the ground is occupied: the
[diffusion IV-surface paper](https://arxiv.org/pdf/2511.07571) already trains with an
arbitrage-violation-suppressing loss, and VolGAN is explicitly arbitrage-free. Using arbitrage as a
validity constraint on financial generative models is **existing practice**; the only unclaimed
inch is using it as a *falsification test on counterfactual output* rather than a *training
penalty on observational output*.

That is an incremental methods contribution, not a headline. It should be a section, not a paper.

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

1. **Run the exact-inverse head-to-head against TNCM-VAE** on their public benchmark. Contained,
   falsifiable, with a published number to beat, and it rests on a *measurement* (G1b) rather than
   on a novelty claim — which, given the record below, is the property to optimise for.
2. **Ship the per-regime skeleton test** in `discover` (small, domain-agnostic, core). Without it
   the interventional-discovery story does not close, as measured above.
3. ~~Bound-based validation as the methodological paper.~~ **Demoted** — see the retraction above.
   Keep it as a section; it is a small delta on a thirty-year-old paradigm that DoWhy already ships.
4. Only then attempt real market data — and with scheduled macro events as intervention targets,
   never with observational PC on a returns panel.

**Scoreboard on my own novelty claims in this line so far.** "The causal-generative × options
intersection is empty" — **falsified**. "Bound-based validation is unclaimed" — **falsified**.
"Inversion error compounds over long paths" — **falsified by my own measurement**. What has
survived is what got measured: the 1830× pairing gain, the G1b bias curve, and the discovery
failure above. **The pattern is unambiguous: measurements have held, novelty claims have not.**
The outstanding recovery-theorem check on T2 should be assumed to follow the same pattern until it
is actually run.

**What I would not do:** point PC at a panel of returns and report the graph. On an efficient
market that produces a confidently wrong answer, and the probe shows it producing a *reversed* edge,
not merely a missing one.
