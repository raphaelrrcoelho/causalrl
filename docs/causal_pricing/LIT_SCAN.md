# Literature scan: causal diffusion × option pricing

> **Revision 2 (July 2026).** Revision 1 claimed the intersection of causal generative modelling
> and options was empty. **That was wrong, and this revision corrects it** — a further search
> round plus a paper supplied directly turned up work squarely inside the gap I had declared.
> The corrections are kept visible rather than edited away.
>
> **Scope and confidence.** Ten targeted web searches plus one paper read in full. This is a
> *scan*, not a systematic review. I could not fetch arXiv, Semantic Scholar, Hugging Face or
> alphaxiv — all return 403 under this session's egress policy — so every paper below except the
> one supplied as a PDF is characterised from **search snippets, not from reading it.** SSRN,
> where quant-finance preprints actually live, was not searchable at all.

## "Diffusion" means at least three different things here

This is the first thing to fix, because Revision 1 conflated the first two and did not know about
the third.

1. **Generative diffusion models** (DDPM/DDIM, score-based). ML sense. Used for IV-surface
   forecasting and path generation.
2. **Causal diffusion processes** — SDEs given interventional semantics (Hansen & Sokol). Pearl
   sense, applied to continuous-time dynamics.
3. **Classical diffusion SDE families with richer noise** — CEV, jump-diffusion, variance-gamma,
   Tsallis/Borland, and **Pearson diffusions**. This is the mathematical-finance sense and has
   nothing to do with causality in either of the above senses.

And a fourth, already flagged: **"causal" in mathematical finance means non-anticipative** (causal
optimal transport, adapted Wasserstein) — a filtration property, not an interventional one.

A cheap but real hazard: **Pearson ≠ Pearl.** "Pearson diffusion for option pricing" is category 3
and is not causal inference in any sense, despite reading like it might be.

---

## The paper that prompted this revision

**[arXiv:2508.14577] "Call Option Price Using Pearson Diffusion Processes"** — Kar, Bhar, Sarkar &
Meka, q-fin.MF, 20 Aug 2025. Read in full.

Category 3, unambiguously. It models the log-return process as a Pearson diffusion

```
dR_t = -theta (R_t - mu) dt + sigma sqrt(2 theta a (1 + R_t^2)) dB_t
```

— linear drift, quadratic squared diffusion coefficient — whose invariant distribution is Pearson
Type IV, chosen because it accommodates the skewness and excess kurtosis that Gaussian log-returns
miss. The contribution is properly done mathematical finance: verify the **Novikov condition** so
the risk-neutral measure is valid and no arbitrage is admitted, prove a unique strong solution
exists under that measure, then fit to Nifty 50 index options (62,442 observations after a
liquidity filter, MLE, 200,000 Monte Carlo replications per price) and beat Black–Scholes and
Heston on MAE/MSE across moneyness and maturity buckets.

**No causal inference appears anywhere in it.** It is an L1 paper, and a good one.

### Why it is nevertheless the most useful thing anyone has sent me on this

Three properties make it an unusually good *substrate* for the causal layer, and none of them are
accidental — they generalise to the whole category-3 family:

1. **It is exactly noise-invertible, so it lands in the T1 identified class.** The Euler step is
   `R_{t+1} = R_t - theta(R_t - mu) dt + sigma sqrt(2 theta a (1 + R_t^2)) sqrt(dt) Z_t`, and the
   diffusion coefficient is bounded away from zero by construction — the `1 +` guarantees it. So
   `Z_t` is recoverable in closed form from `(R_t, R_{t+1})` everywhere, with no degenerate points.
   That is precisely the condition [`PROPOSAL.md`](PROPOSAL.md) T1 says licenses `IDENTIFIED`
   pathwise counterfactuals. The state-dependent-vol arm already implemented in
   `experiments/cpricing/poc_gains.py` (`invert_localvol`) is the same shape of inversion.
2. **Its tail index is an explicit model parameter.** The paper states the invariant distribution
   "has moments of order `k1` for `k1 < 1 + a^{-1}`". So the tail index is `1 + 1/a`, and finite
   variance requires `a < 1`. That is the exact quantity `tail_index_hill` estimates and
   `certify_mean` gates on in [`GAINS.md`](GAINS.md)'s rung-3 story — except here it is *known in
   closed form from the fitted parameter*, which makes it a far better test bed than an estimated
   Hill index. A fitted `a >= 1` means the model's own P&L mean is undefined, and nothing in the
   paper's pipeline would tell you that.
3. **It burns 200,000 Monte Carlo replications per price.** Any sensitivity computed on top of that
   by finite-differencing independent runs pays the full independent-sampling variance — the exact
   inefficiency G2 measured at **1830×** for derivative-shaped queries. This model is invertible,
   so it can have the paired estimator essentially for free.

There is also a cautionary tale in its own related work worth keeping: Borland's Tsallis-noise
option model was later shown by Vellekoop & Nieuwenhuis to **violate the Novikov condition**, thus
admitting arbitrage. A richer noise family can silently break the risk-neutral machinery. Kar et al.
spend real effort proving theirs does not — which is, in substance, hand-writing a certificate.
That is a good argument for the certificate layer aimed at an audience that already believes in the
underlying discipline.

### "Things like it"

The same lineage, all L1, all invertible-in-noise, all candidate substrates: CEV (Cox; Cox & Ross;
Beckers), Merton jump-diffusion (invertibility fails at jumps — which is exactly where the
identification boundary bites and is therefore *interesting*, not a problem), Heston, Variance
Gamma (Madan et al.), Borland's Tsallis model, and Forman & Sørensen's statistical theory of
Pearson diffusions.

---

## CORRECTION to Revision 1: the causal-generative intersection is not empty

Revision 1 said "A × B is empty." **It is not.** What a further search round found:

| Work | Why it matters |
| --- | --- |
| **[Towards Causal Market Simulators](https://arxiv.org/abs/2511.04469)** (Thumm & Ontaneda Mijares, ICAIF 2025 Workshop) | **TNCM-VAE**: VAE + structural causal model, causal constraints enforced via a DAG in the decoder, trained with the **causal Wasserstein distance**, generating counterfactual financial time series. Validated on OU-inspired synthetic AR models; reports L1 distances of 0.03–0.10 on counterfactual probability estimation. Targets stress testing, scenario analysis, backtesting. Code at `github.com/thummd/tncm`. **This is directly inside the gap I claimed.** |
| **[Valuation of Exotic Options and Counterparty Games Based on Conditional Diffusion](https://arxiv.org/abs/2509.13374)** (Sep 2025) | DDPM-style diffusion for exotic option pricing, evaluated through a P–Q dynamic game. Works well on Asians; **explicitly reports failure on heavy tails** (lookbacks, accumulators, snowballs). Future work is risk-neutral calibration and market-maker hedging. |
| [Generative Diffusion Model for Risk-Neutral Derivative Pricing](https://arxiv.org/pdf/2603.20582) | Diffusion generation aimed directly at risk-neutral pricing. |
| [Modeling Causal Mechanisms with Diffusion Models for Interventional and Counterfactual Queries](https://arxiv.org/abs/2302.00860) (DCM) | The proper citation for diffusion-based abduction; latent encodings enabling intervention and counterfactual sampling. |
| [Causal Regime Detection in Energy Markets](https://arxiv.org/pdf/2511.04361) (Nov 2025) | Augmented time-series SCMs applied to a real market — detection, not pricing. |

**What this does to the positioning.** The honest revised claim is narrower and better:

- ~~"Nobody has combined causal generative modelling with financial time series."~~ **False.** TNCM-VAE
  does exactly that, and the exotic-options diffusion paper is already circling risk-neutral
  calibration and hedging.
- **What still appears unclaimed** is the *identification* question: which pricing queries are
  point-identified, which are partially identified, which are not identified at all — and a
  certificate that says which. TNCM-VAE *generates* counterfactuals; it does not ask when they are
  licensed. The exotic-options paper prices; it does not ask what its numbers are epistemically.
- **The overlap is now close enough to be a competitor, not just context.** TNCM-VAE uses causal
  Wasserstein and a DAG-constrained decoder — an approximate, learned inverse. G1b's measurement
  (approximate abduction collapses the pairing gain *and* silently biases the estimate) is a direct,
  testable criticism of that architecture, and TNCM-VAE's public code makes it a runnable baseline
  rather than a rhetorical one.

## Column A — Diffusion / generative models in options (all L1)

[Forecasting IV surfaces with generative diffusion models](https://arxiv.org/pdf/2511.07571) (Nov
2025, arbitrage-suppressing loss, public code — the natural SOTA baseline);
[Deep Learning Option Pricing with Market IV Surfaces](https://arxiv.org/abs/2509.05911) (VAE, 10-dim
latent, SPX 2018–2023); [VolGAN](https://www.tandfonline.com/doi/abs/10.1080/1350486X.2025.2471317)
(arbitrage-free surfaces, data-driven option-book hedging); TU Delft diffusion IV
generation/completion; [GBM-aligned diffusion for financial time series](https://arxiv.org/html/2507.19003);
Diffolio; CoFinDiff. These condition; they do not intervene.

## Column B — Causal diffusion in ML (imaging and general time series)

[Diff-SCM](https://arxiv.org/abs/2202.10166) (Sanchez & Tsaftaris, CLeaR 2022), DCM, CaTSG (causal
time-series generation across Pearl's ladder), causal diffusion autoencoders, Causal-Adapter. This
literature also states the open problem [`GAINS.md`](GAINS.md) targets: DDIM inversion is
*approximate*, and formal counterfactual identifiability would need "improved inverse operators or
diffusion models explicitly designed to recover exogenous noise."

## Column C — Causal-labelled work in finance that means something else

Causal optimal transport / adapted Wasserstein = non-anticipative ([Bulté & Pammer](https://arxiv.org/abs/2303.14085)
bridge the two senses); DML driver-identification studies; Hansen & Sokol's causal SDE theory
(never applied to derivatives).

---

## What this scan still does not cover

- **SSRN not searched** — the biggest hole, and the likeliest home of prior art.
- Paywalled databases (ScienceDirect, RePEc, Web of Science, JSTOR) not searched.
- **arXiv, Semantic Scholar, Hugging Face and alphaxiv all return 403 under this session's egress
  policy.** Everything except 2508.14577 is characterised from search snippets. Several of the
  claims above would change if the papers were read.
- No systematic q-fin listing sweep; keyword-mediated only. English only.
- Practitioner work at banks and funds is largely unpublished; G2's paired-sensitivity claim
  remains the most likely quiet prior art.

Revision 1 asserted an empty intersection after six searches. Four more searches and one paper
falsified it. **The base rate for "nobody has done this" claims in this document should be treated
as poor**, and the recovery-theorem check from [`PROPOSAL.md`](PROPOSAL.md) — still outstanding, and
bearing on the load-bearing T2 claim — should be assumed to carry the same risk.
