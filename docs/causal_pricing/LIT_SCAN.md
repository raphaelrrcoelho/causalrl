# Literature scan: causal diffusion × option pricing

> **Scope and confidence.** Six targeted web searches, July 2026. This is a *scan*, not a
> systematic review. Limitations are real and listed at the bottom — most importantly I could not
> search SSRN, which is where quantitative-finance preprints actually live, and two arXiv fetches
> returned 403 so the paper characterisations below come from search snippets rather than from
> reading the papers. **"I found no evidence of X" here does not mean "X does not exist."**
> Treat every novelty claim as a hypothesis with a stated confidence, not a finding.

## The short answer

The two literatures both exist, both are active, and **they do not touch.** Diffusion models are
being applied to options hard and successfully — entirely at Pearl's L1. Causal diffusion models
are a real and growing ML subfield — entirely in imaging and general time series. I found nothing
at the intersection, and nothing asking the identification question for pricing queries at all.

---

## Column A — Diffusion / generative models in options. Thriving, entirely L1.

| Work | What it does |
| --- | --- |
| [Forecasting IV surfaces with generative diffusion models](https://arxiv.org/pdf/2511.07571) (Nov 2025) | Conditional diffusion for one-day-ahead SPX implied-volatility surfaces, with a loss term that suppresses arbitrage violations. Public code. **The closest thing to a SOTA baseline for this proposal.** |
| [Deep Learning Option Pricing with Market IV Surfaces](https://arxiv.org/abs/2509.05911) | VAE compresses surfaces across strike/maturity to a 10-dim latent, then a network prices from it. SPX 2018–2023. |
| [VolGAN](https://www.tandfonline.com/doi/abs/10.1080/1350486X.2025.2471317) | Arbitrage-free IV surface generation; simulates co-movements of IV, index and VIX; used for data-driven hedging of option books. |
| [Diffusion probabilistic model for IV surface generation and completion](https://resolver.tudelft.nl/uuid:b1ecf501-7105-419f-9d42-0b0f6e2725e5) | Surface generation and missing-quote completion. |
| [Diffusion generative model for financial time series via GBM](https://arxiv.org/html/2507.19003) | GBM-aligned SDE formulation; generated paths usable directly in valuation. |
| [Diffolio](https://arxiv.org/html/2511.07014v2), CoFinDiff | Multivariate financial time-series diffusion; controllable generation. |

**The pattern, and it is uniform.** Every one of these is *forecasting, generation, or completion*.
They condition; they do not intervene. None of the search results surfaced any mention of Pearl,
structural causal models, `do`, abduction, or counterfactual identification. Conditioning a
diffusion on "high-vol regime" samples from `P(surface | regime = high)` — an observational
conditional — which is not `P(surface | do(regime = high))` whenever the regime is confounded with
anything else in the conditioning set. That distinction is the entire opening.

## Column B — Causal diffusion models in ML. Thriving, no finance.

| Work | What it does |
| --- | --- |
| [Diff-SCM](https://arxiv.org/abs/2202.10166) (Sanchez & Tsaftaris, CLeaR 2022) | Deep SCM where abduction is deterministic forward diffusion and intervention acts on the reverse process. **Imaging.** |
| DCM (Chao et al.) | Interventional and counterfactual inference with diffusion models. |
| CaTSG | Causal time-series generation formalised across Pearl's ladder, with backdoor-adjusted guidance. **The closest in spirit** — general time series, not derivatives. |
| Causal diffusion autoencoders, Causal-Adapter, visual disentangled diffusion AEs | Counterfactual generation, all vision. |

This literature also supplies the open problem [`GAINS.md`](GAINS.md) targets: DDIM inversion is
*approximate*, causing information loss under compound interventions, and achieving formal
counterfactual identifiability is stated to require "improved inverse operators or diffusion models
explicitly designed to recover exogenous noise."

## Column C — Causal methods that do exist in finance (and mostly mean something else)

1. **"Causal" as non-anticipative.** Causal optimal transport and adapted Wasserstein distance
   (Backhoff-Veraguas, Beiglböck, Pammer) mean *uses only information available at time t* — a
   filtration property, unrelated to Pearl. Neural-SDE calibration is described as "strongly
   related to causal optimal transport theory." **This is the dominant meaning of the word "causal"
   in mathematical finance and it is not our meaning.** [Bulté & Pammer](https://arxiv.org/abs/2303.14085)
   bridge the two senses and are the right citation for the disambiguation.
2. **Causal ML as driver identification.** DML / average-partial-effect work identifying causal
   drivers of market phenomena (e.g. options-implied risk appetite and liquidity as drivers of
   market troughs). Econometrics applied to markets — not pricing theory.
3. **SCMs applied to a financial market.** [Causal regime detection in energy markets with augmented
   time-series SCMs](https://arxiv.org/pdf/2511.04361) (Nov 2025) is the closest genuine Pearl-style
   SCM application to a market I found — but it does regime *detection*, not pricing or hedging.
4. **Causal SDE theory.** [Hansen & Sokol](https://arxiv.org/abs/1304.0217) (EJP 2014) defines the
   post-intervention SDE and proves it is the limit of post-intervention Euler SEMs; Boeken & Mooij
   extend to dynamic SCMs. Theory only — I found no application to derivatives.

---

## The gap, stated precisely

**A × B is empty.** Nobody appears to have built an interventional/counterfactual diffusion model
for derivatives, and — more importantly for this proposal — nobody has asked *which pricing queries
are identified at all*. The three-rung structure in [`PROPOSAL.md`](PROPOSAL.md) has no competitor
because the question has not been posed in this literature.

Three consequences for positioning:

- **The baseline is clear and strong.** Compete against arXiv 2511.07571 (conditional diffusion,
  arbitrage-aware, public code) on counterfactual queries, not on next-day surface forecasting —
  where it will win, and should.
- **The novelty is the question, not the tooling.** Diff-SCM already does abduction-by-inversion;
  CaTSG already spans Pearl's ladder for time series. Porting either to options is engineering. The
  contribution is identification: *which* pricing queries admit an answer, and what the answer's
  epistemic status is.
- **The terminology collision is confirmed and serious.** Column C.1 means half a
  mathematical-finance audience will read "causal diffusion" as "non-anticipative diffusion."
  Disambiguate in the abstract, not in a footnote.

## What this scan did not cover

Stated plainly, because the gaps are where a novelty claim would die:

- **SSRN was not searched**, and it is the primary venue for quant-finance preprints. This is the
  biggest hole.
- Paywalled databases (ScienceDirect, RePEc, JSTOR, Web of Science) were not searched.
- No systematic sweep of arXiv q-fin listings; results are keyword-search-mediated.
- `arxiv.org` returned **403** on both direct fetches, so 2511.07571 and others are characterised
  from search snippets — I have not read them.
- English-language only.
- **Practitioner work at banks and funds is largely unpublished**, and internal model-risk systems
  are exactly where someone may already compute paired counterfactual sensitivities without
  publishing it. G2 in [`GAINS.md`](GAINS.md) is the claim most likely to be quietly prior art.

The residual due-diligence list in [`PROPOSAL.md`](PROPOSAL.md) — the recovery-theorem literature
above all — remains outstanding and is not addressed by this scan.
