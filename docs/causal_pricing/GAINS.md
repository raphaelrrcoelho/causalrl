# Where the gains are

## Accuracy claims for causal pricing against non-causal baselines — measured, refuted, and proposed

> Companion to [`PROPOSAL.md`](PROPOSAL.md). The proposal buys *honesty* (an identification
> theory plus certificates). This document asks the harder question: **where does causal
> machinery beat a non-causal diffusion on a metric?** Two claims are measured here
> (`experiments/cpricing/poc_gains.py`, output in `GAINS_OUTPUT.txt`); two of my starting
> hypotheses were refuted by those measurements and are recorded as such; the rest are
> proposed with metrics and baselines, and are not yet evidence.

## The frame: do not compete in-distribution

Start with the concession, because getting it wrong is how this kind of project dies.

**A causal model will not beat a well-tuned non-causal diffusion at fitting a volatility surface
or at in-distribution scenario generation.** For pure prediction under a fixed data-generating
process, causal structure is at best a mild inductive bias and at worst a constraint that costs
you fit. `causalrl`'s own README says this in its own domain: "the wins are confined to the
confounded / offline / transfer regime by design."

So the claim is not "causal is better." It is: **there are four query classes where the
non-causal baseline is not merely worse but structurally unable to answer, and those four are
where most of the money and most of the risk actually live.** Each is stated below with a
baseline, a metric, and a status.

---

## G2 — Paired counterfactuals: measured 1850×, and only where I now understand it to be

**Status: MEASURED.** The strongest result in this document, and it is narrower than I first
claimed.

Almost every model-risk and sensitivity question has the form `E[f(theta') - f(theta)]`. A
generative path model that cannot abduct must estimate this by drawing two independent sample
sets and differencing the means — it can *condition* on a regime and sample, but it cannot hold
the realized noise fixed across the change. Abduction can: recover the increments, `do(theta')`,
re-roll the *same* increments, average the per-path difference. Since
`Var(A - B) = Var(A) + Var(B) - 2Cov(A, B)`, the gain is whatever correlation the pairing induces.

At an equal path budget (n = 8000, 50 steps):

| Query | Non-causal SE | Paired SE | Variance ratio |
| --- | --- | --- | --- |
| Vega-shaped: vol 0.20 → 0.2020 | 2.15e-2 | 5.03e-4 | **1830×** |
| Regime flip: vol 0.15 → 0.45 | 1.17e-1 | 1.10e-1 | **1.13×** |

**My first guess was that pairing is a broad win. It is not.** The gain scales inversely with
the size of the intervention: enormous for a derivative-shaped query, worth essentially nothing
for a stress test. That is not a disappointment, it is the scoping rule — and it says exactly
which product to build. A 1830× variance ratio is a 1830× compute multiple: the same standard
error from ~1800× fewer paths. On a desk that recomputes Greeks across a book intraday, that is
the difference between an overnight batch and an interactive one.

**Is this novel?** The mechanism is not — this is common random numbers, which is as old as Monte
Carlo, and pathwise Greeks (Broadie–Glasserman) already exploit it in *analytic* models. **The
novelty is that learned path models cannot currently do it at all.** You cannot hold a trained
diffusion's noise fixed across a change in conditioning in a structurally meaningful way. Every
neural-SDE, market-generator, and diffusion-based scenario engine currently pays the full
independent-sampling variance on every sensitivity it computes. Making generative path models
CRN-capable — via structural invertibility rather than approximate inversion — is the contribution,
and G1b is why the distinction matters.

---

## G1 — Two hypotheses, one refuted, one confirmed and sharper

**Status: first version REFUTED by measurement; replacement MEASURED.**

Diffusion counterfactuals abduct by DDIM inversion, which the counterfactual-diffusion literature
states plainly is *approximate*, with information loss under compound interventions; the same
literature names "improved inverse operators or diffusion models explicitly designed to recover
exogenous noise" as an open problem. Financial paths are long (T = 50–250 steps), so my hypothesis
was: **per-step inversion error compounds over the path, and exact inversion wins by more the
longer the horizon.**

**That is false, and the measurement says so unambiguously.** Injecting a per-step error `eps` and
measuring relative terminal counterfactual error:

| steps | constant vol | state-dependent vol |
| --- | --- | --- |
| 10 | 3.67e-4 | 3.83e-4 |
| 50 | 3.58e-4 | 3.54e-4 |
| 200 | 3.62e-4 | 3.52e-4 |

Flat. Growth from 10 to 200 steps: **0.98×**. The reason is elementary once seen: log-price is a
*sum* of increments, so independent per-step errors average out — terminal error is `O(eps·sqrt(T))`
in the horizon, i.e. constant in the number of discretisation steps. I added a leverage-effect
local-vol model specifically to create state feedback, expecting amplification; the penalty was
**1.0×**. The compounding story is dead and should not appear in any writeup.

**What replaced it is sharper.** The real cost of approximate abduction is that it destroys
precisely the regime where G2's gain lives. Inversion error injects independent noise into the
counterfactual path, decorrelating it from the factual — and the pairing gain *is* that
correlation. Sweeping `eps` on a 1% vol bump:

| inversion `eps` | paired SE | variance ratio | estimate |
| --- | --- | --- | --- |
| 0 (exact) | 5.10e-4 | **1850×** | −0.08035 |
| 1e-4 | 5.10e-4 | 1850× | −0.08035 |
| 1e-3 | 5.12e-4 | 1836× | −0.08031 |
| 1e-2 | 7.15e-4 | 942× | −0.07973 |
| 1e-1 | 4.50e-3 | 23.7× | −0.12137 |
| 1.0 | 3.23e-2 | **0.5×** | **−3.42767** |

Two failure modes, not one. The variance advantage collapses — and by `eps = 1` the paired
estimator is *worse than independent sampling*. Worse, the estimate becomes **biased**: −3.43
against a true value near −0.080, a factor of 40, reported with a confident-looking standard
error. **Approximate abduction does not degrade gracefully; it returns a wrong number quietly.**

So the argument for exact structural inversion is not "long paths." It is: *approximate abduction
cannot ask small questions at all*, and small questions are what Greeks, vegas, and model-risk
sensitivities are. The method follows directly — use per-step conditional normalizing flows
(`ConditionalFlowMechanism`, already in the repo, already exposing `invert`) as the transition
mechanism, so inversion is exact by construction rather than approximate by optimisation.

**Caveat that gates the claim:** `eps` here is *assumed*, not measured from a trained model. The
decisive experiment (E6) is to calibrate real DDIM inversion error on a trained path diffusion and
locate it on that table. If real `eps` lands at 1e-4, diffusion models are fine and this whole
line collapses to a footnote. If it lands at 1e-2 or worse, the flow-based mechanism is the only
way to compute a Greek. **I do not know which, and neither does the literature — that is the
experiment.**

---

## G3 — Regime transfer: causal transport vs. distributionally-robust deep hedging

**Status: PROPOSED. Not measured. Strongest applied claim.**

The baseline here is not a strawman, which is what makes it worth doing. Deep hedging (Buehler et
al.) is the SOTA learned hedger, and its failure mode is *documented by its own literature*:
recent work shows standard deep hedging is "highly vulnerable to small perturbations in the input
distribution," and a 2026 paper names **"regime fragility"** outright. The current best fix is
Wasserstein-ball adversarial training.

The causal argument is a specific one: **distributional robustness hedges against every shift of a
given size; causal transport hedges against the shift that actually occurred.** A Wasserstein ball
is structure-agnostic and therefore conservative — it pays for perturbations that no market
produces. A selection diagram encodes *which mechanisms changed and which did not* (rates moved,
the vol-of-vol did not; the skew re-priced, the correlation structure held), and `transport_formula`
/ `transported_effect` reweight only along the changed mechanisms.

- **Metric:** terminal-P&L CVaR on a held-out regime, at matched training data.
- **Arms:** vanilla deep hedging / Wasserstein-DRO deep hedging / causally-transported hedging.
- **Prediction, and it is falsifiable:** causal transport wins when the shift has exploitable
  structure and loses when it does not. If the regime change is unstructured noise, DRO should win
  and I would expect it to. **The phase diagram — gain against degree of structure — is the
  finding either way**, and the repo's M2 result already produced exactly this shape of diagram in
  a synthetic setting.

---

## G4 — Counterfactual data augmentation for hedging

**Status: PROPOSED. Not measured.**

Real market history is short and precious; simulators generate unlimited paths that miss real
microstructure and real tail dependence. Abduction offers a third option: take *real* historical
paths, recover their increments, intervene on a regime latent, re-roll. The result keeps the
realized noise structure — actual clustering, actual jump timing — while multiplying the regimes
it is observed under.

The gap is well posed in the RL literature: historical market replay "cannot provide policy
training what it really needs — counterfactual feedback about how prices would have moved had the
agent traded differently." Counterfactual data augmentation is established in RL generally; it has
not been applied to hedging with exact structural abduction.

- **Metric:** hedging CVaR on held-out real periods.
- **Arms:** train on simulated paths / train on real paths only / train on real + counterfactual
  augmentations.
- **Baseline to beat:** causal-optimal-transport data augmentation already outperforms non-causal
  generative models on portfolio tasks, so the direction has precedent — and that precedent is the
  bar, not a friend.

---

## G5 — Zero-shot novel interventions

**Status: PROPOSED. Not measured. Highest risk, highest payoff.**

A conditional generative model can only interpolate conditions it saw in training. A correctly
factorised causal model can compose an intervention it never saw, because the mechanisms are
modular. In markets this is testable against a real natural experiment: train on rate regimes
observed 2010–2021, evaluate on the 2022 hiking cycle — a genuine out-of-support intervention that
actually happened, with free data.

- **Metric:** pricing and hedging error on the unseen regime, at matched data budget.
- **Honest risk:** this is where causal claims most often fail in practice, because the
  factorisation has to be *right*, and financial causal structure is contested. A negative result
  here is likely and should be reported.

---

## G0 — A terminology landmine to defuse on page 1

Mathematical finance already uses "causal" to mean something else. **Causal optimal transport** and
the **adapted/causal Wasserstein distance** (Backhoff-Veraguas, Beiglböck, Pammer, and the
robust-finance line) mean *non-anticipative* — a transport that at time `t` uses only information
available at `t`. That is a statement about filtrations, not about interventions, and it is
unrelated to Pearl's sense.

Any paper from this proposal that says "causal" without disambiguating in its first paragraph will
be read by half its referees as being about adapted transport. Two consequences: the disambiguation
is mandatory, and Bulté & Pammer's work relating optimal transport to Pearl-style causal models is
the natural bridge to cite — turning the collision from a hazard into a related-work anchor.

---

## Summary

| # | Claim | Baseline | Status |
| --- | --- | --- | --- |
| G2 | Paired counterfactuals cut variance ~1830× on derivative-shaped queries | independent resampling | **measured**; scope limited to small perturbations |
| G1 | Inversion error compounds over long paths | — | **refuted** (0.98× growth over 20× horizon) |
| G1b | Approximate abduction destroys the pairing gain *and* biases the estimate | DDIM-style inversion | **measured** with assumed `eps`; real `eps` is experiment E6 |
| G3 | Causal transport beats DRO deep hedging on structured regime shift | deep hedging + Wasserstein DRO | proposed |
| G4 | Counterfactual augmentation of real paths beats simulated-only training | causal-OT augmentation | proposed |
| G5 | Zero-shot pricing under an unseen rate regime | conditional generative model | proposed, high risk |

The honest summary is that **one gain is measured and large but narrower than advertised, one
hypothesis is dead, and the applied claims are untested.** That is a better starting position than
six plausible claims and no measurements, and the two refutations cost an afternoon rather than a
referee report.
