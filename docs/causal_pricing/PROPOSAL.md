# A Price Is Not a Counterfactual

## Certified identification for derivative pricing under causal diffusions — a proposal for the causalrl 2.x track

> Status: **proposal, pre-due-diligence.** Feasibility probe run and passing
> (`experiments/cpricing/poc_ladder.py`, output in `POC_OUTPUT.txt`). Novelty claims below are
> from a *preliminary* search only; the residual checks are listed explicitly in
> "Positioning" and must be completed before any of this is written up.

## The pitch

Quantitative finance runs on a conflation that causal inference has a precise vocabulary for and
finance does not.

A derivative price is an **observational** object. It is a conditional expectation under a measure
`Q` reverse-engineered from quoted prices; Breeden–Litzenberger recovers the risk-neutral marginals
from the surface, and everything downstream is `P(payoff | today's quotes)`. In Pearl's hierarchy
it is L1, and it is *superbly* identified — that is what calibration is for.

Almost nothing anyone *does* with a pricing model is L1:

- "What is my P&L if I hedge at 20 vol instead of 25?" — L3. A counterfactual on the realized path.
- "What does my book do if the Fed hikes 50bp?" — L2. An intervention on a latent driver.
- "What if my own hedging flow moves the underlying?" — a cyclic, feedback query.
- "Is my calibrated local-vol surface still right after the regime shifts?" — the Lucas critique,
  restated for derivatives.

These are answered every day with machinery that only licenses L1 answers, and the failure is
silent: the model returns a number either way. **The thesis of this proposal is that derivative
pricing has a three-rung identification structure that has never been stated, and that each rung is
mechanically detectable, so the model can be made to say which rung it is on instead of always
returning a number.**

**Central thesis (one sentence):** *pathwise counterfactual queries on a diffusion are
point-identified exactly when the discretised transition is invertible in its noise; any query
requiring the physical measure or the pricing kernel is partially identified, with a sharp interval
that is literally the same optimisation program as a no-good-deal bound; and under heavy tails the
target functional itself fails to exist and must be downgraded rather than reported — and a library
can tell you which of the three you are in, automatically.*

This is at once a causal-inference paper (interventional semantics of diffusions, extending the
SDE→SCM line), a mathematical-finance paper (a formal identity between sensitivity analysis and
good-deal bounds), and a risk-management paper (model risk as a certificate rather than a number).

---

## Why this repo, specifically

This is not a "causal inference is interesting, finance is interesting" pitch. Four pieces have to
be in one place for the programme to be executable, and `causalrl` is the only stack I know of that
has all four:

1. **`build_unrolled_scm`** (`src/causalrl/scm/unrolled.py`) — a controlled dynamical system with
   shared latents, unrolled into an ordinary DAG carrying the full Pearl ladder. An
   Euler–Maruyama discretisation of an SDE *is* this object. The causal license is
   Hansen & Sokol (EJP 2014): the post-intervention SDE is the uniform-in-probability limit of
   post-intervention structural equation models built on the Euler scheme. The library
   accidentally shipped a causal diffusion engine.
2. **Invertible continuous mechanisms** (`scm/continuous/`) — `LocationScaleMechanism` and
   `ConditionalFlowMechanism` both expose `invert`, and `abduct_location_scale` recovers exogenous
   noise *exactly*, licensing `kind=IDENTIFIED`; the amortized-VI and NUTS paths downgrade to
   `EMPIRICAL`. This is precisely the identified/not-identified boundary for pathwise
   counterfactuals, already drawn and already enforced.
3. **The sharp MSM kernel** (`identification/bounds.py::_fractional_extreme`) — the exact optimum of
   a self-normalised linear functional over a per-unit weight box. That is *the same program* as a
   bounded-pricing-kernel price bound. Plus `tipping_gamma`, `pivotality_certificate`, and the
   pure-numpy two-phase simplex in `magames/_lp.py` for the constrained version.
4. **The certificate layer** (`certify/`) — `Kind ∈ {IDENTIFIED, BOUNDED, EMPIRICAL}`, consumed
   assumptions, witness-or-hedge, provenance. The honesty discipline that makes "which rung" a
   machine-checkable output rather than a discussion section.

Plus two that turn out to matter more than expected: **`bounds/continuous.py`**, which already
implements Hill tail-index diagnostics and *refuses to report a mean* on an infinite-variance
sample (rung 3 is already built, it just has not been pointed at P&L), and
**`experimental/cyclic/`**, whose equilibrium-vs-unrolling comparator is the natural home for
price-impact feedback.

Design constraint, noted up front: `tools/generality_lint.py` bans the words *price*, *market*,
*trader*, *portfolio* from the public surface of `src/causalrl`. All finance vocabulary lives in
`experiments/cpricing/` and these docs; only domain-agnostic primitives go into core. This is the
same split the equilibrium-counterfactuals flagship used and it is the right one.

---

## Three theory targets

### T1 — Pathwise counterfactuals are identified iff the transition is noise-invertible

For a diffusion `dX = mu(X, theta) dt + sigma(X, theta) dW` discretised on a grid, the
counterfactual query "what would this *realized path* have done under `do(theta := theta')`" is
point-identified from the observed path alone iff each step's map from noise to next state is
invertible — which for the location-scale (and conditional-flow) class it is, in closed form.
Under a non-invertible mechanism (stochastic volatility with unobserved variance, jumps with
unidentified marks) the exogenous is only partially recovered and the counterfactual is
`EMPIRICAL`, with a posterior-predictive check attached as a falsifiable assumption.

The result is a clean theorem for a claim practitioners make constantly and never justify: *P&L
attribution across models on the same historical path is a counterfactual, and it is only
well-defined when the model's noise is recoverable.* Attribution across non-invertible models is
not a computation with a wrong answer; it is a query without a unique one.

**Corollary worth its own section — pathwise Greeks are counterfactual derivatives.** The
Broadie–Glasserman pathwise method differentiates the payoff holding the random draw fixed. In
Pearl's language that is exactly abduct → `do(theta + dtheta)` → re-roll → differentiate. A
finite-difference Greek from independent Monte Carlo runs is L2 (a derivative of an interventional
*mean*); a pathwise Greek is L3. They agree in expectation and disagree everywhere else — and the
*distribution* of the pathwise sensitivity, which is what model risk actually is, has no L2
counterpart at all. T1 gives that distribution an identification status.

### T2 — The marginal sensitivity model IS a no-good-deal bound (the bridge result)

Vanilla options identify `Q`. They do not identify `P`, and they do not identify the pricing
kernel. Every genuinely interventional question — expected hedged P&L, the physical distribution of
a book, whether a trade is worth doing — is a `P`-question, so it is *partially identified*, and
the identified set is determined by how far the kernel may stray from the calibrated one.

That is exactly Tan's marginal sensitivity model with the labels changed. MSM bounds the odds ratio
of the true to the nominal propensity by `Gamma`; a gain-loss bound (Bernardo–Ledoit) bounds the
ratio of the true to the benchmark pricing kernel by `L`. Both compute the sup and inf of a
self-normalised linear functional over a bounded-density-ratio set. Same program, same sharp
threshold solution.

**Verified numerically** (`claim_2_good_deal_equivalence`, agreement to `2.7e-14`): the shipped
`ipw_sensitivity_bounds` at sensitivity `Gamma` with uniform nominal propensity `e` reproduces the
bounded-kernel bound at kernel-ratio cap

```
L(Gamma, e) = (1 + odds*Gamma) / (1 + odds/Gamma),    odds = (1 - e)/e
```

which sweeps `[1, Gamma^2)` as `e` runs from 1 to 0 — confirmed in the probe output
(`e=0.001 -> L=3.994` against `Gamma^2 = 4`). So `Gamma = sqrt(L)` is the asymptotic dictionary.

**The honest limit, stated plainly.** This identity holds for the *unconstrained* case: a single
payoff, no requirement that the measure also price the hedging instruments correctly. A real
good-deal bound adds martingale/replication constraints, turning a box program into a constrained
linear-fractional one. That is not a wall — the Charnes–Cooper transform converts a linear-fractional
objective with linear constraints into an LP, and the repo already ships a pure-numpy two-phase
simplex (`magames/_lp.py`) built for exactly this shape of problem. But it is genuine work and the
claim must be stated as "identical in the unconstrained case, extended by Charnes–Cooper in the
constrained one", not glossed.

The operational payoff is `tipping_gamma` translated into finance: **the gain-loss ratio at which a
trade's sign flips.** Not "the model says this is worth 3bp" but "this is worth doing unless you
believe a Sharpe ratio of 1.4 is available in this market" — a falsifiable, economically
interpretable statement of exactly how much model trust a position requires. To my knowledge, no
risk system reports this.

### T3 — Certified failure: heavy tails, feedback, and calibrated-model invalidity

Three distinct non-identification regimes, each with a runtime diagnostic:

**(a) The functional does not exist.** Short-vol and short-gamma P&L are heavy-tailed. If the Hill
tail index is at or below 2 the variance is infinite and the sample mean is not a meaningful
summary; below 1 the mean itself is undefined. `certify_mean` already detects this and downgrades
to a median certificate. The probe illustrates the stakes better than an argument would: on a
simulated short-vol book the naive sample mean is **-0.198** (a losing strategy) while the certified
median is **+0.049** (a winning one), with `hill_alpha = 0.705`. Both numbers are "correct"; only
one is a summary of anything. The library refuses the mean and says why.

**(b) Feedback breaks the fixed point.** When hedging flow moves the underlying (Frey–Stremme,
Schönbucher–Wilmott, Platen–Schweizer), the pricing problem is cyclic. `compare_equilibrium_unrolling`
plus `stability_margin` answer "is the equilibrium price the right counterfactual object here, or
does the unrolled trading dynamic diverge from it?" — the identical T1/T2/T3 trichotomy the
equilibrium-counterfactuals flagship already built, applied to a second domain. That reuse is a
feature: it argues the trichotomy is structural rather than bespoke.

**(c) Calibration is not identification.** A Dupire local-vol surface reproduces today's
`Q`-marginals exactly and has, by construction, no interventional content: intervene on rates or
flip the volatility regime and its prediction carries no causal warrant. Provocative headline
version, and I believe it is defensible: **local volatility is L1-complete and L2-empty.** This is
the Lucas critique for derivatives, and unlike the macro version it is directly testable — you can
calibrate, intervene, and measure the error against a structural model that answers the
interventional query correctly.

---

## What the library needs (three additions, all small, all domain-agnostic)

The feasibility probe ran using only shipped code plus ~40 lines of hand-written path inversion.
That inversion is the one real gap:

1. **`abduct_path`** (`scm/unrolled.py`) — recover the per-step exogenous from an observed
   trajectory by inverting an invertible transition, generalising the single-mechanism
   `abduct_location_scale` to an unrolled chain. Returns a `NoisePosterior` and the exactness flag
   that decides `IDENTIFIED` vs `EMPIRICAL`. This is the T1 front door, and it is maybe 60 lines.
2. **Constrained fractional bounds** (`identification/bounds.py`) — Charnes–Cooper reduction of the
   self-normalised box program with additional linear equality constraints, dispatched to the
   existing `magames/_lp.py` simplex. This is the T2 front door; `_fractional_extreme` becomes its
   unconstrained fast path.
3. **`counterfactual_sensitivity`** — the pathwise-derivative distribution (abduct, perturb, re-roll,
   difference), returning a certificate whose `Kind` is inherited from the abduction's exactness.
   The T1 corollary, made callable.

Everything else — MSM kernels, `tipping_gamma`, Hill diagnostics, conformal intervals, the cyclic
comparator, the certificate type — is already in `main`.

---

## Experimental programme

**E1 — The identification ladder on one book.** A single short-call delta-hedged position, three
queries, three certificates: pathwise counterfactual P&L under a regime flip (IDENTIFIED, exact
abduction); expected P&L under kernel ambiguity (BOUNDED, with the gain-loss interval); tail-risk
summary (hedged, mean downgraded to median). The probe is already a rough draft of this. It is the
figure that explains the whole paper.

**E2 — Gain-loss bounds against the literature's benchmarks.** Reproduce published
Cochrane–Saá-Requejo and Bernardo–Ledoit bounds for standard incomplete-market examples via the MSM
kernel, demonstrating the T2 identity numerically on cases where the finance literature has an
independent answer. This is the credibility gate for T2, and it either works or kills the claim
early — which is the point of running it second.

**E3 — Local volatility is L2-empty.** Calibrate Dupire to a surface generated by a known
structural model (Heston, or a rough-volatility model). Intervene: shift rates, flip the vol
regime, change the correlation. Compare the local-vol model's post-intervention prediction to the
structural model's — which is the ground truth by construction. Quantify the causal error and show
the certificate flags it *without being told* the calibration was reduced-form. The economics- and
practitioner-facing centrepiece.

**E4 — Feedback hedging.** A delta-hedger large enough to move the underlying, as a cyclic SCM.
Show `stability_margin` predicting the regime where the equilibrium price and the unrolled hedging
dynamics diverge, and the certificate hedging exactly there. Direct reuse of the eqcf comparator.

**E5 — Real data, honestly scoped.** Deribit is the right primary stage: options, perpetual
funding, and full order book, free and public, with genuinely heavy tails so the T3 machinery bites
rather than being a technicality. SPX/VIX from CBOE's free tier as a robustness check. The
empirical claim is deliberately modest — "the certificates fire on real surfaces and the tipping
gain-loss ratios are economically interpretable" — because the theory is the contribution and an
overreaching empirical claim would be the easiest thing for a referee to kill.

---

## Positioning, and what I have *not* yet checked

Preliminary searching found the following, and no more:

- **Hansen & Sokol (EJP 2014)**, "Causal interpretation of stochastic differential equations", is
  the foundation for the causal-diffusion side, and its Euler-scheme limit theorem is precisely the
  license `build_unrolled_scm` needs. Boeken & Mooij's dynamic SCMs (2024) and the
  Mooij–Janzing–Schölkopf ODE→SCM line extend it. **None of this literature has been pointed at
  derivative pricing.**
- **Causal diffusion generative models** exist and are active (Diff-SCM; CaTSG's causal time-series
  generation across Pearl's ladder; causal diffusion autoencoders), including finance-flavoured
  scenario generation. These are *generative-modelling* papers — they produce interventional and
  counterfactual samples. None of them asks the identification question, which is the gap this
  proposal occupies. The terminological collision with "diffusion" is worth confronting in the
  paper's first page rather than leaving to a confused referee.
- **The gain-loss literature** confirms the structural premise: restricting the best gain-loss
  ratio is equivalent to the existence of pricing kernels bounded and bounded away from zero — i.e.
  a bounded-density-ratio set, the MSM box. I found **no paper connecting this to the causal
  sensitivity-analysis literature.** T2 appears unclaimed, and it is the result I would most expect
  to be wrong about, because it is the one that would be most obviously valuable if true.
- **Partial identification in finance** exists (Hansen–Jagannathan bounds are, in substance, partial
  identification of the kernel) but is not framed causally, and I found nothing doing partial
  identification of *derivative* prices as a causal identification problem.

**Residual due diligence, mandatory before writing.** This list is short because the search was
short — treat the novelty claims above as hypotheses:

1. Search the mathematical-finance literature directly for MSM/Rosenbaum-style sensitivity applied
   to pricing kernels. These are the places a prior version of T2 would hide: Černý & Hodges'
   no-good-deal framework, Björk & Slinko, Cont's model uncertainty, and the Hobson-style
   robust-hedging school.
2. Read Hansen & Sokol in full, and check whether the post-intervention SDE construction already
   implies T1 for the invertible class — T1 may be a corollary of an existing theorem rather than a
   new one, in which case its contribution is the *identification boundary and the certificate*, and
   the paper must say so.
3. Check the econometrics-of-options literature (Aït-Sahalia, Jackwerth's pricing-kernel
   estimation, Ross's recovery theorem and Borovička–Hansen–Scheinkman's critique of it) — the
   recovery-theorem debate is *precisely* a `P`-from-`Q` identification argument, and it is the
   closest existing thing to this proposal's framing. It must be engaged with directly, not cited
   in passing. **This is the single highest-priority check**: if recovery-theorem work already
   frames the `Q`→`P` gap as formal identification, T2's novelty narrows to the sensitivity-model
   machinery and the certificate.
4. Check whether "pathwise Greeks are counterfactuals" has been said in the Malliavin-calculus
   literature (Fournié et al.) in different words.

**Risks.** T2 is the load-bearing claim and the recovery-theorem literature is where it is most
likely to be pre-empted — mitigation: even if the framing is anticipated, the sharp computable
bound plus tipping-gain-loss-ratio is an instrument nobody ships, and the paper degrades from
"new identification result" to "operationalisation", which is still publishable. T1 may prove to be
a corollary — mitigation: its value was always the boundary and the certificate. E3 risks being
read as "everyone knows local vol is reduced-form" — mitigation: everyone *says* it, nobody
measures the interventional error or gets a machine-checkable warning, and the quantified error in a
policy-relevant intervention is the finding.

**Venues.** The natural split is (1) the identification theory + certificates at **CLeaR** or
**NeurIPS** (the causal-inference audience for T1/T2); (2) the finance-facing paper — "Model risk as
an identification problem" — at **Quantitative Finance** or **Mathematical Finance**, carried by
T2/E2/E3; and (3) `abduct_path` + the constrained bounds feed the library and its software paper.
T2 alone, if it survives check (3) above, is the piece that gets cited outside its home field.

**Sequencing.** Weeks 1–2: the four residual checks, T2 first — it is the cheapest to kill and the
most valuable to keep. Weeks 2–5: `abduct_path` + `counterfactual_sensitivity` in core, E1 as the
worked example. Weeks 5–9: T2 proof + Charnes–Cooper constrained bounds + E2 against published
benchmarks. Months 3–5: E3 and the local-vol result. Months 5–7: E4, E5, first submission. If check
(3) kills T2's novelty, the programme still stands on T1 + T3 + E1 + E3 as "certified identification
for pathwise counterfactuals in derivative books" — narrower, still frontier, still worth doing.

---

## Feasibility evidence

`experiments/cpricing/poc_ladder.py`, run against `main`, output in `POC_OUTPUT.txt`:

| Claim | Result |
| --- | --- |
| C1 — exact path abduction + regime counterfactual on an unrolled-SCM diffusion | round-trip error **0.0**; noise recovery **1.5e-5** (float32) |
| C2 — shipped MSM kernel == bounded-pricing-kernel bound | agreement **2.7e-14**; `L -> Gamma^2` confirmed as `e -> 0` |
| C3 — heavy-tailed P&L downgraded | `hill_alpha = 0.705`; mean **-0.198** refused, median **+0.049** certified |

C1 also produced the object the proposal is ultimately about: a paired, per-path counterfactual P&L
distribution — mean **-11.86** with a 5–95% spread of **[-25.4, -2.6]**. The mean is what an
interventional analysis gives you. The spread is what model risk actually is, and it only exists at
L3.
