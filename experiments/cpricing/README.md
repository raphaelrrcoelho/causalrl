# cpricing — causal derivative pricing (feasibility probe)

Supporting code for [`docs/causal_pricing/PROPOSAL.md`](../../docs/causal_pricing/PROPOSAL.md),
"A Price Is Not a Counterfactual".

This directory holds the probe that gates the proposal, not research results. `poc_ladder.py`
checks that the shipped `causalrl` stack can actually carry the three identification rungs the
proposal claims:

| Claim | What it checks | Rung |
| --- | --- | --- |
| C1 | Exact path abduction on an Euler-discretised diffusion built with `build_unrolled_scm`, then `do(regime := flipped)` re-rolling the same Brownian increments | `IDENTIFIED` |
| C2 | The shipped Tan MSM kernel solves the same program as a bounded-pricing-kernel (gain-loss) price bound, against an independent brute-force reference | `BOUNDED` |
| C3 | `certify_mean` refuses the mean of an infinite-variance P&L sample and downgrades to a median certificate | hedge |

```bash
uv run python experiments/cpricing/poc_ladder.py
```

Last recorded output: [`POC_OUTPUT.txt`](POC_OUTPUT.txt). Each claim carries an assertion, so the
script fails loudly rather than printing a wrong number.

## poc_gains.py — do the mechanics buy *accuracy*?

Supports [`docs/causal_pricing/GAINS.md`](../../docs/causal_pricing/GAINS.md). Measures the causal
estimators against non-causal baselines rather than arguing for them.

| Claim | Result |
| --- | --- |
| G2 | Paired counterfactuals vs. independent resampling at equal budget: **1830×** variance ratio on a vega-shaped query, **1.13×** on a regime flip. The gain scales inversely with intervention size. |
| G1 | "Inversion error compounds over long paths" — **refuted**, 0.98× growth over a 20× horizon range. Errors average out as `O(eps*sqrt(T))`. |
| G1b | What approximate abduction actually costs: the pairing gain collapses `1850x -> 0.5x` as `eps` grows, and the estimate becomes silently **biased** (−3.43 vs a true −0.080). |

```bash
uv run python -m experiments.cpricing.poc_gains
```

Output in [`GAINS_OUTPUT.txt`](GAINS_OUTPUT.txt). The `eps` in G1/G1b is *assumed*, not measured
from a trained diffusion — calibrating it is the experiment that decides the claim.

Finance vocabulary is confined to this directory and to `docs/causal_pricing/` — `tools/generality_lint.py`
keeps it out of `src/causalrl` (invariant I7).
