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

Finance vocabulary is confined to this directory and to `docs/causal_pricing/` — `tools/generality_lint.py`
keeps it out of `src/causalrl` (invariant I7).
