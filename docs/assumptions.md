# Assumption Glossary

Every causalrl `Certificate` records the assumptions it consumed as typed `Assumption` entries
(`name`, `params`, `checkable`, `diagnostic`). This page explains each one: what it means, whether
it is checkable from data, and how a violation is handled. The library is one-sidedly honest —
outside a supported class it returns a **hedge** or a typed exception, never a silent point estimate.

## Epistemic kinds

A certificate's `kind` is never conflated across these three:

- **`IDENTIFIED`** — a point claim backed by identification under the stated graph/assumptions.
- **`BOUNDED`** — partial identification: an interval valid under an explicit sensitivity budget.
- **`EMPIRICAL`** — simulation/sample evidence only, with no identification guarantee.

## Assumptions

### `backdoor`
The adjustment set blocks all back-door paths from treatment to outcome. `params.adjustment_set`
lists the covariates. Checkable given the graph (the graph is the witness); the numeric estimate
additionally assumes correct nuisance models (relaxed by the doubly-robust `aipw` / `dml` methods).

### `overlap` (positivity)
Every unit has propensity bounded away from 0 and 1 (or, for importance sampling, a finite effective
sample size). `params.eps` / `params.min_ess_fraction` set the threshold; `diagnostic` reports the
observed extremes / ESS. Checkable — a violation **downgrades to a hedge** (`overlap-violation`)
rather than returning an unstable inverse-weighted estimate.

### `MSM` (marginal sensitivity model)
Unmeasured confounding shifts the true propensity from the nominal one by at most an odds-ratio
`gamma >= 1` (Tan's model). `params.gamma` is the budget. Not checkable from data — it is a stated
sensitivity parameter; the certificate is `BOUNDED` and the interval widens monotonically in
`gamma`, collapsing to the point estimate at `gamma = 1`.

### `logged-propensities`
The importance weights / nominal propensities used for off-policy evaluation are the true logging
probabilities (up to the `MSM` budget when one is declared). Not checkable from the logs alone.

### `moment-condition`
The target functional's moments exist (e.g. a finite mean). Checkable via a heavy-tail diagnostic;
when it fails, the mean target **downgrades** to a valid quantile/tail target with the downgrade
recorded (`hedge.downgraded_from`).

### `mi-cap` / pivotality
A cap on how much a hidden confounder can move the decision, expressed as a mutual-information or
odds-ratio budget (the pivotality layer). `BOUNDED`; `certify_decision` reports the tipping `gamma`
at which the decision would flip.

### `selection-nodes-S` (transport)
The named selection nodes mark exactly where the source and target regimes differ (population or
mechanism). The transport formula / regret certificate is valid when the marked diagram is correct;
`Regime` selection nodes are the witness. Non-transportable queries hedge.

### `quantile-sketch`
A streamed quantile is answered by a Greenwald–Khanna sketch whose true rank is within `epsilon * n`
of the requested rank. `params.epsilon` is the guaranteed rank-error budget, recorded in provenance
so the approximation travels with the claim.

### `fqe-model`
A fitted-Q-evaluation off-policy value estimate. Model-based with no identification guarantee under
hidden confounding, so the certificate is `EMPIRICAL`; upgrade to `certify_effect` or bound it with
`certify_policy` when the structure licenses it.

## Provenance

Every certificate also records reproducibility metadata (`Provenance`): the library version, seeds,
a data fingerprint, a graph hash, and a timestamp — so a claim can be reproduced and audited.
