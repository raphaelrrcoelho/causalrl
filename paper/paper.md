---
title: 'causalrl: Assumption-aware causal reinforcement learning with honest certificates'
tags:
  - Python
  - causal inference
  - reinforcement learning
  - off-policy evaluation
  - sensitivity analysis
  - transportability
authors:
  - name: Raphael Coelho
    orcid: 0000-0000-0000-0000
    affiliation: 1
affiliations:
  - name: Independent researcher
    index: 1
date: 10 July 2026
bibliography: paper.bib
---

# Summary

`causalrl` is a Python library for **assumption-aware** causal decision-making and reinforcement
learning. It organises identification, partial identification, off-policy evaluation, transport, and
multi-agent equilibria around one idea: every inferential routine returns a serializable
`Certificate` that states its claim, an explicit *epistemic kind* — `IDENTIFIED` (point-identified),
`BOUNDED` (partial identification under a stated budget), or `EMPIRICAL` (simulation evidence only) —
the assumptions it consumed, and provenance for reproduction. The library is one-sidedly honest:
outside a supported class it returns a typed hedge or a weaker valid target, never a silently
unreliable point estimate. It spans the graph-level identification stack (the ID algorithm, general
transportability, intervention sets), the estimation stack (doubly-robust estimators, sharp
marginal-sensitivity-model bounds, conformal wrappers), a multi-agent core (typed populations and
robust-equilibrium certificates), and a streaming data plane that runs the same certificates at
simulator scale over a columnar trajectory log.

# Statement of need

Causal reinforcement learning sits between two mature toolkits that do not meet its needs. Causal
inference libraries provide identification and estimation but assume a fixed dataset and an analyst
in the loop; reinforcement-learning libraries provide scalable policy learning but treat off-policy
evaluation as a point estimate, ignoring the unmeasured confounding that pervades logged decision
data. Practitioners are left to bolt sensitivity analysis onto an RL pipeline by hand, with no shared
representation of *what was assumed* to reach a number.

`causalrl` addresses this gap with a single certificate type that makes the epistemic status of every
result explicit and machine-checkable, and with routines that refuse rather than overclaim. A user
can evaluate a policy offline under a declared confounding budget and read the *tipping* `gamma` at
which the decision would flip [@tan2006; @zhao2019]; obtain sharp bounds under an estimated propensity
model; transport a claim across calibrated environment configurations drawn from a
simulation-based-inference posterior [@bareinboim2016]; certify a single agent embedded in a fixed
population; and stream any of these certificates over a log larger than memory. Estimation uses
doubly-robust and cross-fitted estimators [@chernozhukov2018; @bang2005], confounded off-policy value
bounds follow [@kallus2020], streaming quantiles use a mergeable sketch with a hard rank-error bound
[@greenwald2001], and the identification core implements the ID algorithm and its transport
generalisations [@shpitser2006; @bareinboim2016]. Optional backends (PyTorch, JAX, d3rlpy) and
interop adapters (DoWhy, EconML, NumPyro/SBI, PettingZoo) are lazily imported, so the numpy core
stays dependency-light while scale and neural mechanisms are available when installed.

The library targets researchers and practitioners in causal RL, off-policy evaluation, and
sensitivity analysis who need results that carry their assumptions with them. It is tested against
analytic oracles, ships reproducible benchmarks, and follows semantic versioning; version 2.0
consolidates the certificate as the default return type of its inferential routines.

# Acknowledgements

We thank the open-source causal-inference and reinforcement-learning communities whose methods this
library implements and cites.

# References
