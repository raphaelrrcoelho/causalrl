# causalrl

[![CI](https://github.com/raphaelrrcoelho/causalrl/actions/workflows/ci.yml/badge.svg)](https://github.com/raphaelrrcoelho/causalrl/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-mkdocs--material-blue.svg)](https://raphaelrrcoelho.github.io/causalrl/)
[![PyPI](https://img.shields.io/pypi/v/causalrl.svg)](https://pypi.org/project/causalrl/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

Causal reinforcement learning: the 9-task causal RL taxonomy, made runnable.

`causalrl` supplies **the causal layer for sequential decisions, not a new trainer**: agents that
plan inside a given or learned structural causal model, graph algorithms for causal bandits,
demonstration environments, and explicit-latent SCMs with `see` (L1), `do` (L2), and
`counterfactual` (L3) queries, organised around the
[9-task taxonomy of causal RL](https://crl.causalai.net/). Train a policy however you like, then
hand its actions to `certify_policy`: it bounds whether the value improvement over the logging
policy survives hidden confounding, and abstains when it cannot.

Scope is explicit and enforced in code. Out-of-class identification queries raise
`NotIdentifiableError` with the witnessing hedge (or return `None` for the conservative helpers)
rather than guessing a formula. The two halves also sit at deliberately different maturities: the
**planners and environments are demo-scale** — tabular to modest function approximation, on
synthetic worlds built to isolate one failure mode each — while the **decision, certificate and
off-policy-evaluation layers run on real data**. The `examples/causal_mbrl_*.py` scripts are the
evidence: on NHEFS, LaLonde and Twins they fit an agent, call `.act()` per unit and certify the
resulting policy; on the Open Bandit Dataset and Coat they run the off-policy-evaluation and
sensitivity kernels (`certify_policy`, `msm_policy_value_bounds`) against measured ground truth. See
[Guarantees & Scope](https://raphaelrrcoelho.github.io/causalrl/guarantees/).

## Install

```bash
pip install causalrl            # core: graph, POMIS, tabular agents/environments
pip install "causalrl[torch]"   # + SCM sampling, neural mechanisms, Torch-backed demos
```

From a clone, for development:

```bash
uv sync --extra dev             # tests, lint, typing, notebooks
uv sync --extra docs            # local documentation site and API reference
```

The core graph, POMIS, tabular-agent, and tabular-environment surfaces do not require PyTorch;
SCM sampling, neural mechanisms, and structural-bandit environments do.
Full documentation: **<https://raphaelrrcoelho.github.io/causalrl/>**.

## Quickstart

A causal agent that conditions on its "intuition" beats a confounding-naive agent on the
Multi-Armed Bandit with Unobserved Confounders — even though both arms have identical
*interventional* means.

```python
from causalrl.agents.bandits import CausalThompsonSampling
from causalrl.envs.suite.mabuc import MABUCEnv

env = MABUCEnv(seed=1)
agent = CausalThompsonSampling(n_arms=2, n_contexts=2, seed=0)

obs, _ = env.reset(seed=1)
for _ in range(8000):
    action = agent.act(obs)
    _, reward, _, _, _ = env.step(action)
    agent.update(obs, action, reward)
    obs, _ = env.reset()
# CausalThompsonSampling -> ~0.75 reward/step; any confounding-naive policy is capped near 0.50,
# since both arms share an interventional mean.
```

**Offline, confounded — one agent routes it.** Given confounded logs, `CausalMBRLAgent` picks the
right causal planner (back-door-adjust an observed confounder, discover the structure first,
transport across a covariate shift, handle a continuous confounder, or plan a sequential regime)
behind one `fit → act` surface, and tells you the identification it relied on.

```python
from causalrl import CausalMBRLAgent
from causalrl.envs.suite.simpson_bandit import SimpsonBandit

env = SimpsonBandit(seed=3)                       # observed confounder Z on a back-door A <- Z -> Y
agent = CausalMBRLAgent(env.n_actions, graph=env.graph)
agent.fit(env.sample(50_000, seed=3))             # columnar {Z, A, Y} logs
agent.act({"state": 0})                           # -> 1, the interventional optimum
agent.explain()  # "CausalMBRLAgent(strategy=backdoor, adjustment_set={Z})"
# A confounding-naive marginal is fooled by Simpson's paradox and ships the worse arm.
```

Same class, other regimes: `graph=None` with `variables`/`tiers` discovers the structure first;
`transport=("W",)` carries the policy across a covariate shift; `continuous_confounder=True` fits a
function approximator over a continuous confounder; `horizon=…` plans a confounded sequential regime
(the medicine DTR). The wins are confined to the confounded / offline / transfer regime by design —
see the [causal-MBRL results note](docs/causal_mbrl_agent/RESULTS.md).

## What it does

| Task (taxonomy) | Capability | Key entry points |
| --- | --- | --- |
| Decision under confounding | Counterfactual Thompson sampling on the MABUC | `CausalThompsonSampling` |
| Confounded offline agent | One front-door → back-door / discovery / transport / function-approx / sequential | `CausalMBRLAgent` |
| Learned world model | Fit an SCM from logs, then act in it as a Gymnasium env | `fit_scm`, `orient`, `counterfactual_interval` |
| Learn the model while acting | Refit the I-MEC from the agent's own experiments; Thompson-sample over structure | `OnlineCausalMBRL` |
| 1 — Offline→online | Learn from confounded logs via causal bounds | `UCDTR`, `DOVI`, `DeepDeconfoundedQ` |
| 2 — Where to intervene | POMIS / MIS, incl. non-manipulable variables | `pomis`, `minimal_intervention_sets` |
| 3 — Counterfactual policy | Act on `E[Y_do(a) \| intent]` | `CounterfactualOptimalPolicy` |
| 4 — Transportability | Recover effects across domains | `transport_formula`, `transported_effect` |
| 5 — Causal discovery | PC / FCI structure learning | `discover`, `CPDAG` |
| 6 — Causal imitation | Imitability + confounded cloning | `is_imitable`, `CausalImitator` |
| 7 — Causal curriculum | Prerequisite-ordered skill learning | `causal_curriculum` |
| 8 — Reward shaping | Policy-invariant causal potentials | `causal_potential`, `q_learning` |
| 9 — Causal games | Influence diagrams + equilibria | `pure_nash_equilibria`, `CausalGame` |
| Identification | Complete ID / gID / sID / mz; partial-ID, sensitivity & decision certificates | `identify_effect`, `manski_bounds`, `certify_decision` |

A runnable example for every row is in the
[**Tour by Task**](https://raphaelrrcoelho.github.io/causalrl/tour/); end-to-end notebooks are in
[`examples/`](examples) and the [Tutorials](https://raphaelrrcoelho.github.io/causalrl/tutorials/).
Six [task guides](https://raphaelrrcoelho.github.io/causalrl/guides/) — five that certify a policy
and one that trains an agent online — are scripts in [`examples/guides/`](examples/guides) executed
end to end in CI.

## What the numbers say — a deliberate negative

Scored the way an RL practitioner scores things — `.act()` per unit, then **regret** against ground
truth — the causal *point estimates* do not win on real data, and the shipped examples print that
themselves rather than hiding it:

- **Twins** (`examples/causal_mbrl_twins.py`, 11,984 pairs with both potential outcomes): our
  policy reaches 0.8316 survival against the per-pair oracle's 0.8747 — **regret 0.0431**, the
  worst of the learned policies, and behind the trivial constant "always the heavier twin"
  (0.8358).
- **LaLonde** (`examples/causal_mbrl_lalonde.py`, priced by the NSW randomized experiment): the
  contextual policy enrols 73.5% of the population for **$476/person of regret**, where the
  marginal rule it is built from ("enrol everyone") leaves $0 — and its off-policy value from the
  observational logs, −$453/person, has the wrong sign outright.

Both examples then **abstain**: `certify_policy` refuses Twins at Γ≈1.07 and LaLonde at Γ≈1.10, and
on LaLonde the randomized experiment vindicates the refusal. That is the recorded finding, and it
is the positioning — **the defensible edge is the decision and certificate layer, not the number.**
Full write-up: [real-data results](docs/causal_mbrl_agent/REAL_DATA.md).

## How it compares

`causalrl` is **causal-RL-first**, where the established causal libraries are estimation-first:

- **DoWhy / EconML / CausalML** target treatment-effect estimation and the
  identify→estimate→refute workflow on i.i.d. data. They are mature, production-grade tools.
  `causalrl` instead targets *sequential decision-making*: intervention-set selection (POMIS),
  confounded offline-to-online RL, counterfactual policies, and causal curricula / shaping /
  games. Those are the parts of the Bareinboim taxonomy these libraries do not cover.
- For pure graph identification it overlaps with **Ananke / pgmpy / Y0**.

On the RL side it is a layer, not a competitor:

- **`d3rlpy` (and offline-RL libraries generally) train the policy**; `causalrl` does not
  reimplement any of that, and pairs with them instead. `src/causalrl/scale/d3rlpy.py` is the
  bridge in both directions — `to_mdp_dataset` hands a `ConfoundedTrajectoryDataset` to a d3rlpy
  algorithm, `policy_actions` reads the trained policy's greedy actions back, and `certify_fqe`
  wraps a fitted-Q evaluation as a certificate. `pip install causalrl[scale]`; see the
  [Scale guide](https://raphaelrrcoelho.github.io/causalrl/scale/).
- **What it adds on top of a trained policy** is the assumption those libraries' evaluators take
  for granted. Off-policy evaluation — importance sampling, doubly-robust, FQE — is valid only when
  the logged actions are unconfounded given the recorded state, and logs written by a human, a
  clinician or a legacy heuristic often are not. `certify_policy` bounds the value improvement over
  the logging policy under Tan's marginal sensitivity model, reports the **tipping Γ** at which the
  ship/keep decision flips, and with `alpha=…` gates it on a finite-sample conformal lower bound
  (`conformal_action_value`). When the bound will not carry the decision, its `recommendation` is
  `abstain` rather than a green light.

Use `causalrl` when your problem is a causal *decision* over time; use DoWhy/EconML when it is a
treatment-effect *estimate*; use d3rlpy when you need the policy *trained*, and `causalrl` to
decide whether to ship it.

## Stability

The public API — the names exported from the top-level `causalrl` package — is stable and follows
[semantic versioning](https://semver.org): from **v1.0.0** on, breaking changes to exported names
move the major version. The 0.99.x line deliberately let the surface settle in real use first; 1.0
commits to it. See [Guarantees & Scope](https://raphaelrrcoelho.github.io/causalrl/guarantees/) for
what each method does and does not promise.

## Reproducible benchmarks

```bash
uv run --extra dev python benchmarks/scbandit_report.py confounded-chain \
  --seeds 0,1,2,3,4 --steps 8000 --tail-window 2000 --n-mc 2000
```

The JSON report includes each seed's result plus summary uncertainty. These maintained
demonstrations validate package behaviour on the stated environments; they are not general
performance guarantees.

## Development

```bash
uv run pytest                               # tests
uv run ruff check .                         # lint
uv run pyright src                          # types
uv run --extra docs mkdocs build --strict   # documentation
```

Contributions are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md).

## Citing

If you use `causalrl` in research, cite the metadata in [CITATION.cff](CITATION.cff) and the
primary source for the method you used (each is attributed inline in the
[Tour by Task](https://raphaelrrcoelho.github.io/causalrl/tour/) and its source module). See
[Citing causalrl](https://raphaelrrcoelho.github.io/causalrl/citing/).

## Acknowledgements

This library would not exist without the body of work it stands on. Particular thanks to:

- **Elias Bareinboim**, whose [9-task taxonomy of causal reinforcement learning](https://crl.causalai.net/)
  is the organising spine of `causalrl`, and whose results with collaborators are the core of
  nearly every slice — `do`-calculus completeness (with Shpitser & Pearl), transportability and
  selection diagrams (with Pearl), counterfactual data fusion (with Forney & Pearl), POMIS /
  structural causal bandits (with Lee), and causal imitation learning (with Zhang & Kumor).
- **Judea Pearl**, for the do-calculus and Pearl Causal Hierarchy that make every L1 / L2 / L3
  query in this library well-defined.
- **Sanghack Lee**, for the [reference POMIS implementation](https://github.com/sanghack81/SCMMAB-NIPS2018)
  the intervention-set engine is adapted from (MIT-licensed; attribution in
  `src/causalrl/identification/intervention_sets.py`).

Other foundational references — Spirtes, Glymour & Scheines; Zhang; Manski; Tan; Koller & Milch;
Ng, Harada & Russell; Bengio et al. — are cited inline at the slice that uses each.
