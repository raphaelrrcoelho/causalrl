# causalrl

[![CI](https://github.com/raphaelrrcoelho/causalrl/actions/workflows/ci.yml/badge.svg)](https://github.com/raphaelrrcoelho/causalrl/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-mkdocs--material-blue.svg)](https://raphaelrrcoelho.github.io/causalrl/)
[![PyPI](https://img.shields.io/pypi/v/causalrl.svg)](https://pypi.org/project/causalrl/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

Causal intervention-selection and causal-RL research tools.

`causalrl` provides graph algorithms for causal bandits, demonstration environments and agents,
and explicit-latent structural causal models with `see` (L1), `do` (L2), and `counterfactual`
(L3) queries, organised around the [9-task taxonomy of causal RL](https://crl.causalai.net/).

Scope is explicit and enforced in code: out-of-class identification queries raise
`NotIdentifiableError` with the witnessing hedge (or return `None` for the conservative helpers)
rather than guessing a formula, and learning agents are tabular/demo-scale, not production RL. See
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

## How it compares

`causalrl` is **causal-RL-first**, where the established causal libraries are estimation-first:

- **DoWhy / EconML / CausalML** target treatment-effect estimation and the
  identify→estimate→refute workflow on i.i.d. data. They are mature, production-grade tools.
  `causalrl` instead targets *sequential decision-making*: intervention-set selection (POMIS),
  confounded offline-to-online RL, counterfactual policies, and causal curricula / shaping /
  games. Those are the parts of the Bareinboim taxonomy these libraries do not cover.
- For pure graph identification it overlaps with **Ananke / pgmpy / Y0**. It deliberately does
  **not** reimplement offline RL at scale; pair it with a dedicated library such as
  [`d3rlpy`](https://github.com/takuseno/d3rlpy) for that.

Use `causalrl` when your problem is a causal *decision* over time; use DoWhy/EconML when it is a
treatment-effect *estimate*.

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
