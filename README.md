# causalrl

[![CI](https://github.com/raphaelrrcoelho/causalrl/actions/workflows/ci.yml/badge.svg)](https://github.com/raphaelrrcoelho/causalrl/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

Causal intervention-selection and causal-RL research tools.

`causalrl` provides graph algorithms for causal bandits, demonstration environments and agents,
and explicit-latent structural causal models with `see` (L1), `do` (L2), and
`counterfactual` (L3) queries. The implemented slices are organized around the
[9-task taxonomy of causal RL](https://crl.causalai.net/).

## Install

```bash
uv pip install -e .             # graph, POMIS, tabular agents/environments
uv pip install -e ".[torch]"    # SCM sampling, neural mechanisms, Torch-backed demos
uv sync --extra dev             # contributors: tests, lint, typing, notebooks
```

## Supported scope

- `pomis` and `minimal_intervention_sets` implement the causal-bandit intervention-set slice,
  including non-manipulable variables through latent projection.
- `StructuralCausalModel` executes explicit-latent DAGs. Use bidirected-edge ADMGs for
  analytical graph algorithms; represent a shared latent cause as an explicit SCM node.
- `backdoor_adjustment_set` and `is_identifiable` are conservative scoped helpers, not a full
  ID/sID/gID engine. Unsupported confounded ADMG cases are refused or reported as unknown.
- `UCDTR`, `DOVI`, and `DeepDeconfoundedQ` are benchmark/demo agents, not production
  offline-RL integrations. The qualitative sensitivity-interval helper is experimental at
  `causalrl.experimental.ope`, not a validated OPE estimator.

## Quickstart: MABUC

A causal agent that conditions on its "intuition" beats a confounding-naive agent on the
Multi-Armed Bandit with Unobserved Confounders, even though both arms have identical
interventional means.

```python
from causalrl.agents.bandits import CausalThompsonSampling, NaiveThompsonSampling
from causalrl.envs.suite.mabuc import MABUCEnv

env = MABUCEnv(seed=1)
agent = CausalThompsonSampling(n_arms=2, n_contexts=2, seed=0)

obs, _ = env.reset(seed=1)
for _ in range(8000):
    action = agent.act(obs)
    _, reward, _, _, _ = env.step(action)
    agent.update(obs, action, reward)
    obs, _ = env.reset()
```

The causal agent converges to ~0.75 reward/step; a `NaiveThompsonSampling` baseline that
ignores the intuition is stuck near 0.50. See `examples/mabuc_vertical_slice.ipynb` for the
full walkthrough across every layer (SCM, environment, agents, evaluation).

## v0.2: Causal offline-to-online (Task 1)

Combine confounded offline logs with online interaction. On a confounded dynamic treatment
regime, an agent that reads the logs through **Manski causal bounds** (UC-DTR / DOVI /
DeepDeconfoundedQ) reaches the optimal policy, while a **naive** offline learner that trusts
the logs is *biased* — it picks the wrong treatment and never recovers.

```python
from causalrl.agents.offline_online import UCDTR
from causalrl.data.dataset import generate_logs
from causalrl.envs.suite.dtr import DTREnv
from causalrl.eval.harness import run_episodes

logs = generate_logs(DTREnv(seed=100), n_episodes=4000, seed=100)
agent = UCDTR(n_states=3, n_actions=2, seed=0)
agent.ingest_offline(logs)              # reads logs via causal bounds, not raw means
returns = run_episodes(agent, DTREnv(seed=0), n_episodes=4000, seed=0)
# UC-DTR ~0.73 (optimal 0.75) vs naive-offline ~0.675 (biased by the confounding)
```

A note on scope: Manski *natural* bounds cannot strictly prune, so the headline is
**causal-vs-naive** (not a regret win over from-scratch online learning). The deep agent is a
lightweight net for the toy demo; `d3rlpy` is the designated backbone at real scale. See
`examples/offline_to_online.ipynb` for the three-way comparison.

## v0.4: Where to intervene (Task 2)

Given the causal graph, **POMIS** (Possibly-Optimal Minimal Intervention Sets) prunes the
exponential space of interventions to the few that could be optimal. On a confounded chain
`X1→X2→X3→Y` (with `X1↔Y`), the only POMISs are `∅` and `{X3}`, so a POMIS agent plays 3
arms instead of brute force's 27 — and discovers that *observing* (`∅`) beats every fixed
intervention, the MABUC effect carried onto a chain.

```python
from causalrl import POMISThompsonSampling, pomis
from causalrl.envs.suite.scbandit import make_confounded_chain_env

env = make_confounded_chain_env(seed=1)
print(pomis(env.graph, "Y"))            # [frozenset(), frozenset({'X3'})]

agent = POMISThompsonSampling(
    env.graph, env.reward, env.arms, seed=0, manipulable=env.manipulable
)
env.reset(seed=1)
for _ in range(8000):
    a = agent.act({})
    _, r, _, _, _ = env.step(a)
    agent.update({}, a, r)
# POMIS agent converges to ~1.0 (the observational arm); a brute-force agent over all 27
# arms converges far slower, and a naive do(X3)-only agent is stuck near 0.5.
```

The POMIS engine is adapted from the MIT-licensed reference implementation of
Lee & Bareinboim, *Structural Causal Bandits: Where to Intervene?* (NeurIPS 2018),
[`sanghack81/SCMMAB-NIPS2018`](https://github.com/sanghack81/SCMMAB-NIPS2018). See
`examples/where_to_intervene.ipynb`.

## v0.5: Non-manipulable variables

Real systems have variables you can *observe* but not *intervene on* (cholesterol, say). Given
a manipulable subset, **`pomis` gains a `manipulable=` argument**: by latent-projecting out the
non-manipulable variables it still finds the right lever even when the true cause is untouchable
(Lee & Bareinboim, *Structural Causal Bandits with Non-Manipulable Variables*, AAAI 2019). On
the front-door graph `X→Z→Y` (with `X↔Y`, `Z` non-manipulable) the POMIS is `{∅, {X}}` — an
agent steers `Y` through `X`, while a naive agent that just filters the unconstrained POMIS is
left observing.

```python
from causalrl import pomis
from causalrl.envs.suite.scbandit import make_frontdoor_env

env = make_frontdoor_env(seed=1)                  # X->Z->Y, X<->Y, with Z non-manipulable
print(pomis(env.graph, "Y", manipulable={"X"}))   # [frozenset(), frozenset({'X'})]
# A manipulability-aware POMISThompsonSampling reaches do(X=1) ~0.56; the naive baseline that
# ignores the constraint collapses to observation ~0.50.
```

## Layout

- `causalrl.scm` — ADMG graph operations plus explicit-latent DAG `StructuralCausalModel` (`see`/`do`/`counterfactual`)
- `causalrl.identification` — scoped conservative criteria, Manski `causal_q_bounds`, and POMIS (`pomis`, `minimal_intervention_sets`)
- `causalrl.envs` — Gymnasium-compatible causal environments (`MABUCEnv`, `DTREnv`, `SequentialDTREnv`, `ConfoundedGridworld`, `SequentialMABUCEnv`, `StructuralCausalBanditEnv`)
- `causalrl.data` — `ConfoundedTrajectoryDataset` and offline-log generation
- `causalrl.agents` — bandit agents plus causal offline-to-online learners (`UCDTR`, `DOVI`, `DeepDeconfoundedQ`) and baselines
- `causalrl.eval` — regret metrics, the offline-to-online harness, and IPW evaluation; exploratory sensitivity utilities live in `causalrl.experimental`

## Development

```bash
uv run pytest          # tests
uv run ruff check .    # lint
uv run pyright src     # types
```
