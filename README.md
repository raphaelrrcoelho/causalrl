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
uv sync --extra docs            # local documentation site and API reference
```

Documentation covers [guarantees and scope](docs/guarantees.md), the
[reproducible benchmark protocol](docs/benchmarks.md), and the [API reference](docs/api.md).
Research use should cite the metadata in [CITATION.cff](CITATION.cff).

## Supported scope

- `pomis` and `minimal_intervention_sets` implement the causal-bandit intervention-set slice,
  including non-manipulable variables through latent projection.
- `StructuralCausalModel` executes explicit-latent DAGs. Use bidirected-edge ADMGs for
  analytical graph algorithms; represent a shared latent cause as an explicit SCM node.
- `identify_effect` runs the complete Shpitser–Pearl ID algorithm, with general identification from
  surrogate experiments (gID), transportability across domains (sID / mz / meta), and FCI for
  latent-confounder discovery; a non-identifiable effect raises with a witnessing hedge.
- `manski_bounds` and `ipw_sensitivity_bounds` give validated partial-identification and
  marginal-sensitivity-model bounds. `UCDTR`, `DOVI`, and `DeepDeconfoundedQ` are benchmark/demo
  agents, not production offline-RL integrations.
- Multi-stage `DOVI` requires `transition_assumption="unconfounded"` for a certified
  transition-value backup; `allow_heuristic=True` permits explicitly un-certified exploration.

## Stability

As of **v1.0.0** the public API — the names exported from the top-level `causalrl` package — is
stable and follows [semantic versioning](https://semver.org): breaking changes require a major
version bump. See [Guarantees And Scope](docs/guarantees.md) for what each method does and does not
promise.

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

## v0.6: Counterfactual decision-making (Task 3)

Your own *intent* — the action you are naturally inclined to take — carries information about a
hidden confounder. **Counterfactual decision-making** asks "given that I'm inclined toward `i`, what
is the best action?", i.e. `E[Y_{do(a)} | intent = i]`, and acts on it. On a 3-arm confounded bandit
where every fixed intervention `do(a)` averages only ~0.367, conditioning on intent recovers the
~0.8 optimum — the MABUC lesson carried to `K > 2` arms.

```python
from causalrl import CounterfactualOptimalPolicy
from causalrl.envs.suite.counterfactual_bandit import (
    build_counterfactual_scm,
    make_counterfactual_bandit_env,
)

scm = build_counterfactual_scm()                      # U->I, U->Y, I->X, X->Y
agent = CounterfactualOptimalPolicy(
    scm, outcome="Y", action_node="X", intent_node="I", arms=[0, 1, 2], intents=[0, 1, 2],
)
env = make_counterfactual_bandit_env(seed=1)
obs, _ = env.reset(seed=1)
action = agent.act(obs)                               # plays arm == intuition
# The counterfactual-optimal policy reaches ~0.8; the best fixed do(a) arm only ~0.367.
```

The ETT estimand (`effect_of_treatment_on_treated`) and the counterfactual estimator are faithful to
Bareinboim, Forney & Pearl, *Bandits with Unobserved Confounders* (NeurIPS 2015) and Pearl,
*Causality* §8.2.1.

## v0.7: Transportability (Task 4)

An effect learned in one population does not always hold in another. Given a **selection diagram**
marking which mechanisms differ across domains, `transport_formula` decides whether the target
effect is recoverable and how. On the canonical covariate-shift graph `Z→X, Z→Y, X→Y` (domains
differ in `P(Z)`), reusing the source effect is biased, but reweighting the source conditionals by
the *target* covariate distribution transports it exactly.

```python
from causalrl import transport_formula, transported_effect
from causalrl.envs.suite.transport import make_transport_domains

source, target, diagram = make_transport_domains()        # differ only in P(Z)
formula = transport_formula(diagram, treatment="X", outcome="Y")
print(formula.kind, sorted(formula.adjustment_set))       # adjustment ['Z']

transported = transported_effect(
    formula, treatment="X", treated_value=1.0, outcome="Y", source=source, target=target,
)
# transported ~0.82 matches the true target effect; the naive source effect is ~0.58.
```

Conservative by design — like the rest of `causalrl.identification`, it returns `None` outside the
supported class (direct / S-admissible adjustment) rather than guessing. Faithful to Bareinboim &
Pearl, *Transportability of Causal Effects* (AAAI 2012; J. Causal Inference 2013).

## v0.8: Learning causal models (Task 5)

When the graph is unknown, **learn it**. `discover` runs the PC algorithm over discrete data
(conditional independence via conditional mutual information, then collider + Meek orientation) and
returns a CPDAG; a fully oriented result bridges into the rest of the library for planning.

```python
from causalrl import pomis
from causalrl.discovery import discover
from causalrl.envs.suite.discovery import sample_discovery_data

data = sample_discovery_data(n=10_000, seed=0)        # collider X->Z<-Y, plus Z->W
graph = discover(data, ["X", "Y", "Z", "W"]).to_causal_graph()
print(sorted(graph.directed_edges))                   # [('X', 'Z'), ('Y', 'Z'), ('Z', 'W')]
print(pomis(graph, "W"))                              # [frozenset({'Z'})] — plan on the learned model
```

PC assumes causal sufficiency (no latent confounders) and faithfulness; the CPDAG may stay partially
oriented, and `to_causal_graph` raises rather than guess. Faithful to Spirtes, Glymour & Scheines
and Meek (UAI 1995).

## v0.9: Causal imitation learning (Task 6)

When an unobserved confounder drives both the expert's actions and the outcome, naively cloning the
action distribution is **biased** — the cloner acts independently of the confounding the expert used.
`is_imitable` says whether imitation is even feasible and, if so, which observed set to condition on;
`CausalImitator` clones `P(A | Z)` and reproduces the expert's reward.

```python
from causalrl.imitation import CausalImitator, is_imitable
from causalrl.envs.suite.imitation import (
    ImitationEnv,
    generate_demonstrations,
    make_imitation_diagram,
)

graph, observable = make_imitation_diagram()      # observed confounder: W->A, W->Y, A->Y
print(is_imitable(graph, action="A", outcome="Y", observable=observable))  # True (adjust on W)

demos = generate_demonstrations(ImitationEnv(seed=0))
imitator = CausalImitator(n_actions=2, adjustment=["W"])
imitator.fit(demos, action="A")
# Deployed, the causal imitator earns ~0.9 (matches the expert); marginal BC earns ~0.5.
```

When the confounder is latent (no observed admissible set) `is_imitable` returns `False` rather than
a biased policy. Faithful to Zhang, Kumor & Bareinboim (NeurIPS 2020).

## v0.10: Causal curriculum learning (Task 7)

Learn skills in causal order. `causal_curriculum` topologically sorts the prerequisite graph so every
cause is mastered before its effects; a learner that follows it reaches the goal, while one fed a
prerequisite-violating order strands the blocked skills.

```python
from causalrl.curriculum import PrerequisiteLearner, causal_curriculum
from causalrl.envs.suite.curriculum import make_skill_diamond

graph, goal = make_skill_diamond()                  # S0 -> {S1, S2} -> S3
order = causal_curriculum(graph, goal)              # a valid topological order ending at S3
learner = PrerequisiteLearner(graph)
learner.train(order)
print(learner.masters(goal))                        # True
learner.train(list(reversed(order)))
print(learner.masters(goal))                        # False — prerequisites violated
```

Faithful to Bengio, Louradour, Collobert & Weston, *Curriculum Learning* (ICML 2009); the causal
contribution is the topological ordering rule.

## v0.11: Causal reward shaping (Task 8)

Speed learning without changing the optimum. Potential-based shaping adds `γΦ(s') − Φ(s)` to the
reward — policy-invariant for *any* potential — and using the causal value `V*` as the potential
turns a sparse reward dense, so a learner converges far faster.

```python
from causalrl.shaping import causal_potential, q_learning, value_iteration
from causalrl.envs.suite.shaping import make_sparse_chain_mdp

mdp = make_sparse_chain_mdp(length=12)              # reward only at the goal
optimal = value_iteration(mdp)[1]                   # "always right"
shaped = q_learning(mdp, potential=causal_potential(mdp), episodes=20, seed=0)
unshaped = q_learning(mdp, episodes=20, seed=0)
print(shaped == optimal, unshaped == optimal)       # True False — shaping reaches it; sparse lags
```

The optimal policy is provably unchanged by any potential (Ng, Harada & Russell, ICML 1999); the
causal contribution is using `V*` from the model as the potential.

## v0.12: Causal game theory (Task 9)

Represent a multi-agent game as a **causal influence diagram** (a decision and a utility node per
agent) and solve for equilibria. `pure_nash_equilibria` enumerates the pure-strategy Nash equilibria;
on the canonical games it recovers the textbook answers.

```python
from causalrl.games import pure_nash_equilibria
from causalrl.envs.suite.games import matching_pennies, prisoners_dilemma

print(pure_nash_equilibria(prisoners_dilemma()))   # [{'row': 1, 'col': 1}] — mutual defection
print(pure_nash_equilibria(matching_pennies()))    # [] — only a mixed equilibrium exists
```

Faithful to Koller & Milch (multi-agent influence diagrams, 2003) and Hammond et al., *Reasoning
about Causality in Games* (2023). **This completes the 9-task causal-RL taxonomy.**

## Layout

- `causalrl.scm` — ADMG graph operations plus explicit-latent DAG `StructuralCausalModel` (`see`/`do`/`counterfactual`)
- `causalrl.identification` — scoped conservative criteria, Manski `causal_q_bounds`, POMIS (`pomis`, `minimal_intervention_sets`), counterfactual ETT (`counterfactual_expectation`, `effect_of_treatment_on_treated`), and transportability (`transport_formula`, `transported_effect`, `is_backdoor_admissible`)
- `causalrl.discovery` — constraint-based causal structure learning (`discover`, `conditional_mutual_information`, `CPDAG`)
- `causalrl.imitation` — causal imitation learning (`is_imitable`, `imitation_backdoor_set`, `CausalImitator`, `BehavioralCloning`)
- `causalrl.curriculum` — causal curriculum learning (`causal_curriculum`, `is_valid_curriculum`, `PrerequisiteLearner`)
- `causalrl.shaping` — causal / potential-based reward shaping (`apply_potential_shaping`, `causal_potential`, `value_iteration`, `q_learning`, `TabularMDP`)
- `causalrl.games` — multi-agent causal influence diagrams & equilibria (`CausalGame`, `pure_nash_equilibria`, `best_response`, `is_nash_equilibrium`)
- `causalrl.envs` — Gymnasium-compatible causal environments (`MABUCEnv`, `DTREnv`, `SequentialDTREnv`, `ConfoundedGridworld`, `SequentialMABUCEnv`, `StructuralCausalBanditEnv`)
- `causalrl.data` — `ConfoundedTrajectoryDataset` and offline-log generation
- `causalrl.agents` — bandit agents plus causal offline-to-online learners (`UCDTR`, `DOVI`, `DeepDeconfoundedQ`) and baselines
- `causalrl.eval` — regret metrics, the offline-to-online harness, and IPW evaluation; exploratory sensitivity utilities live in `causalrl.experimental`

## Development

```bash
uv run pytest          # tests
uv run ruff check .    # lint
uv run pyright src     # types
uv run --extra docs mkdocs build --strict  # documentation
```

## Reproducible Benchmarks

```bash
uv run --extra dev python benchmarks/scbandit_report.py confounded-chain \
  --seeds 0,1,2,3,4 --steps 8000 --tail-window 2000 --n-mc 2000
```

The JSON report includes each seed's result plus summary uncertainty. These maintained
demonstrations validate package behavior on the stated environments; they are not general
performance guarantees.
