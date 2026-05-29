# Tour by Task

One runnable example per capability, organised by the
[9-task taxonomy of causal RL](https://crl.causalai.net/). Every slice is conservative by
design and faithful to its primary source (cited inline). The MABUC quickstart is in
[Getting Started](index.md).

## Causal offline-to-online (Task 1)

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

Manski *natural* bounds cannot strictly prune, so the headline is **causal-vs-naive** (not a
regret win over from-scratch online learning). The deep agent is a lightweight net for the
toy demo; `d3rlpy` is the designated backbone at real scale.

## Where to intervene — POMIS (Task 2)

Given the causal graph, **POMIS** (Possibly-Optimal Minimal Intervention Sets) prunes the
exponential space of interventions to the few that could be optimal. On a confounded chain
`X1→X2→X3→Y` (with `X1↔Y`), the only POMISs are `∅` and `{X3}`, so a POMIS agent plays 3 arms
instead of brute force's 27 — and discovers that *observing* (`∅`) beats every fixed
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
# POMIS agent converges to ~1.0 (the observational arm); brute force over all 27 arms is
# far slower, and a naive do(X3)-only agent is stuck near 0.5.
```

The POMIS engine is adapted from the MIT-licensed reference implementation of Lee &
Bareinboim, *Structural Causal Bandits: Where to Intervene?* (NeurIPS 2018),
[`sanghack81/SCMMAB-NIPS2018`](https://github.com/sanghack81/SCMMAB-NIPS2018).

## Non-manipulable variables

Real systems have variables you can *observe* but not *intervene on* (cholesterol, say). Given
a manipulable subset, `pomis` gains a `manipulable=` argument: by latent-projecting out the
non-manipulable variables it still finds the right lever even when the true cause is
untouchable (Lee & Bareinboim, *Structural Causal Bandits with Non-Manipulable Variables*,
AAAI 2019). On the front-door graph `X→Z→Y` (with `X↔Y`, `Z` non-manipulable) the POMIS is
`{∅, {X}}`.

```python
from causalrl import pomis
from causalrl.envs.suite.scbandit import make_frontdoor_env

env = make_frontdoor_env(seed=1)                  # X->Z->Y, X<->Y, Z non-manipulable
print(pomis(env.graph, "Y", manipulable={"X"}))   # [frozenset(), frozenset({'X'})]
# A manipulability-aware agent reaches do(X=1) ~0.56; a naive baseline collapses to ~0.50.
```

## Counterfactual decision-making (Task 3)

Your own *intent* — the action you are naturally inclined to take — carries information about a
hidden confounder. **Counterfactual decision-making** asks "given that I'm inclined toward `i`,
what is the best action?", i.e. `E[Y_do(a) | intent = i]`, and acts on it. On a 3-arm confounded
bandit where every fixed `do(a)` averages only ~0.367, conditioning on intent recovers the ~0.8
optimum.

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
# Counterfactual-optimal ~0.8; the best fixed do(a) arm only ~0.367.
```

Faithful to Bareinboim, Forney & Pearl, *Bandits with Unobserved Confounders* (NeurIPS 2015)
and Pearl, *Causality* §8.2.1.

## Transportability (Task 4)

An effect learned in one population does not always hold in another. Given a **selection
diagram** marking which mechanisms differ across domains, `transport_formula` decides whether
the target effect is recoverable and how. On the covariate-shift graph `Z→X, Z→Y, X→Y` (domains
differ in `P(Z)`), reusing the source effect is biased, but reweighting the source conditionals
by the *target* covariate distribution transports it exactly.

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

Conservative by design — returns `None` outside the supported class (direct / S-admissible
adjustment) rather than guessing. See [Transportability](transportability.md) for the full
gID / sID / mz / meta surface. Faithful to Bareinboim & Pearl (AAAI 2012; J. Causal Inference
2013).

## Learning causal models (Task 5)

When the graph is unknown, **learn it**. `discover` runs the PC algorithm over discrete data
(conditional independence via conditional mutual information, then collider + Meek orientation)
and returns a CPDAG; a fully oriented result bridges into the rest of the library for planning.

```python
from causalrl import pomis
from causalrl.discovery import discover
from causalrl.envs.suite.discovery import sample_discovery_data

data = sample_discovery_data(n=10_000, seed=0)        # collider X->Z<-Y, plus Z->W
graph = discover(data, ["X", "Y", "Z", "W"]).to_causal_graph()
print(sorted(graph.directed_edges))                   # [('X','Z'), ('Y','Z'), ('Z','W')]
print(pomis(graph, "W"))                              # [frozenset({'Z'})] — plan on the learned model
```

PC assumes causal sufficiency and faithfulness; the CPDAG may stay partially oriented, and
`to_causal_graph` raises rather than guess. See [Causal Discovery](discovery.md) for FCI / PAG
under latent confounding. Faithful to Spirtes, Glymour & Scheines and Meek (UAI 1995).

## Causal imitation learning (Task 6)

When an unobserved confounder drives both the expert's actions and the outcome, naively cloning
the action distribution is **biased**. `is_imitable` says whether imitation is even feasible
and, if so, which observed set to condition on; `CausalImitator` clones `P(A | Z)` and
reproduces the expert's reward.

```python
from causalrl.imitation import CausalImitator, is_imitable
from causalrl.envs.suite.imitation import (
    ImitationEnv, generate_demonstrations, make_imitation_diagram,
)

graph, observable = make_imitation_diagram()      # observed confounder: W->A, W->Y, A->Y
print(is_imitable(graph, action="A", outcome="Y", observable=observable))  # True (adjust on W)

demos = generate_demonstrations(ImitationEnv(seed=0))
imitator = CausalImitator(n_actions=2, adjustment=["W"])
imitator.fit(demos, action="A")
# Deployed, the causal imitator earns ~0.9 (matches the expert); marginal BC earns ~0.5.
```

When the confounder is latent (no observed admissible set) `is_imitable` returns `False` rather
than a biased policy. Faithful to Zhang, Kumor & Bareinboim (NeurIPS 2020).

## Causal curriculum learning (Task 7)

Learn skills in causal order. `causal_curriculum` topologically sorts the prerequisite graph so
every cause is mastered before its effects; a learner that follows it reaches the goal, while
one fed a prerequisite-violating order strands the blocked skills.

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

Faithful to Bengio, Louradour, Collobert & Weston, *Curriculum Learning* (ICML 2009); the
causal contribution is the topological ordering rule.

## Causal reward shaping (Task 8)

Speed learning without changing the optimum. Potential-based shaping adds `γΦ(s') − Φ(s)` to
the reward — policy-invariant for *any* potential — and using the causal value `V*` as the
potential turns a sparse reward dense.

```python
from causalrl.shaping import causal_potential, q_learning, value_iteration
from causalrl.envs.suite.shaping import make_sparse_chain_mdp

mdp = make_sparse_chain_mdp(length=12)              # reward only at the goal
optimal = value_iteration(mdp)[1]                   # "always right"
shaped = q_learning(mdp, potential=causal_potential(mdp), episodes=20, seed=0)
unshaped = q_learning(mdp, episodes=20, seed=0)
print(shaped == optimal, unshaped == optimal)       # True False
```

The optimal policy is provably unchanged by any potential (Ng, Harada & Russell, ICML 1999);
the causal contribution is using `V*` from the model as the potential.

## Causal game theory (Task 9)

Represent a multi-agent game as a **causal influence diagram** (a decision and a utility node
per agent) and solve for equilibria. `pure_nash_equilibria` enumerates the pure-strategy Nash
equilibria; on the canonical games it recovers the textbook answers.

```python
from causalrl.games import pure_nash_equilibria
from causalrl.envs.suite.games import matching_pennies, prisoners_dilemma

print(pure_nash_equilibria(prisoners_dilemma()))   # [{'row': 1, 'col': 1}] — mutual defection
print(pure_nash_equilibria(matching_pennies()))    # [] — only a mixed equilibrium exists
```

Faithful to Koller & Milch (multi-agent influence diagrams, 2003) and Hammond et al.,
*Reasoning about Causality in Games* (2023).

## Identification machinery

Beneath the task slices is a conservative identification layer:

- `identify_effect` — the complete Shpitser–Pearl ID algorithm; non-identifiable effects raise
  with a witnessing hedge.
- **gID** — general identification from surrogate experiments (Bareinboim & Pearl, JMLR 2015).
- **sID / mz / meta** — cross-domain and multi-source transportability via c-factor routing
  (Bareinboim & Pearl, AAAI 2012 / NeurIPS 2014).
- **FCI / PAG** — latent-confounder-aware structure discovery (Zhang 2008).
- `manski_bounds`, `ipw_sensitivity_bounds` — validated partial-identification and
  marginal-sensitivity-model bounds (Manski 1990; Tan 2006).

See [Guarantees & Scope](guarantees.md) for exactly what each routine promises, and
[Reproducing the Literature](classics.md) for canonical examples from the causal-RL canon.
