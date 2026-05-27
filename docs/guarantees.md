# Guarantees And Scope

This page separates implemented guarantees from research demonstrations and unsupported
extensions.

## Stable Contracts

### Graph And Intervention Sets

`CausalGraph` represents an ADMG for analytical operations. `pomis` and
`minimal_intervention_sets` implement the single-reward structural-causal-bandit slice and
support an explicit `manipulable` subset through latent projection.

```python
from causalrl import CausalGraph, pomis

graph = CausalGraph(
    directed_edges=[("X", "Z"), ("Z", "Y")],
    bidirected_edges=[("X", "Y")],
)
assert set(pomis(graph, "Y", manipulable={"X"})) == {frozenset(), frozenset({"X"})}
```

### Executable SCMs

`StructuralCausalModel` executes explicit-latent DAGs only. An ADMG with a bidirected edge is
useful for graph analysis, but cannot be sampled as an SCM unless the shared latent causes are
represented as nodes. Constructors validate graph/mechanism/exogenous-distribution alignment.
Sampling uses private Torch state and does not overwrite an experiment's global RNG state.

### Environment Interoperability

Public environments satisfy Gymnasium's environment checker. `reset(seed=...)` follows
Gymnasium seeding behavior, and rollout utilities end episodes on either `terminated` or
`truncated`.

## Assumption-Dependent Methods

Multi-stage `DOVI` propagates value through learned transitions. Its causal interpretation
requires transitions that are not confounded:

```python
from causalrl import DOVI

agent = DOVI(
    n_states=5,
    n_actions=2,
    horizon=2,
    transition_assumption="unconfounded",
)
assert agent.is_certified
```

For exploratory runs where that premise is not available, callers must opt in:

```python
agent = DOVI(n_states=10, n_actions=4, horizon=6, allow_heuristic=True)
assert not agent.is_certified
```

`POMISThompsonSampling` should be given the legal intervention variables explicitly:

```python
agent = POMISThompsonSampling(
    env.graph,
    env.reward,
    env.arms,
    seed=0,
    manipulable=env.manipulable,
)
```

Inference of manipulability from arm enumeration remains available only as a deprecated
compatibility fallback.

## Causal-RL Taxonomy Methods

These slices implement the Bareinboim 9-task taxonomy. Each is faithful to its cited source within a
stated scope; conservative helpers return `None` or raise outside that scope rather than guess.

- **Counterfactual decision-making (ETT).** `counterfactual_expectation` and
  `effect_of_treatment_on_treated` evaluate Layer-3 queries on an executable SCM;
  `CounterfactualOptimalPolicy` acts by the Regret Decision Criterion. Requires an executable
  explicit-latent SCM.
- **Transportability.** `transport_formula` / `is_transportable` decide direct and S-admissible
  adjustment transportability over selection diagrams (via m-separation); `transported_effect`
  computes the reweighted estimate. This is not the full hedge-based sID/gID engine — unsupported
  cases return `None`.
- **Causal discovery.** `discover` runs the PC algorithm (conditional independence by conditional
  mutual information, then collider and Meek orientation) and returns a `CPDAG`. Assumes causal
  sufficiency and faithfulness; the CMI test is thresholded, not a calibrated hypothesis test; and
  `CPDAG.to_causal_graph` refuses to orient an equivalence class.
- **Causal imitation.** `is_imitable` / `imitation_backdoor_set` decide imitability via the
  π-backdoor criterion (an observed back-door-admissible set); `CausalImitator` clones `P(A | Z)`.
- **Causal curriculum.** `causal_curriculum` orders skills by the causal topological order;
  `PrerequisiteLearner` models causally-gated mastery (an abstract learner, not an RL policy).
- **Causal reward shaping.** `apply_potential_shaping` is policy-invariant for any potential and
  `causal_potential` supplies `V*`, over deterministic tabular MDPs.
- **Causal games.** `CausalGame` / `pure_nash_equilibria` represent MACIDs and enumerate
  pure-strategy Nash equilibria for normal-form games; mixed equilibria are out of scope.

## Not Yet Claimed

- General ID, full hedge-based sID, or gID identification (only direct and S-admissible-adjustment
  transportability is implemented).
- Causal discovery with latent confounders (FCI) or score-based search (GES); mixed-strategy
  equilibria.
- Published confounding-sensitivity or doubly robust OPE bounds.
- Production-ready deep or offline-RL training integrations.
- General statistical guarantees from the maintained toy benchmark environments.

The experimental sensitivity helper lives under `causalrl.experimental.ope`.
