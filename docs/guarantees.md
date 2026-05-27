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
- **General identification (ID algorithm).** `identify_effect` runs the sound and complete
  Shpitser-Pearl ID algorithm: it returns a do-free `Estimand` for `P(y | do(x))` in any ADMG or
  raises `NotIdentifiableError` with the witnessing hedge; `estimate_effect` evaluates the estimand
  on data and `is_identifiable_effect` gives the decision. Validated by simulation on the back-door
  and front-door graphs (the estimand matches the true `do()` distribution) and on the bow-arc and
  instrumental-variable graphs (correctly non-identifiable). `requires_experiment` answers Task 2's
  "when to intervene": an experiment is needed exactly when the effect is not observationally
  identifiable. Scope: a single observational distribution over discrete variables.
- **Transportability.** `transport_formula` / `is_transportable` decide direct and S-admissible
  adjustment transportability over selection diagrams (via m-separation); `transported_effect`
  computes the reweighted estimate. This is the adjustment-based slice, not the full hedge-based
  cross-domain sID — unsupported cases return `None`.
- **Causal discovery.** `discover` runs the PC algorithm (conditional independence by conditional
  mutual information, then collider and Meek orientation) and returns a `CPDAG`;
  `discover_interventional` additionally orients edges from interventional (L2) data by the
  invariance principle, yielding the interventional essential graph. Assumes causal sufficiency and
  faithfulness; the CMI test is thresholded, not a calibrated hypothesis test; and
  `CPDAG.to_causal_graph` refuses to orient an equivalence class.
- **Causal imitation.** `is_imitable` / `imitation_backdoor_set` decide imitability via the
  π-backdoor criterion (an observed back-door-admissible set); `CausalImitator` clones `P(A | Z)`.
- **Causal curriculum.** `causal_curriculum` orders skills by the causal topological order;
  `PrerequisiteLearner` models causally-gated mastery; `curriculum_q_learning` trains Q-learning
  through a sequence of subtasks (warm-start transfer), reaching a sparse target that flat learning
  on the same budget misses.
- **Causal reward shaping.** `apply_potential_shaping` is policy-invariant for any potential and
  `causal_potential` supplies `V*`, over deterministic tabular MDPs.
- **Causal games.** `CausalGame` / `pure_nash_equilibria` represent MACIDs and enumerate
  pure-strategy Nash equilibria for any number of agents; `mixed_nash_equilibria` finds all
  mixed-strategy equilibria of a two-player game exactly by support enumeration (rational
  arithmetic). Mixed equilibria for more than two players need nonlinear solvers and stay out of
  scope.

## Not Yet Claimed

- Full cross-domain transportability (the complete sID algorithm) or general identification from
  arbitrary surrogate experiments (gID). General single-domain observational ID *is* implemented
  (`identify_effect`); transportability remains the direct / S-admissible-adjustment slice.
- Causal discovery with latent confounders (FCI) or score-based search (GES); mixed-strategy
  equilibria for more than two players.
- Published confounding-sensitivity or doubly robust OPE bounds.
- Production-ready deep or offline-RL training integrations.
- General statistical guarantees from the maintained toy benchmark environments.

The experimental sensitivity helper lives under `causalrl.experimental.ope`.
