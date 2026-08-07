# Guarantees And Scope

This page separates implemented guarantees from research demonstrations and unsupported
extensions.

The public API — the names exported from the top-level `causalrl` package — is considered stable
under [semantic versioning](https://semver.org). It is released as **v0.99.0**, a humble step short
of a 1.0 tag while the API settles in real use.

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

The `manipulable` set is required: inferring it from arm enumeration was removed in v0.99.0.

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
- **General identification from surrogate experiments (gID).** `identify_effect_with_experiments` /
  `is_gid_identifiable` extend the ID recursion: where observation hits a hedge, the needed c-factor
  is obtained from an available experiment (Tian's `Identify` subroutine), and
  `estimate_effect_with_experiments` evaluates the result on observational plus (randomized)
  experimental data. With no experiments it coincides exactly with ID; validated by simulation on a
  graph that is not observationally identifiable but is identified by a surrogate experiment.
- **Transportability (sID, mz, meta).** `transport_formula` / `is_transportable` give a readable
  closed form for the two workhorse cases (direct and S-admissible adjustment) over selection
  diagrams. `identify_transport` / `transport_estimand` / `is_transportable_effect` decompose the
  target effect into c-factors and route each to whichever domain can supply it. At c-factor
  granularity a factor is invariant exactly when it touches no selection-marked variable, so for a
  single observational source this routing is the *complete* single-domain sID: it reduces to the ID
  algorithm and raises a witnessing **transport-hedge** when no domain supplies a needed factor.
  `Domain` + `identify_transport_general` / `is_transportable_general` / `estimate_transport_general`
  generalize this to **multiple source domains** (meta-transportability) and to **surrogate
  experiments** in a domain (mz-transportability): each c-factor is searched across the
  domains/experiments that can supply it, with the target as the fallback. Validated by simulation: a
  covariate-shift estimate matches the target's true `do()`, a source experiment breaks a bow-arc
  hedge, and an effect is assembled from invariant factors contributed by different sources. Sound
  throughout; the one case it does not stitch is a single c-factor identifiable only by *combining
  several experiments* (resolved per-experiment, reported non-transportable rather than guessed).
- **Causal discovery.** `discover` runs the PC algorithm (conditional independence by conditional
  mutual information, then collider and Meek orientation) and returns a `CPDAG`;
  `discover_interventional` additionally orients edges from interventional (L2) data by the
  invariance principle, yielding the interventional essential graph. These assume causal sufficiency
  and faithfulness; the CMI test is thresholded, not a calibrated hypothesis test; and
  `CPDAG.to_causal_graph` refuses to orient an equivalence class. `discover_latent` runs the **FCI**
  algorithm — dropping causal sufficiency — and returns a `PAG` with the complete orientation rules
  R1-R10 (Zhang 2008, sound and complete for latent confounders and selection bias): `a <-> b` marks
  a latent confounder and a circle endpoint is undetermined by the equivalence class. Validated
  against the true MAG of the data-generating DAG-with-latents (the M-bias collider among them).
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
  arithmetic), and for three or more agents by support enumeration with a numerical Newton solve of
  the multilinear indifference system — every returned profile is verified to be an ε-Nash
  equilibrium (no agent gains more than `1e-6` by deviating to a pure action).
- **Partial-identification / OPE bounds.** `causal_q_bounds` (confounded RL logs) and `manski_bounds`
  (observational data) give the sharp no-assumptions Manski interval on a counterfactual mean;
  `ipw_sensitivity_bounds` gives the marginal-sensitivity-model (Tan's Γ) interval, collapsing to the
  IPW point at Γ=1 and containing the truth once Γ exceeds the true confounding odds ratio. Outcomes
  are assumed bounded; validated against a confounded SCM with a known effect.

## Not Yet Claimed

- Multi-experiment c-factor stitching. Transportability resolves each c-factor from a single
  domain/experiment; a c-factor identifiable only by *combining several experiments* (the deepest
  gID case) is reported non-transportable rather than guessed. Single-domain observational ID
  (`identify_effect`), gID (`identify_effect_with_experiments`), and the multi-domain mz/meta
  transportability (`identify_transport_general`) above are implemented and validated.
- Score-based causal discovery (GES) and interventional FCI.
- Doubly-robust OPE *point* estimators (the bounds above are partial-identification intervals, not
  point estimates).
- Production-ready deep or offline-RL training integrations.
- General statistical guarantees from the maintained toy benchmark environments.
