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

## Not Yet Claimed

- General ID, sID, or gID identification.
- Published confounding-sensitivity or doubly robust OPE bounds.
- Production-ready deep or offline-RL training integrations.
- General statistical guarantees from the maintained toy benchmark environments.

The experimental sensitivity helper lives under `causalrl.experimental.ope`.
