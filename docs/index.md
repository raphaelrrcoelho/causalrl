# causalrl

Causal intervention-selection and causal-RL research tools. `causalrl` provides graph
algorithms for causal bandits, demonstration environments and agents, and explicit-latent
structural causal models with `see` (L1), `do` (L2), and `counterfactual` (L3) queries,
organised around the [9-task taxonomy of causal RL](https://crl.causalai.net/).

It is built for **honesty about scope**: identification routines return `None` (or raise with
a witnessing hedge) outside their supported class rather than guessing, and agents are marked
as benchmark/demo rather than production. Read [Guarantees & Scope](guarantees.md) before
relying on any causal claim.

## Install

```bash
uv pip install -e .             # graph, POMIS, tabular agents/environments
uv pip install -e ".[torch]"    # SCM sampling, neural mechanisms, Torch-backed demos
```

The core graph, POMIS, tabular-agent, and tabular-environment surfaces do not require
PyTorch; SCM sampling, neural mechanisms, and structural-bandit environments do.

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
# CausalThompsonSampling -> ~0.75 reward/step; a confounding-naive baseline is stuck near 0.50.
```

## What it does

| Task (taxonomy) | Capability | Key entry points |
| --- | --- | --- |
| Decision under confounding | Counterfactual Thompson sampling on the MABUC | `CausalThompsonSampling` |
| 1 — Offline→online | Learn from confounded logs via causal bounds | `UCDTR`, `DOVI`, `DeepDeconfoundedQ` |
| 2 — Where to intervene | POMIS / MIS, incl. non-manipulable variables | `pomis`, `minimal_intervention_sets` |
| 3 — Counterfactual policy | Act on `E[Y_do(a) | intent]` | `CounterfactualOptimalPolicy` |
| 4 — Transportability | Recover effects across domains | `transport_formula`, `transported_effect` |
| 5 — Causal discovery | PC / FCI structure learning | `discover`, `CPDAG` |
| 6 — Causal imitation | Imitability + confounded cloning | `is_imitable`, `CausalImitator` |
| 7 — Causal curriculum | Prerequisite-ordered skill learning | `causal_curriculum` |
| 8 — Reward shaping | Policy-invariant causal potentials | `causal_potential`, `q_learning` |
| 9 — Causal games | Influence diagrams + equilibria | `pure_nash_equilibria`, `CausalGame` |
| Identification | Complete ID / gID / sID / mz, partial-ID & sensitivity bounds | `identify_effect`, `manski_bounds`, `ipw_sensitivity_bounds` |

A runnable example for every row is in the [Tour by Task](tour.md).

## Start here

- [Tour by Task](tour.md) — one runnable example per capability.
- [Tutorials](tutorials.md) — end-to-end notebooks across the full stack.
- [Guarantees & Scope](guarantees.md) — what each method does and does not promise.
- [Reproducible Benchmarks](benchmarks.md) — the maintained, multi-seed demonstrations.
- [API Reference](api.md) — stable entry points.
- [Citing causalrl](citing.md).
