# causalrl

`causalrl` is a research-oriented Python library for causal intervention selection and
causal reinforcement-learning demonstrations. Its strongest implemented slice is structural
causal bandits: graph algorithms select intervention sets, executable explicit-latent SCMs
generate data, and benchmark agents compare restricted action spaces.

## Install

```bash
uv pip install causalrl
uv pip install "causalrl[torch]"  # SCMs and Torch-backed examples
```

The core graph, POMIS, tabular-agent, and tabular-environment surfaces do not require
PyTorch. SCM sampling, neural mechanisms, and structural-bandit environments do.

## Start Here

- Read [Guarantees And Scope](guarantees.md) before relying on causal claims.
- Reproduce the maintained demonstrations through [Reproducible Benchmarks](benchmarks.md).
- Browse the [API Reference](api.md) for current stable entry points.

## Implemented Slices

| Area | Supported behavior |
| --- | --- |
| Graph algorithms | ADMG operations, latent projection, POMIS/MIS with manipulability |
| SCM execution | Explicit-latent DAG models with observational, interventional, and sampled counterfactual queries |
| Environments | Gymnasium-compatible causal demos with reproducible local RNG behavior |
| Agents | Tabular benchmark/demo agents for implemented environments |
| Evaluation | Regret metrics, IPW point evaluation, deterministic multi-seed benchmark reporting |

The package deliberately marks exploratory or assumption-dependent functionality instead of
presenting it as a general solution.
