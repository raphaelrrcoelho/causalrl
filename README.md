# causalrl

Causal reinforcement learning: structural causal models meet RL.

`causalrl` provides a Pearl-Causal-Hierarchy-aware substrate — structural causal models
with `see` (L1), `do` (L2), and `counterfactual` (L3) queries — and causal RL algorithms
built on top, organized around the [9-task taxonomy of causal RL](https://crl.causalai.net/).

## Install

```bash
uv sync --extra dev
```

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

## Layout

- `causalrl.scm` — `CausalGraph`, mechanisms, and `StructuralCausalModel` (`see`/`do`/`counterfactual`)
- `causalrl.identification` — back-door sets and identifiability criteria
- `causalrl.envs` — Gymnasium-compatible causal environments (`MABUCEnv`)
- `causalrl.agents` — causal and baseline bandit agents
- `causalrl.eval` — regret metrics and off-policy evaluation under confounding

## Development

```bash
uv run pytest          # tests
uv run ruff check .    # lint
uv run pyright src     # types
```
