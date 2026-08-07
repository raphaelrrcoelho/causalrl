# Task Guides

Five runnable, end-to-end walkthroughs of the core workflows. Each is a self-contained script under
`examples/guides/` that runs on the numpy core and is executed in CI, so the code here always works.
Run any of them with `python examples/guides/<name>.py`.

## 1. Evaluate a policy offline under confounding

`examples/guides/01_evaluate_offline_under_confounding.py`

You have logged `(state, action, reward)` data and a candidate policy. `certify_policy` bounds the
policy's value improvement over the behaviour policy under a marginal-sensitivity budget and reports
the **tipping `gamma`** — the confounding strength at which the "ship it" decision would flip.

```python
from causalrl import certify_policy

cert = certify_policy(dataset, target_actions, gamma_max=5.0)
print(cert.decision, cert.certified, cert.tipping_gamma)
```

## 2. Bound an effect under `gamma` with estimated propensities

`examples/guides/02_bound_under_gamma_estimated_propensities.py`

With only an *estimated* propensity model, `ipw_sensitivity_bounds` returns a `BOUNDED` certificate
at each `gamma`; the interval widens monotonically and `tipping_gamma` finds where it first crosses a
reference.

```python
from causalrl import ipw_sensitivity_bounds

cert = ipw_sensitivity_bounds(outcomes, estimated_propensities, gamma=2.0)  # BOUNDED
lo, hi = cert.value.lower, cert.value.upper
```

## 3. Transport across regimes, including a mechanism swap

`examples/guides/03_transport_across_regimes.py`

A posterior over environment parameters becomes a cloud of calibrated `Regime`s (a `Regime` can mark
a mechanism swap via a selection node); `across_regimes` reports the `[min, max]` envelope of a
functional over the ensemble.

```python
from causalrl import regimes_from_posterior, across_regimes

regimes = regimes_from_posterior(posterior, selection=["reward_mechanism"])
envelope = across_regimes(regimes, transported_value)   # worst-case across calibrated configs
```

## 4. Certify an agent inside a population

`examples/guides/04_certify_agent_in_population.py`

`linear_gaussian_population_env` exposes one ego agent in a fixed population as a per-agent causal
environment; a doubly-robust `certify_effect` recovers the ego's action effect and matches the
interventional ground truth from `env.do(...)`.

```python
from causalrl import certify_effect, linear_gaussian_population_env

env = linear_gaussian_population_env(ego="ego", ego_effect=1.5)
cert = certify_effect(graph, "ego", "Y", data, method="aipw")
```

For the *learning* side of a population, `run_no_regret` plays the game with one no-regret learner
per free agent and hands the realized empirical joint to the certificate layer:

```python
from causalrl import certify_cce_do, run_no_regret

run = run_no_regret(population, 20_000, do={"A2": 0})
cert = certify_cce_do(game, welfare, do={"A2": 0}, no_regret=False, epsilon=run.regret)
```

## 5. Scale it

`examples/guides/05_scale_it.py`

The same importance-sampling certificate runs over a log far larger than memory: `stream_policy_value`
consumes a `TrajectoryLog` one batch at a time and never materialises it. Swap the in-memory log for a
Parquet path (`log.to_parquet(path)` then `stream_policy_value(path)`) and it streams row-groups from
disk unchanged.

```python
from causalrl.ope.ipw import stream_policy_value

cert = stream_policy_value(log, weight="weight", reward="reward", batch_size=100_000)
```
