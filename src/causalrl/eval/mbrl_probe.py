"""M0 kill-gate: does a confounding-robust causal agent beat a naive one on TRUE value, and does
the advantage survive a reward shift?

The verdict is READ from the report (compare ``causal_shifted`` vs ``naive_shifted``, with
``causal_source`` vs ``naive_source`` as the in-distribution check); it is deliberately not asserted
in code — whether the causal agent wins is the empirical question this harness answers.
"""

from __future__ import annotations

from collections.abc import Sequence

from causalrl.agents.baselines import NaiveOffline
from causalrl.agents.mbrl import CertifiedPolicyAgent
from causalrl.data.dataset import generate_logs
from causalrl.envs.suite.confounded_context import make_confounded_context_env
from causalrl.eval.benchmark import BenchmarkEstimate


def _policy_from_agent(agent: NaiveOffline | CertifiedPolicyAgent, n_states: int) -> list[int]:
    return [int(agent.act({"state": s})) for s in range(n_states)]


def run_m0_kill_gate(
    *,
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
    gamma: float = 0.9,
    n_episodes: int = 4000,
    gamma_max: float = 5.0,
) -> dict[str, BenchmarkEstimate]:
    """Train a causal (certify-gated) and a naive agent on confounded source logs; report each
    agent's exact true policy value on the source and shifted oracle, per seed."""
    rows: dict[str, list[float]] = {
        "causal_source": [],
        "naive_source": [],
        "causal_shifted": [],
        "naive_shifted": [],
    }
    for seed in seeds:
        source = make_confounded_context_env(gamma=gamma, shift=False, seed=seed)
        shifted = make_confounded_context_env(gamma=gamma, shift=True, seed=seed)
        dataset = generate_logs(source, n_episodes=n_episodes, seed=seed)

        causal = CertifiedPolicyAgent(source.n_states, source.n_actions, gamma_max=gamma_max)
        causal.ingest_offline(dataset)
        naive = NaiveOffline(source.n_states, source.n_actions)
        naive.ingest_offline(dataset)

        causal_pi = _policy_from_agent(causal, source.n_states)
        naive_pi = _policy_from_agent(naive, source.n_states)
        rows["causal_source"].append(source.true_policy_value(causal_pi))
        rows["naive_source"].append(source.true_policy_value(naive_pi))
        rows["causal_shifted"].append(shifted.true_policy_value(causal_pi))
        rows["naive_shifted"].append(shifted.true_policy_value(naive_pi))

    return {
        name: BenchmarkEstimate.from_values(name, seeds=seeds, values=values)
        for name, values in rows.items()
    }
