"""Reproducible benchmark reporting for the implemented structural-bandit demonstrations."""

from __future__ import annotations

import math
import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from causalrl.agents.base import Agent
from causalrl.agents.scbandit import (
    BruteForceInterventionTS,
    FixedSetThompsonSampling,
    NaivePOMISThompsonSampling,
    POMISThompsonSampling,
)
from causalrl.envs.suite.scbandit import (
    StructuralCausalBanditEnv,
    make_confounded_chain_env,
    make_frontdoor_env,
)


@dataclass(frozen=True)
class BenchmarkEstimate:
    """A per-seed benchmark measurement with simple descriptive uncertainty."""

    name: str
    seeds: tuple[int, ...]
    values: tuple[float, ...]
    mean: float
    std: float
    ci95_low: float
    ci95_high: float

    @classmethod
    def from_values(
        cls, name: str, *, seeds: Sequence[int], values: Sequence[float]
    ) -> BenchmarkEstimate:
        seed_tuple = tuple(seeds)
        value_tuple = tuple(float(value) for value in values)
        if not seed_tuple or len(seed_tuple) != len(value_tuple):
            raise ValueError("seeds and values must be non-empty sequences of equal length")
        mean = statistics.fmean(value_tuple)
        std = statistics.stdev(value_tuple) if len(value_tuple) > 1 else 0.0
        margin = 1.96 * std / math.sqrt(len(value_tuple))
        return cls(name, seed_tuple, value_tuple, mean, std, mean - margin, mean + margin)

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "seeds": list(self.seeds),
            "values": list(self.values),
            "mean": self.mean,
            "std": self.std,
            "ci95_low": self.ci95_low,
            "ci95_high": self.ci95_high,
        }


def report_to_dict(report: dict[str, BenchmarkEstimate]) -> dict[str, dict[str, object]]:
    """Convert a report into JSON-serializable dictionaries."""
    return {name: estimate.to_dict() for name, estimate in report.items()}


def _tail_mean_rewards(
    name: str,
    *,
    seeds: Sequence[int],
    n_steps: int,
    tail_window: int,
    env_factory: Callable[[int], StructuralCausalBanditEnv],
    agent_factory: Callable[[StructuralCausalBanditEnv, int], Agent],
) -> BenchmarkEstimate:
    if n_steps < 1 or not 1 <= tail_window <= n_steps:
        raise ValueError("tail_window must be between 1 and n_steps")
    values: list[float] = []
    for seed in seeds:
        env = env_factory(seed)
        agent = agent_factory(env, seed)
        obs, _ = env.reset(seed=seed)
        rewards: list[float] = []
        for _ in range(n_steps):
            action = agent.act(obs)
            next_obs, reward, _terminated, _truncated, _info = env.step(action)
            agent.update(obs, action, reward)
            rewards.append(reward)
            obs = next_obs
        values.append(statistics.fmean(rewards[-tail_window:]))
    return BenchmarkEstimate.from_values(name, seeds=seeds, values=values)


def run_confounded_chain_benchmark(
    *,
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
    n_steps: int = 8000,
    tail_window: int = 2000,
    n_mc: int = 2000,
) -> dict[str, BenchmarkEstimate]:
    """Report POMIS, brute-force, and fixed-set behavior on the confounded-chain demo."""

    def env_factory(seed: int) -> StructuralCausalBanditEnv:
        return make_confounded_chain_env(seed=seed, n_mc=n_mc)

    return {
        "pomis": _tail_mean_rewards(
            "pomis",
            seeds=seeds,
            n_steps=n_steps,
            tail_window=tail_window,
            env_factory=env_factory,
            agent_factory=lambda env, seed: POMISThompsonSampling(
                env.graph, env.reward, env.arms, seed=seed, manipulable=env.manipulable
            ),
        ),
        "brute_force": _tail_mean_rewards(
            "brute_force",
            seeds=seeds,
            n_steps=n_steps,
            tail_window=tail_window,
            env_factory=env_factory,
            agent_factory=lambda env, seed: BruteForceInterventionTS(env.arms, seed=seed),
        ),
        "fixed_set": _tail_mean_rewards(
            "fixed_set",
            seeds=seeds,
            n_steps=n_steps,
            tail_window=tail_window,
            env_factory=env_factory,
            agent_factory=lambda env, seed: FixedSetThompsonSampling(env.arms, {"X3"}, seed=seed),
        ),
    }


def run_frontdoor_benchmark(
    *,
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
    n_steps: int = 30000,
    tail_window: int = 10000,
    n_mc: int = 20000,
) -> dict[str, BenchmarkEstimate]:
    """Report manipulability-aware and naive-filter behavior on the front-door demo."""

    def env_factory(seed: int) -> StructuralCausalBanditEnv:
        return make_frontdoor_env(seed=seed, n_mc=n_mc)

    return {
        "manipulability_aware": _tail_mean_rewards(
            "manipulability_aware",
            seeds=seeds,
            n_steps=n_steps,
            tail_window=tail_window,
            env_factory=env_factory,
            agent_factory=lambda env, seed: POMISThompsonSampling(
                env.graph, env.reward, env.arms, seed=seed, manipulable=env.manipulable
            ),
        ),
        "naive_filter": _tail_mean_rewards(
            "naive_filter",
            seeds=seeds,
            n_steps=n_steps,
            tail_window=tail_window,
            env_factory=env_factory,
            agent_factory=lambda env, seed: NaivePOMISThompsonSampling(
                env.graph, env.reward, env.arms, seed=seed
            ),
        ),
    }
