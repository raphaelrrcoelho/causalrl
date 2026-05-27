"""Headline: counterfactual decision-making beats the best interventional decision."""

from __future__ import annotations

from causalrl.agents.bandits import CausalThompsonSampling, NaiveThompsonSampling
from causalrl.agents.base import Agent
from causalrl.agents.counterfactual import CounterfactualOptimalPolicy
from causalrl.envs.suite.counterfactual_bandit import (
    build_counterfactual_scm,
    make_counterfactual_bandit_env,
)


def _run(agent: Agent, seed0: int, n: int = 8000) -> float:
    env = make_counterfactual_bandit_env(seed=seed0)
    total = 0.0
    for t in range(n):
        obs, _ = env.reset(seed=seed0 + t)
        action = agent.act(obs)
        _, reward, _, _, _ = env.step(action)
        agent.update(obs, action, reward)
        total += reward
    return total / n


def test_counterfactual_beats_interventional_and_naive() -> None:
    oracle = CounterfactualOptimalPolicy(
        build_counterfactual_scm(),
        outcome="Y",
        action_node="X",
        intent_node="I",
        arms=[0, 1, 2],
        intents=[0, 1, 2],
        n=40_000,
        seed=0,
    )
    r_oracle = _run(oracle, seed0=1)
    r_causal = _run(CausalThompsonSampling(n_arms=3, n_contexts=3, seed=0), seed0=2)
    r_naive = _run(NaiveThompsonSampling(n_arms=3, seed=0), seed0=3)

    assert r_oracle > 0.75  # model-based counterfactual optimum ~0.8
    assert r_causal > 0.70  # online RDC learner converges to the per-intent optimum
    assert r_naive < 0.45  # confounding-naive agent stuck near the do-optimum ~0.367
    assert r_causal > r_naive + 0.20
    assert r_oracle > r_naive + 0.30
