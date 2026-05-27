"""Headline: the causal imitator matches the expert; naive behavioral cloning is biased."""

from __future__ import annotations

from causalrl.agents.base import Agent
from causalrl.envs.suite.imitation import ImitationEnv, expert_policy, generate_demonstrations
from causalrl.imitation import BehavioralCloning, CausalImitator


def _deploy(agent: Agent, seed0: int, n: int = 4000) -> float:
    env = ImitationEnv(seed=seed0)
    total = 0.0
    for t in range(n):
        obs, _ = env.reset(seed=seed0 + t)
        _, reward, _, _, _ = env.step(agent.act(obs))
        total += reward
    return total / n


def _expert_reward(seed0: int, n: int = 4000) -> float:
    env = ImitationEnv(seed=seed0)
    total = 0.0
    for t in range(n):
        obs, _ = env.reset(seed=seed0 + t)
        _, reward, _, _, _ = env.step(expert_policy(obs))
        total += reward
    return total / n


def test_causal_imitator_matches_expert_naive_bc_biased() -> None:
    demos = generate_demonstrations(ImitationEnv(seed=0), n=4000, seed=0)
    causal = CausalImitator(n_actions=2, adjustment=["W"], seed=0)
    causal.fit(demos, action="A")
    bc = BehavioralCloning(n_actions=2, seed=0)
    bc.fit(demos, action="A")

    r_expert = _expert_reward(seed0=300)
    r_causal = _deploy(causal, seed0=100)
    r_bc = _deploy(bc, seed0=200)

    assert abs(r_expert - 0.9) < 0.03
    assert abs(r_causal - 0.9) < 0.03  # causal imitator reproduces the expert
    assert abs(r_bc - 0.5) < 0.05  # naive BC severs the confounding, stuck near 0.5
    assert r_causal > r_bc + 0.3
