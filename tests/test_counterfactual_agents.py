"""The counterfactual-optimal policy and the Regret Decision Criterion table."""

from __future__ import annotations

from causalrl.agents.counterfactual import CounterfactualOptimalPolicy
from causalrl.envs.suite.counterfactual_bandit import (
    build_counterfactual_scm,
    make_counterfactual_bandit_env,
)
from causalrl.identification.counterfactual import regret_decision_table


def test_regret_decision_table_optimum_is_the_intent() -> None:
    scm = build_counterfactual_scm()
    table = regret_decision_table(
        scm,
        outcome="Y",
        action_node="X",
        intent_node="I",
        arms=[0, 1, 2],
        intents=[0, 1, 2],
        n=40_000,
        seed=0,
    )
    for intent in (0, 1, 2):
        best_arm = max(table[intent], key=lambda arm: table[intent][arm])
        assert best_arm == intent
        assert abs(table[intent][intent] - 0.8) < 0.03


def test_oracle_plays_intent_and_scores_high() -> None:
    scm = build_counterfactual_scm()
    agent = CounterfactualOptimalPolicy(
        scm,
        outcome="Y",
        action_node="X",
        intent_node="I",
        arms=[0, 1, 2],
        intents=[0, 1, 2],
        n=40_000,
        seed=0,
    )
    env = make_counterfactual_bandit_env(seed=0)
    total = 0.0
    n = 4000
    for t in range(n):
        obs, _ = env.reset(seed=t)
        action = agent.act(obs)
        assert action == obs["intuition"]  # oracle plays arm = intent
        _, reward, _, _, _ = env.step(action)
        agent.update(obs, action, reward)
        total += reward
    assert abs(total / n - 0.8) < 0.03
