import pytest

from causalrl.agents.scbandit import (
    BruteForceInterventionTS,
    FixedSetThompsonSampling,
    POMISThompsonSampling,
)
from causalrl.envs.suite.scbandit import make_confounded_chain_env


def test_pomis_agent_restricts_to_pomis_arms():
    env = make_confounded_chain_env(seed=0)
    agent = POMISThompsonSampling(
        env.graph, env.reward, env.arms, seed=0, manipulable=env.manipulable
    )
    # POMIS = {empty, {X3}} -> allowed arms are {}, {X3:0}, {X3:1}.
    key_sets = [frozenset(env.arms[i].keys()) for i in agent.allowed]
    assert len(agent.allowed) == 3
    assert set(key_sets) == {frozenset(), frozenset({"X3"})}
    assert key_sets.count(frozenset()) == 1  # the observational arm
    assert key_sets.count(frozenset({"X3"})) == 2  # do(X3=0), do(X3=1)


def test_brute_force_considers_all_arms():
    env = make_confounded_chain_env(seed=0)
    agent = BruteForceInterventionTS(env.arms, seed=0)
    assert len(agent.allowed) == len(env.arms) == 27


def test_fixed_set_agent_restricts_to_one_set():
    env = make_confounded_chain_env(seed=0)
    agent = FixedSetThompsonSampling(env.arms, {"X3"}, seed=0)
    assert len(agent.allowed) == 2  # {X3:0}, {X3:1}
    for i in agent.allowed:
        assert set(env.arms[i].keys()) == {"X3"}


def test_act_returns_allowed_index_and_update_runs():
    env = make_confounded_chain_env(seed=0)
    agent = POMISThompsonSampling(
        env.graph, env.reward, env.arms, seed=0, manipulable=env.manipulable
    )
    a = agent.act({})
    assert a in agent.allowed
    agent.update({}, a, 1.0)  # must not raise


def test_bounded_reward_updates_beta_posterior_fractionally():
    agent = BruteForceInterventionTS([{}], seed=0)
    agent.update({}, 0, 0.25)
    assert agent._alpha[0] == 1.25
    assert agent._beta[0] == 1.75


@pytest.mark.parametrize("reward", [-0.01, 1.01])
def test_reward_outside_unit_interval_is_rejected(reward: float):
    agent = BruteForceInterventionTS([{}], seed=0)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        agent.update({}, 0, reward)


def test_pomis_agent_requires_explicit_manipulable():
    env = make_confounded_chain_env(seed=0)
    with pytest.raises(ValueError, match="manipulable"):
        POMISThompsonSampling(env.graph, env.reward, env.arms, seed=0)


def test_pomis_agent_rejects_arm_outside_explicit_manipulable_contract():
    env = make_confounded_chain_env(seed=0)
    with pytest.raises(ValueError, match="non-manipulable"):
        POMISThompsonSampling(env.graph, env.reward, env.arms, seed=0, manipulable={"X1", "X2"})
