import pytest

from causalrl.agents.scbandit import NaivePOMISThompsonSampling, POMISThompsonSampling
from causalrl.envs.suite.scbandit import make_confounded_chain_env, make_frontdoor_env


def test_pomis_agent_is_manipulability_aware():
    env = make_frontdoor_env(seed=0)
    agent = POMISThompsonSampling(
        env.graph, env.reward, env.arms, seed=0, manipulable=env.manipulable
    )
    # POMIS^{Z} = {empty, {X}} -> arms {}, do(X=0), do(X=1) (all 3).
    key_sets = {frozenset(env.arms[i].keys()) for i in agent.allowed}
    assert key_sets == {frozenset(), frozenset({"X"})}
    assert len(agent.allowed) == 3


def test_naive_agent_collapses_to_observation():
    env = make_frontdoor_env(seed=0)
    agent = NaivePOMISThompsonSampling(env.graph, env.reward, env.arms, seed=0)
    # Unconstrained POMIS is {empty, {Z}}; {Z} maps to no arm (Z non-manipulable) -> only observe.
    assert [frozenset(env.arms[i].keys()) for i in agent.allowed] == [frozenset()]


def test_pomis_agent_unchanged_when_all_manipulable():
    # v0.4 confounded chain: every var manipulable -> identical pruning to before (3 arms).
    env = make_confounded_chain_env(seed=0)
    agent = POMISThompsonSampling(
        env.graph, env.reward, env.arms, seed=0, manipulable=env.manipulable
    )
    assert len(agent.allowed) == 3
    assert {frozenset(env.arms[i].keys()) for i in agent.allowed} == {
        frozenset(),
        frozenset({"X3"}),
    }


def test_pomis_agent_infers_nonmanipulable_contract_only_as_deprecated_fallback():
    env = make_frontdoor_env(seed=0)
    with pytest.warns(DeprecationWarning, match="manipulable"):
        agent = POMISThompsonSampling(env.graph, env.reward, env.arms, seed=0)
    assert len(agent.allowed) == 3
