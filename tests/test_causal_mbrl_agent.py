"""CausalMBRLAgent: the front-door routes to the right planner behind a uniform fit/act surface."""

from __future__ import annotations

import numpy as np
import pytest

from causalrl.agents.causal_mbrl import CausalMBRLAgent
from causalrl.data.dataset import generate_logs
from causalrl.envs.suite.continuous_confounded import ContinuousConfoundedBandit
from causalrl.envs.suite.seq_dtr import SequentialDTREnv
from causalrl.envs.suite.simpson_bandit import SimpsonBandit
from causalrl.envs.suite.transport_bandit import TransportableConfoundedBandit
from causalrl.scm.graph import CausalGraph


def test_backdoor_route_recovers_the_optimum() -> None:
    env = SimpsonBandit(seed=3)
    agent = CausalMBRLAgent(env.n_actions, graph=env.graph)
    agent.fit(env.sample(50_000, seed=3))
    assert agent.strategy == "backdoor"
    assert agent.adjustment_set == ("Z",)
    assert agent.act({"state": 0}) == 1
    assert "strategy=backdoor" in agent.explain()


def test_discovery_route_learns_structure_and_acts() -> None:
    env = SimpsonBandit(seed=0)
    agent = CausalMBRLAgent(
        env.n_actions, variables=("Z", "A", "Y"), tiers=(("Z",), ("A",), ("Y",))
    )
    agent.fit(env.sample(5_000, seed=0))
    assert agent.strategy == "discovery"
    assert agent.adjustment_set == ("Z",)
    assert agent.act({"state": 0}) == 1


def test_transport_route_carries_the_policy_across_the_shift() -> None:
    env = TransportableConfoundedBandit(gamma=1.0, shift=0.6, seed=0)
    source = env.sample(20_000, domain="source", seed=0)
    target_w = env.sample(20_000, domain="target", seed=1)["W"]
    agent = CausalMBRLAgent(env.n_actions, graph=env.graph, transport=("W",))
    agent.fit(source, target_covariates={"W": target_w})
    assert agent.strategy == "transport"
    assert agent.act({"state": 0}) == env.optimal_action(domain="target") == 0
    assert "transportable=True" in agent.explain()


def test_function_approx_route_handles_a_continuous_confounder() -> None:
    env = ContinuousConfoundedBandit(gamma=1.0, seed=0)
    agent = CausalMBRLAgent(env.n_actions, graph=env.graph, continuous_confounder=True)
    agent.fit(env.sample(6000, seed=0))
    assert agent.strategy == "function_approx"
    assert agent.act({"state": 0}) == env.optimal_action() == 0
    assert "confounder=Z" in agent.explain()


def test_g_formula_route_deconfounds_many_covariates() -> None:
    rng = np.random.default_rng(0)
    n = 4000
    x = rng.normal(0.0, 1.0, (n, 2))
    a = (rng.random(n) < 1.0 / (1.0 + np.exp(-(1.2 * x[:, 0] - 0.8 * x[:, 1])))).astype(int)
    y = 2.0 * a + 3.0 * x[:, 0] - 1.5 * x[:, 1] + rng.normal(0.0, 1.0, n)
    data = {"A": a, "Y": y, "X0": x[:, 0], "X1": x[:, 1]}
    agent = CausalMBRLAgent(2, covariates=("X0", "X1"))
    agent.fit(data)
    assert agent.strategy == "g_formula"
    assert agent.act({"state": 0}) == 1  # action 1 has the positive true effect
    assert "covariates=" in agent.explain()


def test_sequential_route_fits_logs_and_acts() -> None:
    env = SequentialDTREnv(horizon=2, seed=0)
    logs = generate_logs(SequentialDTREnv(horizon=2, seed=11), n_episodes=500, seed=11)
    agent = CausalMBRLAgent(2, horizon=2, n_states=env.n_states)
    agent.fit(logs)
    assert agent.strategy == "sequential"
    obs, _ = env.reset(seed=0)
    assert agent.act(obs) in (0, 1)


def test_routing_validation_errors() -> None:
    w_graph = CausalGraph(directed_edges=[("Z", "A"), ("Z", "Y"), ("W", "Y"), ("A", "Y")])
    with pytest.raises(ValueError):
        CausalMBRLAgent(2)  # no graph and no variables/tiers -> discovery cannot route
    with pytest.raises(ValueError):
        CausalMBRLAgent(2, transport=("W",))  # transport needs a graph
    with pytest.raises(ValueError):
        CausalMBRLAgent(2, horizon=2)  # sequential needs n_states
    agent = CausalMBRLAgent(2, graph=w_graph, transport=("W",))
    with pytest.raises(ValueError):
        agent.fit({})  # transport fit without target_covariates
