"""CertifiedPolicyAgent (certify-gated) and BackdoorAdjustedAgent (active deconfounding)."""

from __future__ import annotations

import numpy as np
import pytest

from causalrl.agents.baselines import NaiveOffline
from causalrl.agents.mbrl import (
    BackdoorAdjustedAgent,
    CertifiedPolicyAgent,
    DiscoveryBackdoorAgent,
    FunctionApproxBackdoorAgent,
    TransportBackdoorAgent,
)
from causalrl.data.dataset import ConfoundedTrajectoryDataset, Transition
from causalrl.envs.suite.continuous_confounded import ContinuousConfoundedBandit
from causalrl.envs.suite.simpson_bandit import SimpsonBandit
from causalrl.envs.suite.transport_bandit import TransportableConfoundedBandit
from causalrl.scm.graph import CausalGraph


def _clean_improvement_dataset() -> ConfoundedTrajectoryDataset:
    # Unconfounded 50/50 logs, single context: action 1 pays 1.0, action 0 pays 0.0.
    trs: list[Transition] = []
    for _ in range(100):
        trs.append(Transition(0, 1, 1.0, 0, True))
        trs.append(Transition(0, 0, 0.0, 0, True))
    return ConfoundedTrajectoryDataset(trs, n_states=1, n_actions=2)


def test_certifies_and_ships_the_improving_action() -> None:
    # Low gamma_max: only modest confounding need be ruled out, so a clean strong improvement
    # certifies and is shipped.
    agent = CertifiedPolicyAgent(n_states=1, n_actions=2, gamma_max=1.2)
    agent.ingest_offline(_clean_improvement_dataset())
    assert agent.policy == [1]
    assert agent.act({"state": 0}) == 1


def test_does_not_deviate_to_a_worse_action() -> None:
    # Action 0 pays 1.0 and behavior plays it most; action 1 pays 0.0. Deviating to action 1 has a
    # negative contrast, so the agent keeps action 0 (whether by certifying it or by abstaining).
    trs = [Transition(0, 0, 1.0, 0, True) for _ in range(150)]
    trs += [Transition(0, 1, 0.0, 0, True) for _ in range(50)]
    ds = ConfoundedTrajectoryDataset(trs, n_states=1, n_actions=2)
    agent = CertifiedPolicyAgent(n_states=1, n_actions=2, gamma_max=5.0)
    agent.ingest_offline(ds)
    assert agent.policy == [0]


def test_never_logged_action_is_skipped_not_crashed() -> None:
    # Only action 1 is ever logged; candidates assigning the never-logged action 0 must be skipped
    # (no support), and the agent still returns a valid one-action-per-state policy.
    trs = [Transition(s % 2, 1, 1.0, 0, True) for s in range(40)]
    ds = ConfoundedTrajectoryDataset(trs, n_states=2, n_actions=2)
    agent = CertifiedPolicyAgent(n_states=2, n_actions=2)
    agent.ingest_offline(ds)
    assert len(agent.policy) == 2
    assert all(a in (0, 1) for a in agent.policy)


def test_downside_gate_changes_the_policy_the_agent_ships() -> None:
    """The agent-side path into the conformal layer: with ``alpha`` the agent gates candidates on
    a calibrated worst-case return, so a mean-improving but heavy-tailed action is no longer
    shipped and the agent abstains to behavior. Without it the same agent ships that action."""
    trs = [Transition(0, 0, 0.5, 0, True) for _ in range(1000)]
    trs += [Transition(0, 1, 1.5, 0, True) for _ in range(950)]
    trs += [Transition(0, 1, -5.0, 0, True) for _ in range(50)]
    ds = ConfoundedTrajectoryDataset(trs, n_states=1, n_actions=2)

    ungated = CertifiedPolicyAgent(n_states=1, n_actions=2, gamma_max=1.02)
    ungated.ingest_offline(ds)
    assert ungated.policy == [1]

    gated = CertifiedPolicyAgent(n_states=1, n_actions=2, gamma_max=1.02, alpha=0.1)
    gated.ingest_offline(ds)
    assert gated.policy == [0]
    assert gated.act({"state": 0}) == 0


def test_policy_has_one_action_per_state() -> None:
    # Both actions logged in both states.
    trs: list[Transition] = []
    for s in range(2):
        trs += [Transition(s, 0, float(s == 0), 0, True) for _ in range(30)]
        trs += [Transition(s, 1, float(s == 1), 0, True) for _ in range(30)]
    ds = ConfoundedTrajectoryDataset(trs, n_states=2, n_actions=2)
    agent = CertifiedPolicyAgent(n_states=2, n_actions=2)
    agent.ingest_offline(ds)
    assert len(agent.policy) == 2
    assert all(a in (0, 1) for a in agent.policy)


def test_backdoor_agent_recovers_the_interventional_optimum() -> None:
    env = SimpsonBandit(seed=3)
    data = env.sample(50_000, seed=3)
    agent = BackdoorAdjustedAgent(env.n_actions, graph=env.graph)
    agent.fit(data)
    # Back-door adjustment for the observed Z recovers do(A=1) > do(A=0): pick the true optimum.
    assert agent.adjustment == ("Z",)
    assert agent.act({"state": 0}) == 1


def test_discovery_agent_learns_structure_and_recovers_optimum() -> None:
    tiers = (("Z",), ("A",), ("Y",))
    for seed in range(5):
        env = SimpsonBandit(seed=seed)
        agent = DiscoveryBackdoorAgent(env.n_actions, variables=("Z", "A", "Y"))
        agent.discover_and_fit(env.sample(5_000, seed=seed), tiers=tiers)
        # Skeleton discovery + temporal tiers -> back-door set {Z}, reliably across seeds.
        assert agent.adjustment == ("Z",)
        assert agent.act({"state": 0}) == 1


def test_transport_agent_beats_fooled_naive_under_confounding_and_shift() -> None:
    # Strong (overlap-preserving) confounding + a large covariate shift: the naive marginal is
    # fooled into the harmful arm 1; the deconfound+transport agent keeps the safe, optimal arm 0.
    env = TransportableConfoundedBandit(gamma=1.0, shift=0.6, seed=0)
    source = env.sample(20_000, domain="source", seed=0)
    target_w = env.sample(20_000, domain="target", seed=1)["W"]

    agent = TransportBackdoorAgent(env.n_actions, graph=env.graph, transport=("W",))
    agent.fit(source, target_covariates={"W": target_w})
    assert agent.adjustment == ("Z",)
    assert agent.act({"state": 0}) == env.optimal_action(domain="target") == 0

    transitions = [
        Transition(0, int(a), float(y), 0, True)
        for a, y in zip(source["A"], source["Y"], strict=True)
    ]
    dataset = ConfoundedTrajectoryDataset(transitions, n_states=1, n_actions=2)
    naive = NaiveOffline(env.n_states, env.n_actions)
    naive.ingest_offline(dataset)
    assert naive.act({"state": 0}) == 1  # confounded marginal picks the trap

    # The causal agent's realized target value strictly exceeds the naive agent's.
    causal_value = env.true_action_value(agent.act({"state": 0}), domain="target")
    naive_value = env.true_action_value(naive.act({"state": 0}), domain="target")
    assert causal_value > naive_value


def test_transport_effect_is_identifiable() -> None:
    # The target effect P*(Y | do(A)) is transportable via S-admissible adjustment on {Z, W}.
    env = TransportableConfoundedBandit(gamma=0.5, shift=0.5)
    agent = TransportBackdoorAgent(env.n_actions, graph=env.graph, transport=("W",))
    assert agent.transportable is True


def test_function_approx_agent_beats_fooled_naive_on_continuous_confounder() -> None:
    # Continuous confounder + nonlinear reward: ridge-RBF back-door recovers arm 1's true low value
    # and keeps the safe arm 0, while the confounded marginal is fooled into the harmful arm 1.
    env = ContinuousConfoundedBandit(gamma=1.0, seed=0)
    data = env.sample(6000, seed=0)

    agent = FunctionApproxBackdoorAgent(env.n_actions, graph=env.graph)
    agent.fit(data)
    assert agent.confounder == "Z"
    assert agent.values[1] < 0.45  # true E[Y|do(1)] ~ 0.38, comfortably below arm 0's 0.5
    assert agent.act({"state": 0}) == env.optimal_action() == 0

    transitions = [
        Transition(0, int(a), float(y), 0, True) for a, y in zip(data["A"], data["Y"], strict=True)
    ]
    dataset = ConfoundedTrajectoryDataset(transitions, n_states=1, n_actions=2)
    naive = NaiveOffline(env.n_states, env.n_actions)
    naive.ingest_offline(dataset)
    assert naive.act({"state": 0}) == 1  # confounded marginal picks the trap


def test_function_approx_agent_requires_a_single_confounder() -> None:
    # The RBF outcome model is 1-D; a graph with a 2-variable back-door set is rejected up front.
    graph = CausalGraph(
        directed_edges=[("U1", "A"), ("U1", "Y"), ("U2", "A"), ("U2", "Y"), ("A", "Y")]
    )
    with pytest.raises(ValueError):
        FunctionApproxBackdoorAgent(2, graph=graph)


# --- act() is a policy over the observation, not a constant fixed at fit time -------------------
#
# Each test below asserts that TWO observations whose within-stratum contrast has opposite SIGNS
# get DIFFERENT actions. Mutating any of these `act` bodies back to `return int(self._best_action)`
# fails them: one of the two observations always disagrees with the marginal argmax.

_FLIP_GRAPH = CausalGraph(directed_edges=[("Z", "A"), ("Z", "Y"), ("A", "Y")])


def _stratum_flip_data(seed: int = 0, n: int = 4000) -> dict[str, np.ndarray]:
    """Confounded logs whose WITHIN-stratum optimum flips: action 1 wins at ``Z=0`` (0.8 vs 0.2),
    action 0 wins at ``Z=1`` (0.9 vs 0.5). Back-door adjustment still leaves one marginal winner
    (``E[Y|do(1)]=0.68 > E[Y|do(0)]=0.48``), so a constant policy must get the ``Z=1`` 40% wrong.
    """
    rng = np.random.default_rng(seed)
    z = (rng.random(n) < 0.4).astype(int)
    a = (rng.random(n) < np.where(z == 1, 0.3, 0.7)).astype(int)  # confounded assignment
    mean = np.where(z == 1, np.where(a == 1, 0.5, 0.9), np.where(a == 1, 0.8, 0.2))
    return {"Z": z, "A": a, "Y": mean + rng.normal(0.0, 0.1, n)}


def test_backdoor_agent_act_conditions_on_the_adjustment_stratum() -> None:
    agent = BackdoorAdjustedAgent(2, graph=_FLIP_GRAPH)
    agent.fit(_stratum_flip_data())
    assert agent.adjustment == ("Z",)
    assert agent.values[1] > agent.values[0]  # the marginal (constant) decision is action 1
    assert agent.act({"Z": 0}) == 1
    assert agent.act({"Z": 1}) == 0  # ... yet action 0 wins inside Z=1
    assert agent.act({}) == 1  # no context supplied -> the marginal decision
    assert agent.act({"Z": 7}) == 1  # stratum never logged -> the marginal decision


def test_discovery_agent_act_conditions_on_the_discovered_adjustment_set() -> None:
    agent = DiscoveryBackdoorAgent(2, variables=("Z", "A", "Y"))
    agent.discover_and_fit(_stratum_flip_data(), tiers=(("Z",), ("A",), ("Y",)))
    assert agent.adjustment == ("Z",)
    assert agent.act({"Z": 0}) == 1
    assert agent.act({"Z": 1}) == 0
    assert agent.act({}) == 1


def test_transport_agent_act_conditions_on_the_source_cell() -> None:
    # E[Y | A=1, Z=z, W=w] - 0.5 = (+0.25 if z else -0.25) - 0.20*[w=1]: arm 1 wins wherever Z=1,
    # while the TRANSPORTED MARGINAL says arm 0 in every cell of the phase diagram.
    env = TransportableConfoundedBandit(gamma=1.0, shift=0.6, seed=0)
    source = env.sample(20_000, domain="source", seed=0)
    target_w = env.sample(20_000, domain="target", seed=1)["W"]
    agent = TransportBackdoorAgent(env.n_actions, graph=env.graph, transport=("W",))
    agent.fit(source, target_covariates={"W": target_w})

    assert agent.act({}) == env.optimal_action(domain="target") == 0  # marginal: never arm 1
    assert agent.act({"Z": 1, "W": 0}) == 1  # ... but arm 1 is better for a Z=1 unit
    assert agent.act({"Z": 0, "W": 0}) == 0
    with pytest.raises(KeyError):
        agent.act({"Z": 1})  # a partial context would silently answer a different query


def test_function_approx_agent_act_plays_the_arm_the_average_rejects() -> None:
    # E[Y|do(1)] ~ 0.38 < 0.5 so the best CONSTANT action is arm 0 -- but arm 1's reward bump peaks
    # at z ~ 0.85, where q(1, z) ~ 1.3. Only a contextual act() can collect it.
    env = ContinuousConfoundedBandit(gamma=1.0, seed=0)
    agent = FunctionApproxBackdoorAgent(env.n_actions, graph=env.graph)
    agent.fit(env.sample(6000, seed=0))
    assert env.optimal_action() == 0
    assert agent.act({}) == 0  # no context -> the marginal decision
    assert agent.act({"Z": 0.85}) == 1  # inside the bump arm 1 is far better
    assert agent.act({"Z": 0.20}) == 0  # outside it arm 0 wins


def test_function_approx_agent_act_before_fit_is_the_default_not_a_crash() -> None:
    env = ContinuousConfoundedBandit(gamma=1.0, seed=0)
    assert FunctionApproxBackdoorAgent(env.n_actions, graph=env.graph).act({"Z": 0.85}) == 0


def test_an_empty_adjustment_set_is_an_honest_constant() -> None:
    # A -> Y with no observed parent of A: the model holds NO context, so the decision is
    # necessarily constant. That is what the act() docstring says, and it must not silently
    # condition on an unrelated key the observation happens to carry.
    agent = BackdoorAdjustedAgent(2, graph=CausalGraph(directed_edges=[("A", "Y")]))
    data = _stratum_flip_data()
    agent.fit(data)
    assert agent.adjustment == ()
    assert agent.act({"Z": 0}) == agent.act({"Z": 1}) == agent.act({}) == 1
