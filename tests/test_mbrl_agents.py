"""CertifiedPolicyAgent (certify-gated) and BackdoorAdjustedAgent (active deconfounding)."""

from __future__ import annotations

from causalrl.agents.baselines import NaiveOffline
from causalrl.agents.mbrl import (
    BackdoorAdjustedAgent,
    CertifiedPolicyAgent,
    DiscoveryBackdoorAgent,
    TransportBackdoorAgent,
)
from causalrl.data.dataset import ConfoundedTrajectoryDataset, Transition
from causalrl.envs.suite.simpson_bandit import SimpsonBandit
from causalrl.envs.suite.transport_bandit import TransportableConfoundedBandit


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
