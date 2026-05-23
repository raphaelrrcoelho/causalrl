from causalrl.agents.deep_deconfounded import DeepDeconfoundedQ
from causalrl.data.dataset import ConfoundedTrajectoryDataset, Transition


def test_acts_in_range_and_updates():
    agent = DeepDeconfoundedQ(n_states=7, n_actions=2, seed=0)
    a = agent.act({"state": 0, "t": 0})
    assert a in (0, 1)
    agent.update({"state": 0, "t": 0}, a, reward=1.0)


def test_targets_are_clamped_to_causal_bounds():
    transitions = [Transition(0, 1, 0.0, 1, True)] * 10 + [Transition(0, 0, 1.0, 1, True)]
    d = ConfoundedTrajectoryDataset(transitions, n_states=7, n_actions=2)
    agent = DeepDeconfoundedQ(n_states=7, n_actions=2, seed=0)
    agent.ingest_offline(d)
    _lo, hi = agent.bound(0, 1)
    clamped = agent.clamp_target(state=0, action=1, target=5.0)
    assert clamped <= hi + 1e-6


def test_learns_intuition_on_seq_mabuc():
    from causalrl.envs.suite.seq_mabuc import SequentialMABUCEnv
    from causalrl.eval.harness import run_episodes

    env = SequentialMABUCEnv(horizon=3, seed=0)
    agent = DeepDeconfoundedQ(n_states=env.n_states, n_actions=2, seed=0)
    returns = run_episodes(agent, env, n_episodes=3000, seed=0)
    early = sum(returns[:500]) / 500
    late = sum(returns[-500:]) / 500
    assert late > early
