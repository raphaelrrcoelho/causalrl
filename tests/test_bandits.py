import numpy as np

from causalrl.agents.bandits import CausalThompsonSampling, NaiveThompsonSampling


def test_naive_agent_acts_and_updates():
    agent = NaiveThompsonSampling(n_arms=2, seed=0)
    obs = {"intuition": 1}
    action = agent.act(obs)
    assert action in (0, 1)
    agent.update(obs, action, reward=1.0)  # should not raise


def test_causal_agent_learns_arm_equals_intuition():
    agent = CausalThompsonSampling(n_arms=2, n_contexts=2, seed=0)
    # Teach it: arm == intuition wins, arm != intuition loses.
    for intuition in (0, 1):
        obs = {"intuition": intuition}
        for _ in range(200):
            agent.update(obs, action=intuition, reward=1.0)
            agent.update(obs, action=1 - intuition, reward=0.0)
    for intuition in (0, 1):
        # after learning, greedy action should equal intuition
        votes = [agent.act({"intuition": intuition}) for _ in range(200)]
        assert sum(v == intuition for v in votes) > 150


def test_seed_is_reproducible_and_independent():
    # Same seed -> identical action sequence, independent of global RNG state.
    obs = {"intuition": 0}
    np.random.seed(12345)
    a = NaiveThompsonSampling(n_arms=2, seed=7)
    seq_a = [a.act(obs) for _ in range(50)]
    np.random.seed(999)  # perturb global RNG; must not affect the seeded agent
    b = NaiveThompsonSampling(n_arms=2, seed=7)
    seq_b = [b.act(obs) for _ in range(50)]
    assert seq_a == seq_b

    # Different seeds -> different sequences (the seed actually flows through).
    c = NaiveThompsonSampling(n_arms=2, seed=8)
    seq_c = [c.act(obs) for _ in range(50)]
    assert seq_a != seq_c
