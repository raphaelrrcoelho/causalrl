"""The confounded imitation environment: expert vs wrong-action rewards."""

from __future__ import annotations

import numpy as np

from causalrl.envs.suite.imitation import ImitationEnv, expert_policy, generate_demonstrations


def test_expert_reward_is_high() -> None:
    env = ImitationEnv(seed=0)
    total = 0.0
    n = 4000
    for t in range(n):
        obs, _ = env.reset(seed=t)
        _, reward, terminated, _, _ = env.step(expert_policy(obs))
        assert terminated
        total += reward
    assert abs(total / n - 0.9) < 0.03


def test_wrong_action_reward_is_low() -> None:
    env = ImitationEnv(seed=2)
    total = 0.0
    n = 4000
    for t in range(n):
        obs, _ = env.reset(seed=1000 + t)
        _, reward, _, _, _ = env.step(1 - obs["W"])  # always the wrong arm
        total += reward
    assert abs(total / n - 0.1) < 0.03


def test_demonstrations_are_expert_rollouts() -> None:
    demos = generate_demonstrations(ImitationEnv(seed=0), n=500, seed=0)
    assert set(demos) == {"W", "A", "Y"}
    assert all(len(demos[key]) == 500 for key in demos)
    assert np.array_equal(demos["A"], demos["W"])  # the expert plays A = W
