import pytest

from causalrl.data.dataset import generate_logs
from causalrl.envs.suite.seq_dtr import SequentialDTREnv


def test_shapes_and_horizon():
    env = SequentialDTREnv(horizon=2, seed=0)
    assert env.horizon == 2
    assert env.n_actions == 2
    assert env.n_states == 2 * 2 + 1  # stage*2 + Z, plus terminal
    obs, _ = env.reset(seed=0)
    assert obs["state"] in (0, 1)  # stage 0: state == Z
    assert obs["t"] == 0


def test_dv_table_and_optimal_value_have_foresight_gap():
    env = SequentialDTREnv(horizon=2)
    # Immediately-greedy at Z=0 is a=1 (higher per-stage dv)...
    assert env.dv(0, 1) > env.dv(0, 0)
    # ...but the lookahead-optimal first action at Z=0 is a=0 (foresight gap).
    assert env.do_value(0, 0, stage=0) > env.do_value(0, 1, stage=0)
    # Optimal expected return (random initial Z): 0.5*1.15 + 0.5*0.95 = 1.05.
    assert abs(env.optimal_value - 1.05) < 1e-9


def test_last_stage_do_value_is_immediate():
    env = SequentialDTREnv(horizon=2)
    for z in (0, 1):
        for a in (0, 1):
            assert abs(env.do_value(z, a, stage=1) - env.dv(z, a)) < 1e-9


def test_naive_offline_estimate_is_confounded_toward_treatment_one():
    env = SequentialDTREnv(horizon=2, seed=2)
    d = generate_logs(env, n_episodes=20000, seed=2)
    # state 0 == (stage 0, Z 0). U inflates a=1's apparent reward above its true do-value.
    assert d.mean_reward(0, 1) > env.dv(0, 1) + 0.05
    assert d.mean_reward(0, 1) > d.mean_reward(0, 0)


def test_logs_generate():
    env = SequentialDTREnv(horizon=2, seed=0)
    d = generate_logs(env, n_episodes=300, seed=0)
    assert len(d) == 600  # 2 stages per episode
    assert d.n_actions == 2


def test_horizon_must_be_positive():
    with pytest.raises(ValueError):
        SequentialDTREnv(horizon=0)
