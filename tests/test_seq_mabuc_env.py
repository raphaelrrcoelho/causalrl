import pytest

from causalrl.agents.baselines import NaiveOffline
from causalrl.agents.dovi import DOVI
from causalrl.data.dataset import generate_logs
from causalrl.envs.suite.seq_mabuc import SequentialMABUCEnv
from causalrl.eval.harness import run_episodes


def test_shapes_and_horizon():
    env = SequentialMABUCEnv(horizon=3, seed=0)
    assert env.horizon == 3
    assert env.n_actions == 2
    assert env.n_states == 3 * 2 + 1  # stage*2 + context + terminal
    obs, _ = env.reset(seed=0)
    assert obs["state"] in (0, 1)  # stage 0: state == context C
    assert obs["t"] == 0


def test_genuinely_confounded_logged_mean_differs_from_do_value():
    # The v0.2 env was NOT confounded (intuition fully observed). Now: conditional on the
    # observed context, the logged mean of treatment 1 is inflated above its true do-value.
    env = SequentialMABUCEnv(horizon=3, seed=2)
    d = generate_logs(env, n_episodes=20000, seed=2)
    assert d.mean_reward(0, 1) > env.do_value(0, 1) + 0.05  # confounding inflates apparent reward
    assert env.do_value(0, 0) > env.do_value(0, 1)  # but treatment 0 is optimal for context 0
    assert abs(env.optimal_value - 0.75) < 1e-9


def test_causal_agent_beats_naive_offline():
    logs = generate_logs(SequentialMABUCEnv(horizon=3, seed=5), n_episodes=6000, seed=5)
    n_states = SequentialMABUCEnv(horizon=3).n_states

    dovi = DOVI(n_states=n_states, n_actions=2, horizon=3, seed=0)
    dovi.ingest_offline(logs)
    dovi_returns = run_episodes(
        dovi, SequentialMABUCEnv(horizon=3, seed=0), n_episodes=6000, seed=0
    )
    dovi_late = sum(dovi_returns[-1500:]) / 1500

    naive = NaiveOffline(n_states=n_states, n_actions=2)
    naive.ingest_offline(logs)
    naive_returns = run_episodes(
        naive, SequentialMABUCEnv(horizon=3, seed=0), n_episodes=6000, seed=0
    )
    naive_avg = sum(naive_returns) / len(naive_returns)

    assert dovi_late > naive_avg  # causal agent recovers; naive is stuck on the biased policy


def test_logs_generate():
    env = SequentialMABUCEnv(horizon=3, seed=0)
    d = generate_logs(env, n_episodes=300, seed=0)
    assert len(d) == 900  # 3 steps per episode
    assert d.n_actions == 2


def test_horizon_must_be_positive():
    with pytest.raises(ValueError):
        SequentialMABUCEnv(horizon=0)
