from causalrl.agents.baselines import NaiveOffline
from causalrl.agents.dovi import DOVI
from causalrl.data.dataset import generate_logs
from causalrl.envs.suite.dtr import DTREnv
from causalrl.envs.suite.gridworld import ConfoundedGridworld
from causalrl.envs.suite.seq_dtr import SequentialDTREnv
from causalrl.eval.harness import run_episodes


def test_dovi_reaches_optimum_and_beats_naive_on_dtr():
    logs = generate_logs(DTREnv(seed=11), n_episodes=4000, seed=11)
    dovi = DOVI(n_states=3, n_actions=2, horizon=1, seed=0)
    dovi.ingest_offline(logs)
    dovi_returns = run_episodes(dovi, DTREnv(seed=0), n_episodes=4000, seed=0)
    dovi_late = sum(dovi_returns[-1000:]) / 1000

    naive = NaiveOffline(n_states=3, n_actions=2)
    naive.ingest_offline(logs)
    naive_returns = run_episodes(naive, DTREnv(seed=0), n_episodes=4000, seed=0)
    naive_avg = sum(naive_returns) / len(naive_returns)

    assert dovi_late > 0.70  # DOVI (no eps) converges to near-optimal
    assert dovi_late > naive_avg  # and beats the biased naive baseline


def test_horizon_dovi_beats_myopic_and_naive_on_sequential_dtr():
    # Optimal expected return is 1.05; naive (confounded) and myopic (no lookahead) ~0.85.
    logs = generate_logs(SequentialDTREnv(horizon=2, seed=11), n_episodes=8000, seed=11)
    n_states = SequentialDTREnv(horizon=2).n_states  # 5

    full = DOVI(
        n_states=n_states,
        n_actions=2,
        horizon=2,
        seed=0,
        transition_assumption="unconfounded",
    )
    full.ingest_offline(logs)
    full_returns = run_episodes(full, SequentialDTREnv(horizon=2, seed=0), n_episodes=8000, seed=0)
    full_late = sum(full_returns[-2000:]) / 2000

    myopic = DOVI(n_states=n_states, n_actions=2, horizon=1, seed=0)  # immediate-ceiling / v0.2
    myopic.ingest_offline(logs)
    myopic_returns = run_episodes(
        myopic, SequentialDTREnv(horizon=2, seed=1), n_episodes=8000, seed=1
    )
    myopic_late = sum(myopic_returns[-2000:]) / 2000

    naive = NaiveOffline(n_states=n_states, n_actions=2)
    naive.ingest_offline(logs)
    naive_returns = run_episodes(
        naive, SequentialDTREnv(horizon=2, seed=2), n_episodes=8000, seed=2
    )
    naive_avg = sum(naive_returns) / len(naive_returns)

    assert full_late > 1.00  # near the 1.05 optimum
    assert full_late > naive_avg + 0.10  # beats the confounded baseline by a concrete margin
    assert full_late > myopic_late + 0.10  # and beats the myopic (no-bootstrap) baseline


def test_horizon_dovi_beats_myopic_and_naive_on_gridworld():
    env = ConfoundedGridworld(size=3, seed=0)
    logs = generate_logs(ConfoundedGridworld(size=3, seed=7), n_episodes=2000, seed=7)

    full = DOVI(
        n_states=env.n_states,
        n_actions=4,
        horizon=env.horizon,
        seed=0,
        allow_heuristic=True,
    )
    full.ingest_offline(logs)
    full_returns = run_episodes(full, env, n_episodes=4000, seed=0)
    full_late = sum(full_returns[-500:]) / 500

    myopic = DOVI(n_states=env.n_states, n_actions=4, horizon=1, seed=0)  # immediate-ceiling
    myopic.ingest_offline(logs)
    myopic_returns = run_episodes(
        myopic, ConfoundedGridworld(size=3, seed=0), n_episodes=4000, seed=0
    )
    myopic_late = sum(myopic_returns[-500:]) / 500

    naive = NaiveOffline(n_states=env.n_states, n_actions=4)
    naive.ingest_offline(logs)
    naive_returns = run_episodes(
        naive, ConfoundedGridworld(size=3, seed=0), n_episodes=4000, seed=0
    )
    naive_avg = sum(naive_returns) / len(naive_returns)

    # Horizon-indexed backup propagates goal value; the myopic ceiling agent has none to bootstrap.
    assert full_late - myopic_late >= 0.15
    assert full_late > naive_avg
