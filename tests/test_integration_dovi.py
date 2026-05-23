from causalrl.agents.baselines import NaiveOffline
from causalrl.agents.dovi import DOVI
from causalrl.data.dataset import generate_logs
from causalrl.envs.suite.dtr import DTREnv
from causalrl.envs.suite.gridworld import ConfoundedGridworld
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


def test_dovi_learns_on_gridworld():
    env = ConfoundedGridworld(size=3, seed=0)
    logs = generate_logs(ConfoundedGridworld(size=3, seed=7), n_episodes=2000, seed=7)
    dovi = DOVI(n_states=env.n_states, n_actions=4, horizon=env.horizon, seed=0)
    dovi.ingest_offline(logs)
    returns = run_episodes(dovi, env, n_episodes=4000, seed=0)
    early = sum(returns[:500]) / 500
    late = sum(returns[-500:]) / 500
    assert late >= early
