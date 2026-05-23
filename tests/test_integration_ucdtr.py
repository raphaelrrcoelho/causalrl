from causalrl.agents.baselines import NaiveOffline
from causalrl.agents.offline_online import UCDTR
from causalrl.data.dataset import generate_logs
from causalrl.envs.suite.dtr import DTREnv
from causalrl.eval.harness import run_episodes


def _avg_return(agent, n=4000, with_offline=False, seed=0):
    if with_offline:
        logs = generate_logs(DTREnv(seed=seed + 100), n_episodes=4000, seed=seed + 100)
        agent.ingest_offline(logs)
    returns = run_episodes(agent, DTREnv(seed=seed), n_episodes=n, seed=seed)
    return sum(returns) / len(returns)


def test_naive_offline_is_biased_and_suboptimal():
    # Optimal averages 0.75; naive trusts confounded logs, picks treatment 1 everywhere
    # (wrong for subtype 0), and asymptotes near 0.675.
    logs = generate_logs(DTREnv(seed=1), n_episodes=4000, seed=1)
    agent = NaiveOffline(n_states=3, n_actions=2)
    agent.ingest_offline(logs)
    returns = run_episodes(agent, DTREnv(seed=2), n_episodes=4000, seed=2)
    avg = sum(returns) / len(returns)
    assert avg < 0.72  # demonstrably below the optimal 0.75 -> confounding bites


def test_ucdtr_reaches_optimum_and_beats_naive():
    ucdtr_avg = _avg_return(UCDTR(n_states=3, n_actions=2, seed=0), with_offline=True, seed=0)
    naive = NaiveOffline(n_states=3, n_actions=2)
    naive_avg = _avg_return(naive, with_offline=True, seed=0)
    assert ucdtr_avg > 0.70  # learns online to near-optimal
    assert ucdtr_avg > naive_avg  # beats the biased offline baseline
