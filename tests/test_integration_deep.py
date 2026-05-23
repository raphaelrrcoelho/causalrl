from causalrl.agents.baselines import NaiveOffline
from causalrl.agents.deep_deconfounded import DeepDeconfoundedQ
from causalrl.data.dataset import generate_logs
from causalrl.envs.suite.dtr import DTREnv
from causalrl.eval.harness import run_episodes


def test_deep_agent_beats_naive_offline_on_dtr():
    logs = generate_logs(DTREnv(seed=5), n_episodes=4000, seed=5)

    deep = DeepDeconfoundedQ(n_states=3, n_actions=2, seed=0)
    deep.ingest_offline(logs)
    # train (with exploration)
    run_episodes(deep, DTREnv(seed=0), n_episodes=4000, seed=0)
    # evaluate the LEARNED policy greedily (no exploration)
    deep._eps = 0.0
    eval_returns = run_episodes(deep, DTREnv(seed=1), n_episodes=2000, seed=1)
    deep_eval = sum(eval_returns) / len(eval_returns)

    naive = NaiveOffline(n_states=3, n_actions=2)
    naive.ingest_offline(logs)
    naive_returns = run_episodes(naive, DTREnv(seed=1), n_episodes=2000, seed=1)
    naive_avg = sum(naive_returns) / len(naive_returns)

    assert deep_eval > 0.70  # learned policy is near-optimal
    assert deep_eval > naive_avg  # and beats the biased naive baseline
