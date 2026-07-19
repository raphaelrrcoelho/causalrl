"""M0 kill-gate (b): does an ACTIVE deconfounded agent beat a naive marginal one on a Simpson's-
paradox bandit — recovering the interventional optimum from confounded logs?

Reports each agent's interventional value plus the oracle optimum. The verdict is READ from the
report (``causal.mean`` vs ``naive.mean``), not asserted here.

Supersedes the certify-gated attempt (NO-GO: an abstention rule tops out at the behavior policy). An
active back-door agent has no such ceiling — given an observed admissible adjustment set it
recovers the true optimum, which a naive marginal agent cannot.
"""

from __future__ import annotations

from collections.abc import Sequence

from causalrl.agents.baselines import NaiveOffline
from causalrl.agents.mbrl import BackdoorAdjustedAgent
from causalrl.data.dataset import ConfoundedTrajectoryDataset, Transition
from causalrl.envs.suite.simpson_bandit import SimpsonBandit
from causalrl.eval.benchmark import BenchmarkEstimate


def run_m0_kill_gate(
    *,
    seeds: Sequence[int] = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    n: int = 5000,
) -> dict[str, BenchmarkEstimate]:
    """Back-door-adjusted vs naive-marginal on the Simpson bandit; report interventional values.

    Keys: ``causal`` (back-door-adjusted agent), ``naive`` (marginal ``E[Y|A]`` agent), ``optimal``
    (oracle). GO iff ``causal.mean > naive.mean``.
    """
    rows: dict[str, list[float]] = {"causal": [], "naive": [], "optimal": []}
    for seed in seeds:
        env = SimpsonBandit(seed=seed)
        data = env.sample(n, seed=seed)

        # Naive marginal agent: sees only (action, reward) in a single context -> E[Y | A].
        transitions = [
            Transition(0, int(a), float(y), 0, True)
            for a, y in zip(data["A"], data["Y"], strict=True)
        ]
        dataset = ConfoundedTrajectoryDataset(transitions, n_states=1, n_actions=2)
        naive = NaiveOffline(env.n_states, env.n_actions)
        naive.ingest_offline(dataset)

        # Causal agent: back-door-adjusts for the observed confounder Z.
        causal = BackdoorAdjustedAgent(env.n_actions, graph=env.graph)
        causal.fit(data)

        rows["causal"].append(env.true_action_value(causal.act({"state": 0})))
        rows["naive"].append(env.true_action_value(naive.act({"state": 0})))
        rows["optimal"].append(env.optimal_value)

    return {
        name: BenchmarkEstimate.from_values(name, seeds=seeds, values=values)
        for name, values in rows.items()
    }
