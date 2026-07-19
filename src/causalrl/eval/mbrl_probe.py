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
from dataclasses import dataclass

import numpy as np

from causalrl.agents.baselines import NaiveOffline
from causalrl.agents.dovi import DOVI
from causalrl.agents.mbrl import (
    BackdoorAdjustedAgent,
    DiscoveryBackdoorAgent,
    FunctionApproxBackdoorAgent,
    TransportBackdoorAgent,
)
from causalrl.data.dataset import ConfoundedTrajectoryDataset, Transition, generate_logs
from causalrl.envs.suite.continuous_confounded import ContinuousConfoundedBandit
from causalrl.envs.suite.seq_dtr import SequentialDTREnv
from causalrl.envs.suite.simpson_bandit import SimpsonBandit
from causalrl.envs.suite.transport_bandit import TransportableConfoundedBandit
from causalrl.eval.benchmark import BenchmarkEstimate
from causalrl.eval.harness import run_episodes


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


def run_m1_discovery_gate(
    *,
    seeds: Sequence[int] = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    n: int = 5000,
) -> dict[str, BenchmarkEstimate]:
    """M1: an agent that *learns* the structure (interventional discovery) then adjusts, vs naive.

    Keys: ``discovery`` (discover -> back-door-adjust), ``naive`` (marginal), ``optimal`` (oracle).
    GO iff ``discovery.mean > naive.mean`` (the recovered adjustment set is checked in the tests).
    """
    rows: dict[str, list[float]] = {"discovery": [], "naive": [], "optimal": []}
    for seed in seeds:
        env = SimpsonBandit(seed=seed)
        observational = env.sample(n, seed=seed)
        transitions = [
            Transition(0, int(a), float(y), 0, True)
            for a, y in zip(observational["A"], observational["Y"], strict=True)
        ]
        dataset = ConfoundedTrajectoryDataset(transitions, n_states=1, n_actions=2)
        naive = NaiveOffline(env.n_states, env.n_actions)
        naive.ingest_offline(dataset)

        agent = DiscoveryBackdoorAgent(env.n_actions, variables=("Z", "A", "Y"))
        agent.discover_and_fit(observational, tiers=(("Z",), ("A",), ("Y",)))

        rows["discovery"].append(env.true_action_value(agent.act({"state": 0})))
        rows["naive"].append(env.true_action_value(naive.act({"state": 0})))
        rows["optimal"].append(env.optimal_value)

    return {
        name: BenchmarkEstimate.from_values(name, seeds=seeds, values=values)
        for name, values in rows.items()
    }


def run_m1b_dtr_gate(
    *,
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
    horizon: int = 2,
    n_episodes: int = 8000,
) -> dict[str, BenchmarkEstimate]:
    """M1b: deconfounded sequential planning vs naive on the confounded medicine DTR.

    On ``SequentialDTREnv`` (a hidden comorbidity U, a confounded clinician who plays ``a = U``, and
    a foresight gap), the deconfounded value-iteration agent ``DOVI`` (an existing library agent) is
    trained on confounded logs and rolled out; ``NaiveOffline`` is the confounded baseline. Keys:
    ``causal`` (DOVI late-window mean return), ``naive``, ``optimal``. GO iff ``causal > naive``.
    """
    n_states = SequentialDTREnv(horizon=horizon).n_states
    optimal = SequentialDTREnv(horizon=horizon).optimal_value
    tail = max(1, n_episodes // 4)
    rows: dict[str, list[float]] = {"causal": [], "naive": [], "optimal": []}
    for seed in seeds:
        logs = generate_logs(
            SequentialDTREnv(horizon=horizon, seed=seed + 11), n_episodes=n_episodes, seed=seed + 11
        )
        dovi = DOVI(
            n_states=n_states,
            n_actions=2,
            horizon=horizon,
            seed=seed,
            transition_assumption="unconfounded",
        )
        dovi.ingest_offline(logs)
        dovi_returns = run_episodes(
            dovi, SequentialDTREnv(horizon=horizon, seed=seed), n_episodes=n_episodes, seed=seed
        )
        naive = NaiveOffline(n_states=n_states, n_actions=2)
        naive.ingest_offline(logs)
        naive_returns = run_episodes(
            naive, SequentialDTREnv(horizon=horizon, seed=seed), n_episodes=n_episodes, seed=seed
        )
        rows["causal"].append(sum(dovi_returns[-tail:]) / tail)
        rows["naive"].append(sum(naive_returns) / len(naive_returns))
        rows["optimal"].append(optimal)
    return {
        name: BenchmarkEstimate.from_values(name, seeds=seeds, values=values)
        for name, values in rows.items()
    }


@dataclass(frozen=True)
class PhaseDiagram:
    """A 2-D (gamma x shift) sweep of the causal-minus-naive post-shift gap.

    ``gap``/``causal``/``naive`` are per-cell :class:`BenchmarkEstimate`s keyed ``(gamma, shift)``;
    ``gap_grid`` is the mean-gap matrix (rows=``gammas``, cols=``shifts``); the two ``monotone_*``
    flags say whether that matrix is nondecreasing (within ``mono_tol``) along each axis.
    """

    gammas: tuple[float, ...]
    shifts: tuple[float, ...]
    gap: dict[tuple[float, float], BenchmarkEstimate]
    causal: dict[tuple[float, float], BenchmarkEstimate]
    naive: dict[tuple[float, float], BenchmarkEstimate]
    gap_grid: list[list[float]]
    monotone_in_gamma: bool
    monotone_in_shift: bool


def run_m2_phase_diagram(
    *,
    gammas: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
    shifts: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
    seeds: Sequence[int] = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    n: int = 6000,
    mono_tol: float = 0.02,
) -> PhaseDiagram:
    """M2: 2-D phase diagram of the causal-minus-naive post-shift gap over ``gamma`` x ``shift``.

    Per cell, a :class:`~causalrl.agents.mbrl.TransportBackdoorAgent` (deconfound + transport) and
    a ``NaiveOffline`` marginal agent are fit on SOURCE logs and evaluated at the TARGET; the gap is
    ``causal_target - naive_target``. Returns per-cell :class:`BenchmarkEstimate`s keyed by
    ``(gamma, shift)`` plus the mean-gap grid and whether it is monotone nondecreasing (within
    ``mono_tol``) in each axis -- the "confounding bites where theory predicts" signature.
    """
    gammas = tuple(gammas)
    shifts = tuple(shifts)
    seeds = tuple(seeds)
    gap: dict[tuple[float, float], BenchmarkEstimate] = {}
    causal: dict[tuple[float, float], BenchmarkEstimate] = {}
    naive: dict[tuple[float, float], BenchmarkEstimate] = {}
    for g in gammas:
        for s in shifts:
            causal_vals: list[float] = []
            naive_vals: list[float] = []
            gap_vals: list[float] = []
            for seed in seeds:
                env = TransportableConfoundedBandit(gamma=g, shift=s, seed=seed)
                source = env.sample(n, domain="source", seed=seed)
                target_w = env.sample(n, domain="target", seed=seed + 10_000)["W"]
                agent = TransportBackdoorAgent(env.n_actions, graph=env.graph, transport=("W",))
                agent.fit(source, target_covariates={"W": target_w})
                causal_target = env.true_action_value(agent.act({"state": 0}), domain="target")

                transitions = [
                    Transition(0, int(a), float(y), 0, True)
                    for a, y in zip(source["A"], source["Y"], strict=True)
                ]
                dataset = ConfoundedTrajectoryDataset(transitions, n_states=1, n_actions=2)
                marginal = NaiveOffline(env.n_states, env.n_actions)
                marginal.ingest_offline(dataset)
                naive_target = env.true_action_value(marginal.act({"state": 0}), domain="target")

                causal_vals.append(causal_target)
                naive_vals.append(naive_target)
                gap_vals.append(causal_target - naive_target)
            key = (g, s)
            causal[key] = BenchmarkEstimate.from_values(
                f"causal_g{g}_s{s}", seeds=seeds, values=causal_vals
            )
            naive[key] = BenchmarkEstimate.from_values(
                f"naive_g{g}_s{s}", seeds=seeds, values=naive_vals
            )
            gap[key] = BenchmarkEstimate.from_values(f"gap_g{g}_s{s}", seeds=seeds, values=gap_vals)
    grid = np.array([[gap[(g, s)].mean for s in shifts] for g in gammas])
    mono_gamma = bool(np.all(np.diff(grid, axis=0) >= -mono_tol)) if len(gammas) > 1 else True
    mono_shift = bool(np.all(np.diff(grid, axis=1) >= -mono_tol)) if len(shifts) > 1 else True
    return PhaseDiagram(
        gammas=gammas,
        shifts=shifts,
        gap=gap,
        causal=causal,
        naive=naive,
        gap_grid=grid.tolist(),
        monotone_in_gamma=mono_gamma,
        monotone_in_shift=mono_shift,
    )


def run_m3_function_approx_gate(
    *,
    seeds: Sequence[int] = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    gamma: float = 1.0,
    n: int = 6000,
) -> dict[str, BenchmarkEstimate]:
    """M3: a continuous-confounder function-approximation agent vs naive on a nonlinear reward.

    On ``ContinuousConfoundedBandit`` (continuous ``Z ~ Uniform``, a nonlinear arm-1 bump, and
    behavior that over-samples arm 1 near the bump), a ``FunctionApproxBackdoorAgent`` (ridge
    on RBF features + back-door integration) recovers the true low value of arm 1 and keeps the safe
    arm 0; the ``NaiveOffline`` marginal is fooled into arm 1. Keys: ``causal``, ``naive``,
    ``optimal``. GO iff ``causal.mean > naive.mean``.
    """
    rows: dict[str, list[float]] = {"causal": [], "naive": [], "optimal": []}
    for seed in seeds:
        env = ContinuousConfoundedBandit(gamma=gamma, seed=seed)
        data = env.sample(n, seed=seed)

        agent = FunctionApproxBackdoorAgent(env.n_actions, graph=env.graph)
        agent.fit(data)

        transitions = [
            Transition(0, int(a), float(y), 0, True)
            for a, y in zip(data["A"], data["Y"], strict=True)
        ]
        dataset = ConfoundedTrajectoryDataset(transitions, n_states=1, n_actions=2)
        naive = NaiveOffline(env.n_states, env.n_actions)
        naive.ingest_offline(dataset)

        rows["causal"].append(env.true_action_value(agent.act({"state": 0})))
        rows["naive"].append(env.true_action_value(naive.act({"state": 0})))
        rows["optimal"].append(env.optimal_value())
    return {
        name: BenchmarkEstimate.from_values(name, seeds=seeds, values=values)
        for name, values in rows.items()
    }
