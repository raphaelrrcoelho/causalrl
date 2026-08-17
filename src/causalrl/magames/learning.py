"""No-regret learning dynamics on a finite game: the producer of the empirical joint.

:func:`~causalrl.magames.cce.cce_regret` measures the maximal unilateral deviation gain of a
realized joint distribution over action profiles — "what a no-regret population drives to 0" — and
:func:`~causalrl.magames.cce.certify_cce_do` turns that measured value into a finite-time
certificate. :func:`run_no_regret` supplies the population: it plays the game for ``rounds`` rounds
with one :class:`~causalrl.agents.no_regret.NoRegretLearner` per free agent and returns the realized
empirical joint in exactly the form those two functions already accept, plus the trace of the
measured regret along the way.

Why the loop closes: each learner's external regret grows sublinearly, so the *time-averaged*
maximal deviation gain — which is what :func:`~causalrl.magames.cce.cce_regret` computes — vanishes,
and the empirical joint approaches the coarse-correlated-equilibrium set of the (possibly
``do``-intervened) game. That convergence is the classical no-regret/CCE correspondence (J. Hannan,
*Approximation to Bayes Risk in Repeated Play*, 1957; N. Cesa-Bianchi, G. Lugosi, *Prediction,
Learning, and Games*, CUP 2006, §7.4); the two learners are cited on
:mod:`causalrl.agents.no_regret`. No external code is ported.

The dynamics are simultaneous independent learners
(:attr:`~causalrl.magames.population.LearnerTopology.INDEPENDENT_LEARNERS`), whatever a
:class:`~causalrl.magames.population.Population` declares: read
:func:`~causalrl.magames.population.topology_max_kind` before claiming a ``Kind`` about them.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import numpy as np

from causalrl.agents.no_regret import MultiplicativeWeights, NoRegretLearner, RegretMatching
from causalrl.games import CausalGame
from causalrl.magames._lp import FloatArray
from causalrl.magames.cce import cce_polytope, cce_regret
from causalrl.magames.population import Population

__all__ = ["NoRegretAlgorithm", "NoRegretRun", "run_no_regret"]

NoRegretAlgorithm = Literal["regret_matching", "multiplicative_weights"]


@dataclass(frozen=True)
class NoRegretRun:
    """Realized play of a no-regret population: the empirical joint plus its regret trace.

    ``weights`` is aligned with ``profiles``, which is the profile order of
    :func:`~causalrl.magames.cce.cce_polytope` for the same ``do`` — so both ``weights`` and
    :attr:`empirical_joint` drop straight into :func:`~causalrl.magames.cce.cce_regret` and
    :func:`~causalrl.magames.cce.certify_cce_do`. ``regret_trace`` pairs a horizon with the measured
    regret of the time average up to it; it is the evidence that the population is learning.
    """

    agents: tuple[str, ...]
    profiles: tuple[tuple[int, ...], ...]
    weights: FloatArray
    rounds: int
    algorithm: str
    do: Mapping[str, int]
    regret_trace: tuple[tuple[int, float], ...]

    @property
    def empirical_joint(self) -> dict[tuple[int, ...], float]:
        """The realized joint distribution as ``{profile: probability}`` over played profiles."""
        return {
            profile: float(weight)
            for profile, weight in zip(self.profiles, self.weights, strict=True)
            if weight > 0.0
        }

    @property
    def regret(self) -> float:
        """The measured realized regret at the full horizon — pass it as ``epsilon``."""
        return self.regret_trace[-1][1]

    def marginal(self, agent: str) -> dict[int, float]:
        """What one agent played: the realized joint summed over everyone else.

        Keyed by the agent's own actions, including any it never played (weight 0.0), so the values
        always sum to 1. An agent pinned by ``do`` did not learn, so its marginal is the
        intervention itself.
        """
        if agent not in self.agents:
            raise KeyError(f"unknown agent: {agent!r}")
        index = self.agents.index(agent)
        played: dict[int, float] = {}
        for profile, weight in zip(self.profiles, self.weights, strict=True):
            action = profile[index]
            played[action] = played.get(action, 0.0) + float(weight)
        return played

    def boundary_mass(self, agent: str) -> float:
        """Realized mass on the extreme actions of ``agent``'s action set.

        Meaningful only when the actions are an *ordered* grid — a discretisation of a quantity,
        not a set of labelled strategies — in which case this is the truncation diagnostic: mass
        pressed against the smallest or largest action means the play may be capped by where the
        grid was stopped rather than settled where the game puts it, and the honest response is to
        widen the grid and re-run. An agent with a single available action scores 1.0.
        """
        played = self.marginal(agent)
        extremes = {min(played), max(played)}
        return sum(played[action] for action in extremes)


def _payoff_range(game: CausalGame, agent: str) -> float:
    values = list(game.utilities[agent].values())
    return max(max(values) - min(values), 1e-12)


def _make_learner(
    game: CausalGame,
    agent: str,
    *,
    algorithm: NoRegretAlgorithm,
    rounds: int,
    learning_rate: float | None,
    explore: float,
    seed: int,
) -> NoRegretLearner:
    n_actions = len(game.actions[agent])
    if algorithm == "regret_matching":
        if learning_rate is not None:
            raise ValueError("regret matching is parameter-free; it has no learning_rate")
        return RegretMatching(n_actions, explore=explore, seed=seed)
    if algorithm == "multiplicative_weights":
        return MultiplicativeWeights(
            n_actions,
            learning_rate=learning_rate,
            horizon=rounds,
            payoff_range=_payoff_range(game, agent),
            explore=explore,
            seed=seed,
        )
    raise ValueError(
        f"unknown algorithm {algorithm!r}: expected 'regret_matching' or 'multiplicative_weights'"
    )


def _trace_marks(rounds: int, points: int) -> frozenset[int]:
    """Horizons at which to measure the running time average: log-spaced, always incl. the last."""
    if points < 1 or rounds < 2:
        return frozenset({rounds})
    grid = np.geomspace(1.0, float(rounds), num=min(points, rounds))
    return frozenset({round(float(v)) for v in grid} | {rounds})


def run_no_regret(
    population: Population | CausalGame,
    rounds: int,
    *,
    do: Mapping[str, int] | None = None,
    algorithm: NoRegretAlgorithm = "regret_matching",
    learning_rate: float | None = None,
    explore: float = 0.0,
    seed: int = 0,
    trace_points: int = 8,
) -> NoRegretRun:
    """Play ``rounds`` rounds of no-regret learning and return the realized empirical joint.

    Every agent not pinned by ``do`` gets its own learner and, each round, the counterfactual payoff
    vector of what its actions would have paid against the profile the rest of the population
    actually played (full-information feedback, available because the game's payoffs are known).
    Agents pinned by ``do`` play their forced action and do not learn — the same restriction
    :func:`~causalrl.magames.cce.cce_polytope` applies, so the returned run is feasible for the
    intervened polytope by construction.

    ``algorithm`` selects :class:`~causalrl.agents.no_regret.RegretMatching` (parameter-free) or
    :class:`~causalrl.agents.no_regret.MultiplicativeWeights` (Hedge; ``learning_rate`` defaults to
    the theory rate for this horizon and the agent's own payoff range). ``explore`` mixes in uniform
    play, ``trace_points`` sets how many log-spaced horizons the regret trace records.

    The canonical use is the finite-time certificate, which assumes nothing about the limit::

        run = run_no_regret(population, 20_000, do={"A2": 0})
        cert = certify_cce_do(game, functional, do={"A2": 0}, no_regret=False, epsilon=run.regret)
    """
    game = population.to_game() if isinstance(population, Population) else population
    if rounds < 1:
        raise ValueError("rounds must be at least 1")
    pinned = dict(do or {})
    polytope = cce_polytope(game, do=pinned)  # validates `do` and fixes the profile order
    agents = game.agents
    action_values = {a: list(game.actions[a]) for a in agents}
    free = [a for a in agents if a not in pinned]
    seeds = np.random.default_rng(seed).integers(0, 2**31 - 1, size=len(free))
    learners = {
        agent: _make_learner(
            game,
            agent,
            algorithm=algorithm,
            rounds=rounds,
            learning_rate=learning_rate,
            explore=explore,
            seed=int(seeds[i]),
        )
        for i, agent in enumerate(free)
    }
    positions = {agent: agents.index(agent) for agent in agents}
    marks = _trace_marks(rounds, trace_points)

    counts: dict[tuple[int, ...], int] = {}
    trace: list[tuple[int, float]] = []
    for played in range(1, rounds + 1):
        profile = tuple(
            pinned[a] if a in pinned else action_values[a][learners[a].act()] for a in agents
        )
        counts[profile] = counts.get(profile, 0) + 1
        for agent in free:
            index = positions[agent]
            table = game.utilities[agent]
            learners[agent].observe(
                np.array(
                    [
                        table[(*profile[:index], action, *profile[index + 1 :])]
                        for action in action_values[agent]
                    ],
                    dtype=np.float64,
                )
            )
        if played in marks:
            running = {p: c / played for p, c in counts.items()}
            trace.append((played, cce_regret(game, running, do=pinned)))

    order = {profile: j for j, profile in enumerate(polytope.profiles)}
    weights = np.zeros(len(polytope.profiles))
    for profile, count in counts.items():
        weights[order[profile]] = count / rounds
    return NoRegretRun(
        agents=agents,
        profiles=polytope.profiles,
        weights=weights,
        rounds=rounds,
        algorithm=algorithm,
        do=pinned,
        regret_trace=tuple(trace),
    )
