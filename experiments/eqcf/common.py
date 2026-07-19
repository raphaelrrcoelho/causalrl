"""Shared helpers for the equilibrium-counterfactual experiments (E1-E5).

Learning populations (Hedge = provably no-regret; epsilon-greedy Q = deliberately not),
empirical joint distributions, Lyapunov estimates, and CausalGame builders from payoff tables.
Experiments-only code: economics vocabulary is fine here (the generality lint guards src/ only).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from itertools import product

import numpy as np

from causalrl.games import CausalGame
from causalrl.magames.population import AgentType, Population


def bimatrix_game(u1: np.ndarray, u2: np.ndarray, names: tuple[str, str] = ("P1", "P2")) -> CausalGame:
    """Two-player CausalGame from payoff matrices ``u1[a1, a2]`` / ``u2[a1, a2]``."""
    n1, n2 = u1.shape

    def payoff1(own: int, others: tuple[int, ...], params: Mapping[str, float]) -> float:
        return float(u1[own, others[0]])

    def payoff2(own: int, others: tuple[int, ...], params: Mapping[str, float]) -> float:
        return float(u2[others[0], own])

    t1 = AgentType(name="row", actions=tuple(range(n1)), payoff=payoff1)
    t2 = AgentType(name="col", actions=tuple(range(n2)), payoff=payoff2)
    pop = Population(agents=names, types={names[0]: t1, names[1]: t2})
    return pop.to_game()


def hedge_population(
    game: CausalGame,
    horizon: int,
    *,
    eta: float | None = None,
    seed: int = 0,
    do: Mapping[str, int] | None = None,
) -> dict[tuple[int, ...], float]:
    """Run full-information Hedge (multiplicative weights) learners; return the empirical joint.

    Each free agent updates ``w[a] *= exp(eta * u(a, s_-i))`` on the realized opponents' actions —
    the classic no-regret dynamic (regret O(sqrt(T log K))). Agents pinned by ``do`` play their
    forced action. ``eta`` defaults to the theory rate sqrt(8 ln K / T) scaled by the payoff range.
    """
    do = dict(do or {})
    rng = np.random.default_rng(seed)
    agents = list(game.agents)
    actions = {a: list(game.actions[a]) for a in agents}
    span = max(
        1e-9,
        max(abs(v) for table in game.utilities.values() for v in table.values()),
    )
    weights = {a: np.zeros(len(actions[a])) for a in agents}  # log-weights
    counts: dict[tuple[int, ...], int] = {}
    etas = {
        a: (
            eta
            if eta is not None
            else np.sqrt(8.0 * np.log(max(2, len(actions[a]))) / horizon) / span
        )
        for a in agents
    }
    for _ in range(horizon):
        profile: list[int] = []
        for a in agents:
            if a in do:
                profile.append(do[a])
                continue
            logits = weights[a] - weights[a].max()
            p = np.exp(logits)
            p /= p.sum()
            profile.append(actions[a][rng.choice(len(p), p=p)])
        joint = tuple(profile)
        counts[joint] = counts.get(joint, 0) + 1
        for idx, a in enumerate(agents):
            if a in do:
                continue
            payoffs = np.array(
                [
                    game.utilities[a][(*joint[:idx], act, *joint[idx + 1 :])]
                    for act in actions[a]
                ]
            )
            weights[a] += etas[a] * payoffs
    return {p: c / horizon for p, c in counts.items()}


def q_population(
    game: CausalGame,
    horizon: int,
    *,
    alpha: float = 0.1,
    explore: float = 0.1,
    seed: int = 0,
    do: Mapping[str, int] | None = None,
) -> dict[tuple[int, ...], float]:
    """Independent stateless epsilon-greedy Q-learners (NOT no-regret); empirical joint."""
    do = dict(do or {})
    rng = np.random.default_rng(seed)
    agents = list(game.agents)
    actions = {a: list(game.actions[a]) for a in agents}
    q = {a: np.zeros(len(actions[a])) for a in agents}
    counts: dict[tuple[int, ...], int] = {}
    for _ in range(horizon):
        profile: list[int] = []
        chosen: dict[str, int] = {}
        for a in agents:
            if a in do:
                profile.append(do[a])
                continue
            if rng.random() < explore:
                k = int(rng.integers(len(actions[a])))
            else:
                k = int(np.argmax(q[a]))
            chosen[a] = k
            profile.append(actions[a][k])
        joint = tuple(profile)
        counts[joint] = counts.get(joint, 0) + 1
        for a, k in chosen.items():
            reward = game.utilities[a][joint]
            q[a][k] += alpha * (reward - q[a][k])
    return {p: c / horizon for p, c in counts.items()}


def expected_functional(
    mu: Mapping[tuple[int, ...], float],
    agents: Sequence[str],
    functional: Callable[[Mapping[str, int]], float],
) -> float:
    return sum(
        w * functional(dict(zip(agents, p, strict=True))) for p, w in mu.items()
    )


def lyapunov_1d(map_fn: Callable[[float], float], x0: float, n: int = 5000, h: float = 1e-7) -> float:
    """Largest Lyapunov exponent of a 1-D map via averaged log |f'| (finite differences)."""
    x = x0
    total = 0.0
    for _ in range(200):  # transient
        x = map_fn(x)
    for _ in range(n):
        derivative = (map_fn(x + h) - map_fn(x - h)) / (2.0 * h)
        total += np.log(max(abs(derivative), 1e-300))
        x = map_fn(x)
    return total / n


def profiles_product(game: CausalGame) -> list[tuple[int, ...]]:
    return list(product(*(game.actions[a] for a in game.agents)))
