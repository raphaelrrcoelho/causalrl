"""Causal game theory: multi-agent causal influence diagrams (taxonomy Task 9).

Represent a normal-form game as a (multi-agent) causal influence diagram — a decision node and a
utility node per agent, with edges from each decision to the utilities it affects — and reason about
equilibria via best responses and pure-strategy Nash equilibria.

Faithful to:

- D. Koller, B. Milch, *Multi-Agent Influence Diagrams for Representing and Solving Games*, Games
  and Economic Behavior 2003 (the MAID representation).
- L. Hammond, J. Fox, T. Everitt, R. Carey, A. Abate, M. Wooldridge, *Reasoning about Causality
  in Games*, Artificial Intelligence 2023 ((structural) causal games / MACIDs).

No external code is ported; the influence diagram is built on our own CausalGraph.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations, product

import numpy as np

from causalrl.exceptions import CausalGraphError
from causalrl.scm.graph import CausalGraph

__all__ = [
    "CausalGame",
    "best_response",
    "decision_node",
    "is_nash_equilibrium",
    "mixed_nash_equilibria",
    "pure_nash_equilibria",
    "utility_node",
]


def decision_node(agent: str) -> str:
    """The decision-node name for ``agent`` in the influence diagram."""
    return f"D_{agent}"


def utility_node(agent: str) -> str:
    """The utility-node name for ``agent`` in the influence diagram."""
    return f"U_{agent}"


@dataclass(frozen=True)
class CausalGame:
    """A finite game as a causal influence diagram.

    ``utilities[agent]`` maps each joint action profile (a tuple in ``agents`` order) to ``agent``'s
    payoff; ``graph`` is the (M)ACID with a decision and utility node per agent.
    """

    agents: tuple[str, ...]
    actions: Mapping[str, tuple[int, ...]]
    utilities: Mapping[str, Mapping[tuple[int, ...], float]]
    graph: CausalGraph

    def __post_init__(self) -> None:
        profiles = set(product(*(self.actions[a] for a in self.agents)))
        for agent in self.agents:
            if agent not in self.utilities:
                raise CausalGraphError(f"missing utilities for agent {agent!r}")
            if set(self.utilities[agent]) != profiles:
                raise CausalGraphError(
                    f"utilities for agent {agent!r} must cover the joint action space exactly"
                )
            for node in (decision_node(agent), utility_node(agent)):
                if node not in self.graph.nodes:
                    raise CausalGraphError(f"influence diagram is missing node {node!r}")


def best_response(game: CausalGame, agent: str, profile: Mapping[str, int]) -> frozenset[int]:
    """The actions maximizing ``agent``'s payoff given the other agents' actions in ``profile``."""
    index = game.agents.index(agent)
    base = [profile[a] for a in game.agents]
    payoffs: dict[int, float] = {}
    for action in game.actions[agent]:
        joint = list(base)
        joint[index] = action
        payoffs[action] = game.utilities[agent][tuple(joint)]
    best = max(payoffs.values())
    return frozenset(action for action, value in payoffs.items() if value == best)


def is_nash_equilibrium(game: CausalGame, profile: Mapping[str, int]) -> bool:
    """Whether every agent's action in ``profile`` is a best response to the others."""
    return all(profile[agent] in best_response(game, agent, profile) for agent in game.agents)


def pure_nash_equilibria(game: CausalGame) -> list[dict[str, int]]:
    """All pure-strategy Nash equilibria, by enumeration over the finite joint action space."""
    equilibria: list[dict[str, int]] = []
    for combo in product(*(game.actions[a] for a in game.agents)):
        profile = dict(zip(game.agents, combo, strict=True))
        if is_nash_equilibrium(game, profile):
            equilibria.append(profile)
    return equilibria


def _solve_rational(rows: list[list[Fraction]], rhs: list[Fraction]) -> list[Fraction] | None:
    """Exactly solve a square rational linear system by Gauss-Jordan; ``None`` if singular."""
    n = len(rows)
    aug = [[*rows[i], rhs[i]] for i in range(n)]
    for col in range(n):
        pivot = next((r for r in range(col, n) if aug[r][col] != 0), None)
        if pivot is None:
            return None
        aug[col], aug[pivot] = aug[pivot], aug[col]
        inverse = aug[col][col]
        aug[col] = [value / inverse for value in aug[col]]
        for r in range(n):
            if r != col and aug[r][col] != 0:
                factor = aug[r][col]
                aug[r] = [a - factor * b for a, b in zip(aug[r], aug[col], strict=True)]
    return [aug[r][n] for r in range(n)]


def _indifference_mix(
    payoff: Mapping[tuple[int, ...], float],
    own_support: tuple[int, ...],
    other_support: tuple[int, ...],
    *,
    own_first: bool,
) -> dict[int, Fraction] | None:
    """The mix over ``other_support`` that makes the row player indifferent across ``own_support``.

    ``own_first`` says whether the row player's action is the first coordinate of the payoff key.
    Returns ``None`` when the mix is infeasible (singular system or a negative probability).
    """

    def value(own: int, other: int) -> Fraction:
        key = (own, other) if own_first else (other, own)
        return Fraction(payoff[key]).limit_denominator(10**9)

    k = len(own_support)
    rows = [
        [value(own_support[i], o) - value(own_support[0], o) for o in other_support]
        for i in range(1, k)
    ]
    rows.append([Fraction(1)] * k)
    rhs = [Fraction(0)] * (k - 1) + [Fraction(1)]
    solution = _solve_rational(rows, rhs)
    if solution is None or any(p < 0 for p in solution):
        return None
    return dict(zip(other_support, solution, strict=True))


def _mixed_nash_two_player(game: CausalGame) -> list[dict[str, dict[int, float]]]:
    """Exact two-player support enumeration over rational arithmetic."""
    p1, p2 = game.agents
    a1, a2 = game.actions[p1], game.actions[p2]
    u1, u2 = game.utilities[p1], game.utilities[p2]

    def u(payoff: Mapping[tuple[int, ...], float], x: int, y: int) -> Fraction:
        return Fraction(payoff[(x, y)]).limit_denominator(10**9)

    equilibria: list[dict[str, dict[int, float]]] = []
    seen: set[tuple[tuple[tuple[int, Fraction], ...], ...]] = set()
    for k in range(1, min(len(a1), len(a2)) + 1):
        for s1 in combinations(a1, k):
            for s2 in combinations(a2, k):
                # P2's mix over s2 makes P1 indifferent across s1, and vice versa.
                mix2 = _indifference_mix(u1, s1, s2, own_first=True)
                mix1 = _indifference_mix(u2, s2, s1, own_first=False)
                if mix2 is None or mix1 is None:
                    continue
                # Best-response check: no off-support action of either player is strictly better.
                v1 = sum((mix2[b] * u(u1, s1[0], b) for b in s2), Fraction(0))
                v2 = sum((mix1[a] * u(u2, a, s2[0]) for a in s1), Fraction(0))
                if any(sum((mix2[b] * u(u1, a, b) for b in s2), Fraction(0)) > v1 for a in a1):
                    continue
                if any(sum((mix1[a] * u(u2, a, b) for a in s1), Fraction(0)) > v2 for b in a2):
                    continue
                full1 = {a: mix1.get(a, Fraction(0)) for a in a1}
                full2 = {b: mix2.get(b, Fraction(0)) for b in a2}
                key = (tuple(sorted(full1.items())), tuple(sorted(full2.items())))
                if key in seen:
                    continue
                seen.add(key)
                equilibria.append(
                    {
                        p1: {a: float(p) for a, p in full1.items()},
                        p2: {b: float(p) for b, p in full2.items()},
                    }
                )
    return equilibria


def _profile_expected_utility(
    game: CausalGame, agent: str, action: int, mix: Mapping[str, Mapping[int, float]]
) -> float:
    """``agent``'s expected payoff for ``action`` given the others' mixed strategies ``mix``."""
    index = game.agents.index(agent)
    others = [a for a in game.agents if a != agent]
    total = 0.0
    for combo in product(*(game.actions[o] for o in others)):
        weight = 1.0
        for other, act in zip(others, combo, strict=True):
            weight *= mix[other].get(act, 0.0)
        if weight == 0.0:
            continue
        joint = [0] * len(game.agents)
        joint[index] = action
        for other, act in zip(others, combo, strict=True):
            joint[game.agents.index(other)] = act
        total += weight * game.utilities[agent][tuple(joint)]
    return total


def _is_epsilon_nash(
    game: CausalGame, mix: Mapping[str, Mapping[int, float]], *, epsilon: float
) -> bool:
    """Whether no agent can gain more than ``epsilon`` by deviating to any pure action."""
    for agent in game.agents:
        payoffs = {
            act: _profile_expected_utility(game, agent, act, mix) for act in game.actions[agent]
        }
        current = sum(mix[agent].get(act, 0.0) * payoffs[act] for act in game.actions[agent])
        if max(payoffs.values()) > current + epsilon:
            return False
    return True


def _solve_support_numeric(
    game: CausalGame, supports: Mapping[str, tuple[int, ...]], *, iterations: int = 200
) -> dict[str, dict[int, float]] | None:
    """Newton-solve the indifference system for one support profile (any number of players).

    Returns ``None`` if it does not converge to a valid probability vector. The unknowns are the
    on-support probabilities; the equations are, per agent, equal expected payoff across the support
    plus a sum-to-one constraint (multilinear for three or more players).
    """
    agents = game.agents
    sizes = [len(supports[a]) for a in agents]
    offsets = [0]
    for size in sizes:
        offsets.append(offsets[-1] + size)
    dim = offsets[-1]

    def unpack(x: np.ndarray) -> dict[str, dict[int, float]]:
        return {
            a: {supports[a][j]: float(x[offsets[i] + j]) for j in range(sizes[i])}
            for i, a in enumerate(agents)
        }

    def residual(x: np.ndarray) -> np.ndarray:
        mix = unpack(x)
        out: list[float] = []
        for i, a in enumerate(agents):
            support = supports[a]
            eu = [_profile_expected_utility(game, a, act, mix) for act in support]
            out.extend(eu[j] - eu[0] for j in range(1, len(support)))
            out.append(float(np.sum(x[offsets[i] : offsets[i + 1]])) - 1.0)
        return np.array(out)

    x = np.concatenate([np.full(size, 1.0 / size) for size in sizes])
    for _ in range(iterations):
        fx = residual(x)
        if np.max(np.abs(fx)) < 1e-12:
            break
        jac = np.zeros((dim, dim))
        for k in range(dim):
            bumped = x.copy()
            bumped[k] += 1e-7
            jac[:, k] = (residual(bumped) - fx) / 1e-7
        try:
            x = x + np.linalg.solve(jac, -fx)
        except np.linalg.LinAlgError:
            return None
    if np.max(np.abs(residual(x))) > 1e-6 or bool(np.any(x < -1e-9)):
        return None
    return unpack(np.clip(x, 0.0, 1.0))


def _mixed_nash_general(game: CausalGame) -> list[dict[str, dict[int, float]]]:
    """Mixed Nash equilibria for three or more players: support enumeration, numerical solve, and
    an ε-Nash verification of every candidate."""
    agents = game.agents
    supports_per_agent = {
        a: [
            combo
            for k in range(1, len(game.actions[a]) + 1)
            for combo in combinations(game.actions[a], k)
        ]
        for a in agents
    }
    equilibria: list[dict[str, dict[int, float]]] = []
    seen: set[tuple[tuple[float, ...], ...]] = set()
    for profile in product(*(supports_per_agent[a] for a in agents)):
        supports = dict(zip(agents, profile, strict=True))
        solution = _solve_support_numeric(game, supports)
        if solution is None or not _is_epsilon_nash(game, solution, epsilon=1e-6):
            continue
        full = {a: {act: solution[a].get(act, 0.0) for act in game.actions[a]} for a in agents}
        signature = tuple(tuple(round(full[a][act], 6) for act in game.actions[a]) for a in agents)
        if signature in seen:
            continue
        seen.add(signature)
        equilibria.append(full)
    return equilibria


def mixed_nash_equilibria(game: CausalGame) -> list[dict[str, dict[int, float]]]:
    """All mixed-strategy Nash equilibria (pure and properly mixed), by support enumeration.

    Two-player games are solved **exactly** over rational arithmetic (:class:`fractions.Fraction`),
    so symmetric games yield exact mixes (matching pennies gives ``0.5``/``0.5``). Games with three
    or more agents are solved by support enumeration with a **numerical** (Newton) solve of the
    multilinear indifference system; every returned profile is then verified to be an ε-Nash
    equilibrium (no agent can gain more than ``1e-6`` by deviating to a pure action). Each
    equilibrium maps an agent to ``{action: probability}`` with off-support actions at zero.

    Assumes a non-degenerate game; degenerate games may admit a continuum of equilibria, of which
    only support-extreme points are enumerated, and the numerical solver may miss a support whose
    system is ill-conditioned. Raises :class:`CausalGraphError` for fewer than two agents (use
    :func:`pure_nash_equilibria` for pure equilibria of any game).

    Faithful to the support-enumeration method (R. Porter, E. Nudelman, Y. Shoham, *Simple Search
    Methods for Finding a Nash Equilibrium*, Games and Economic Behavior 2008; B. von Stengel,
    *Computing Equilibria for Two-Person Games*, Handbook of Game Theory 2002). No code is ported.
    """
    if len(game.agents) < 2:
        raise CausalGraphError("mixed_nash_equilibria needs at least two agents")
    if len(game.agents) == 2:
        return _mixed_nash_two_player(game)
    return _mixed_nash_general(game)
