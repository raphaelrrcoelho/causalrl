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


def mixed_nash_equilibria(game: CausalGame) -> list[dict[str, dict[int, float]]]:
    """All Nash equilibria (pure and properly mixed) of a TWO-player game, by support enumeration.

    For each pair of equal-size action supports, the players' indifference conditions form a square
    rational linear system solved exactly with :class:`fractions.Fraction`; a candidate is kept when
    both mixes are valid probability vectors and no off-support action is a strictly better
    response. Each equilibrium maps an agent to ``{action: probability}`` (off-support actions carry
    probability zero). Computation is exact internally and returned as floats, so symmetric games
    yield exact mixes (matching pennies returns ``0.5``/``0.5``).

    Raises :class:`NotImplementedError` for games without exactly two agents: mixed-equilibrium
    computation for more than two players needs nonlinear solvers and is out of scope (use
    :func:`pure_nash_equilibria`, which handles any number of agents). Assumes a non-degenerate
    game; degenerate games may admit a continuum of equilibria, of which only support-extreme
    points are enumerated.

    Faithful to the support-enumeration method (R. Porter, E. Nudelman, Y. Shoham, *Simple Search
    Methods for Finding a Nash Equilibrium*, Games and Economic Behavior 2008; B. von Stengel,
    *Computing Equilibria for Two-Person Games*, Handbook of Game Theory 2002). No code is ported.
    """
    if len(game.agents) != 2:
        raise NotImplementedError(
            "mixed_nash_equilibria supports two-player games; "
            "use pure_nash_equilibria for the general case"
        )
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
