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
from itertools import product

from causalrl.exceptions import CausalGraphError
from causalrl.scm.graph import CausalGraph

__all__ = [
    "CausalGame",
    "best_response",
    "decision_node",
    "is_nash_equilibrium",
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
