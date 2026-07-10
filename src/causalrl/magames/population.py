"""Typed agent populations for multi-agent causal games (plan §8.1).

An :class:`AgentType` is a template — an action set plus a payoff template over the joint profile —
shared across the instances of that type (explicit parameter sharing). A :class:`Population`
instantiates ``N`` named agents from types and materialises the shipped
:class:`~causalrl.games.CausalGame` (a MAID with a decision and utility node per agent). A
:class:`LearnerTopology` labels how the population learns and, per invariant I2, caps the epistemic
``Kind`` any certificate about it may claim: a single learner in a fixed population supports
``IDENTIFIED`` claims (Phase-1 machinery applies to it), while simultaneous independent learners or
centralised training are ``EMPIRICAL`` only.
"""

from __future__ import annotations

import enum
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from itertools import product

from causalrl.certify.certificate import Kind
from causalrl.games import CausalGame, decision_node, utility_node
from causalrl.scm.graph import CausalGraph

__all__ = [
    "AgentType",
    "LearnerTopology",
    "Population",
    "topology_max_kind",
]

# payoff(own_action, others_actions_in_agent_order, params) -> float
PayoffTemplate = Callable[[int, tuple[int, ...], Mapping[str, float]], float]


def _empty_params() -> dict[str, Mapping[str, float]]:
    return {}


class LearnerTopology(enum.Enum):
    """How a population learns; caps the strongest ``Kind`` a certificate about it may claim."""

    SINGLE_LEARNER = "single_learner"  # one learner, fixed population -> IDENTIFIED-capable
    INDEPENDENT_LEARNERS = "independent_learners"  # simultaneous learners -> EMPIRICAL only
    CENTRALIZED = "centralized"  # centralised training -> EMPIRICAL only


_TOPOLOGY_MAX_KIND: dict[LearnerTopology, Kind] = {
    LearnerTopology.SINGLE_LEARNER: Kind.IDENTIFIED,
    LearnerTopology.INDEPENDENT_LEARNERS: Kind.EMPIRICAL,
    LearnerTopology.CENTRALIZED: Kind.EMPIRICAL,
}


def topology_max_kind(topology: LearnerTopology) -> Kind:
    """The strongest epistemic :class:`~causalrl.certify.certificate.Kind` ``topology`` licenses."""
    return _TOPOLOGY_MAX_KIND[topology]


@dataclass(frozen=True)
class AgentType:
    """A template shared by a class of agents: an action set and a payoff over the joint profile.

    ``payoff(own_action, others_actions, params)`` receives the other agents' actions in population
    order (self excluded) and the instance's ``params``; sharing one ``AgentType`` across instances
    *is* the parameter sharing.
    """

    name: str
    actions: tuple[int, ...]
    payoff: PayoffTemplate


@dataclass(frozen=True)
class Population:
    """``N`` named agents instantiated from :class:`AgentType` templates with explicit sharing."""

    agents: tuple[str, ...]
    types: Mapping[str, AgentType]
    params: Mapping[str, Mapping[str, float]] = field(default_factory=_empty_params)
    topology: LearnerTopology = LearnerTopology.SINGLE_LEARNER

    def __post_init__(self) -> None:
        missing = [a for a in self.agents if a not in self.types]
        if missing:
            raise ValueError(f"no AgentType for agents {missing}")

    @property
    def max_kind(self) -> Kind:
        """The strongest ``Kind`` a certificate about this population may claim (per topology)."""
        return topology_max_kind(self.topology)

    def actions(self) -> dict[str, tuple[int, ...]]:
        return {a: self.types[a].actions for a in self.agents}

    def to_game(self) -> CausalGame:
        """Materialise the normal-form :class:`~causalrl.games.CausalGame` (small ``N``).

        Per-agent graphs stay factored in the templates; this fully materialises the joint payoff
        tables and a MAID with an edge from every decision to every utility (a general normal-form
        game: each payoff depends on all actions).
        """
        agents = self.agents
        action_map = self.actions()
        utilities: dict[str, dict[tuple[int, ...], float]] = {}
        for a in agents:
            t = self.types[a]
            idx = agents.index(a)
            p: Mapping[str, float] = self.params.get(a, {})
            table: dict[tuple[int, ...], float] = {}
            for combo in product(*(action_map[x] for x in agents)):
                others = tuple(c for j, c in enumerate(combo) if j != idx)
                table[combo] = float(t.payoff(combo[idx], others, p))
            utilities[a] = table
        nodes = [n for a in agents for n in (decision_node(a), utility_node(a))]
        edges = [(decision_node(i), utility_node(j)) for i in agents for j in agents]
        graph = CausalGraph(directed_edges=edges, nodes=nodes)
        return CausalGame(agents=agents, actions=action_map, utilities=utilities, graph=graph)
