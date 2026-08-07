"""Possibly-Optimal Minimal Intervention Sets (POMIS) and Minimal Intervention Sets (MIS).

Given a semi-Markovian ADMG and a single reward variable, these answer Bareinboim Task 2,
"where to intervene": which interventions could be optimal for some SCM compatible with the
graph. A structural causal bandit may restrict its arms to the POMISs without losing
optimality, pruning the exponential space of interventions.

SCOPE: a single reward variable; the input must be an ADMG whose nodes are observed variables
and whose unobserved confounders are bidirected edges (not explicit nodes). By default ALL
non-reward nodes are manipulable; pass ``manipulable`` to restrict interventions to a subset
(non-manipulable variables are then handled by latent projection — see below).

Algorithms:
- Unconstrained POMIS/MIS: Lee & Bareinboim, "Structural Causal Bandits: Where to Intervene?",
  NeurIPS 2018 (MUCT — minimal unobserved-confounder territory; IB — interventional border;
  recursive enumeration). ADAPTED, in causalrl's CausalGraph idiom, from the reference
  implementation at https://github.com/sanghack81/SCMMAB-NIPS2018 (``npsem/where_do.py``),
  MIT License, Copyright (c) 2018 Sanghack Lee.
- Non-manipulable variables: Lee & Bareinboim, "Structural Causal Bandits with Non-Manipulable
  Variables", AAAI 2019 (R-40). POMIS with non-manipulable set N equals the unconstrained
  POMIS of the latent projection of the graph onto V\\N (their Theorem 4); MIS simply filters
  to sets disjoint from N.
"""

from collections.abc import Iterable

from causalrl.identification.id_algorithm import is_identifiable_effect
from causalrl.intervention import Intervention, InterventionSpace, canonical
from causalrl.scm.graph import CausalGraph


def _cc(graph: CausalGraph, node: str) -> frozenset[str]:
    """The c-component (bidirected-connected set) containing `node` within `graph`.

    Isolated nodes form singleton components, so the fallback return is unreachable on a
    well-formed graph."""
    for component in graph.c_components():
        if node in component:
            return frozenset(component)
    return frozenset({node})


def _pa(graph: CausalGraph, nodes: frozenset[str]) -> set[str]:
    """Union of the strict parents of `nodes` within `graph`."""
    out: set[str] = set()
    for v in nodes:
        out.update(graph.parents(v))
    return out


def _muct(graph: CausalGraph, reward: str) -> frozenset[str]:
    """Minimal unobserved-confounder territory of `reward`."""
    h = graph.induced_subgraph(graph.ancestors(reward))
    queue: set[str] = {reward}
    territory: set[str] = {reward}
    while queue:
        q = queue.pop()
        ws = _cc(h, q)
        territory |= ws
        queue = (queue | h.descendants(ws)) - territory
    return frozenset(territory)


def _muct_ib(graph: CausalGraph, reward: str) -> tuple[frozenset[str], frozenset[str]]:
    """Territory and its interventional border (Pa(territory) outside the territory)."""
    territory = _muct(graph, reward)
    border = frozenset(_pa(graph, territory) - territory)
    return territory, border


def _backward_order(graph: CausalGraph) -> list[str]:
    return list(reversed(graph.topological_order()))


def _canonical(sets: set[frozenset[str]]) -> list[frozenset[str]]:
    return sorted(sets, key=lambda s: (len(s), sorted(s)))


def _sub_pomis(
    graph: CausalGraph, reward: str, ws: list[str], obs: frozenset[str]
) -> set[frozenset[str]]:
    out: set[frozenset[str]] = set()
    for i, w_i in enumerate(ws):
        territory, border = _muct_ib(graph.do_mutilate({w_i}), reward)
        new_obs = obs | frozenset(ws[:i])
        if not (border & new_obs):
            out.add(border)
            new_ws = [w for w in ws[i + 1 :] if w in territory]
            if new_ws:
                sub = graph.do_mutilate(border).induced_subgraph(territory | border)
                out |= _sub_pomis(sub, reward, new_ws, new_obs)
    return out


def pomis(
    graph: CausalGraph, reward: str, manipulable: Iterable[str] | None = None
) -> list[frozenset[str]]:
    """All POMISs for `reward`: a deduplicated, canonically sorted list of frozensets.

    ``frozenset()`` (the observational regime) appears when it is possibly optimal. When
    ``manipulable`` is given, only those variables may be intervened on: by r40's Theorem 4
    this is the unconstrained POMIS of the latent projection onto ``manipulable | {reward}``.
    """
    if manipulable is not None:
        graph = graph.latent_projection(set(manipulable) | {reward})
    graph = graph.induced_subgraph(graph.ancestors(reward))
    territory, border = _muct_ib(graph, reward)
    sub = graph.do_mutilate(border).induced_subgraph(territory | border)
    ws = [w for w in _backward_order(sub) if w in (territory - {reward})]
    result = _sub_pomis(sub, reward, ws, frozenset()) | {border}
    return _canonical(result)


def _sub_miss(
    graph: CausalGraph, reward: str, xs: frozenset[str], ws: list[str]
) -> set[frozenset[str]]:
    out: set[frozenset[str]] = {xs}
    for i, w_i in enumerate(ws):
        h = graph.do_mutilate({w_i})
        h = h.induced_subgraph(h.ancestors(reward))
        h_nodes = set(h.nodes)
        out |= _sub_miss(h, reward, xs | {w_i}, [w for w in ws[i + 1 :] if w in h_nodes])
    return out


def minimal_intervention_sets(
    graph: CausalGraph, reward: str, manipulable: Iterable[str] | None = None
) -> list[frozenset[str]]:
    """All MISs for `reward`: a deduplicated, canonically sorted list of frozensets.

    When ``manipulable`` is given, the result is filtered to sets that avoid the
    non-manipulable variables (r40: a constrained MIS is just the filtered unconstrained MIS).
    """
    graph = graph.induced_subgraph(graph.ancestors(reward))
    ws = [w for w in _backward_order(graph) if w != reward]
    result = _canonical(_sub_miss(graph, reward, frozenset(), ws))
    if manipulable is None:
        return result
    allowed = set(manipulable)
    return [s for s in result if s <= allowed]


class AdmissibleInterventions:
    """POMIS for a graph, recomputed for the variables each context actually permits.

    :func:`pomis` answers "where could it be optimal to intervene" once, at design time. A live
    agent needs the same answer per decision, because feasibility moves with the state: a lever
    can be unavailable this step and available the next. This holds the graph fixed and takes the
    manipulable set as the varying input.

    **Restricting the manipulable set is not a filter of the unconstrained POMIS.** It is the
    unconstrained POMIS of the *latent projection* onto the manipulable variables (Lee &
    Bareinboim, AAAI 2019, Theorem 4 — the r40 result :func:`pomis` already implements). Projecting
    away a variable turns paths through it into bidirected edges, which can change the territory
    and border and so *introduce* possibly-optimal sets that the unconstrained enumeration never
    listed. Dropping the sets that mention an infeasible variable would therefore lose optimality,
    not merely prune; that is why :meth:`sets` re-enters :func:`pomis` rather than filtering a
    cached list.

    Because re-entering is not free and the manipulable set usually changes rarely, results are
    memoised on it. ``cache_size`` bounds the memo; the oldest entry is evicted when it is
    exceeded, which suits the common pattern of a feasible set that oscillates among a handful of
    configurations.
    """

    def __init__(self, graph: CausalGraph, reward: str, *, cache_size: int = 32) -> None:
        if reward not in graph.nodes:
            raise ValueError(f"reward {reward!r} is not a node of the graph")
        if cache_size < 1:
            raise ValueError(
                f"cache_size={cache_size} must be at least 1: the memo holds the result for the "
                "manipulable set most recently asked about, and a zero-size cache would recompute "
                "POMIS on every call while still paying for the bookkeeping."
            )
        self._graph = graph
        self._reward = reward
        self._cache_size = cache_size
        self._memo: dict[frozenset[str], list[frozenset[str]]] = {}

    @property
    def reward(self) -> str:
        """The reward variable these intervention sets target."""
        return self._reward

    def sets(self, manipulable: Iterable[str]) -> list[frozenset[str]]:
        """The POMISs available when exactly ``manipulable`` may be intervened on.

        The reward itself is never manipulable and is ignored if present, matching :func:`pomis`.
        """
        key = frozenset(manipulable) - {self._reward}
        cached = self._memo.get(key)
        if cached is not None:
            return [frozenset(s) for s in cached]
        result = pomis(self._graph, self._reward, manipulable=key)
        if len(self._memo) >= self._cache_size:
            del self._memo[next(iter(self._memo))]  # dicts preserve insertion order: oldest first
        self._memo[key] = result
        return [frozenset(s) for s in result]

    def arms(self, space: InterventionSpace) -> list[Intervention]:
        """Every admissible intervention worth considering in ``space``, deduplicated.

        The two halves of the decision, composed: :meth:`sets` says which variables could be worth
        setting given what ``space`` permits, and
        :meth:`~causalrl.intervention.InterventionSpace.assignments` turns each of those sets into
        the concrete assignments an agent chooses between. The observational regime appears as the
        empty intervention whenever the empty set is possibly optimal.

        Order is deterministic — sets in :func:`pomis`'s canonical order (by size, then name), and
        assignments in the order the space enumerates them — so a tie broken by position is stable
        across runs rather than dependent on set iteration order.
        """
        seen: set[tuple[tuple[str, object], ...]] = set()
        out: list[Intervention] = []
        for variables in self.sets(space.variables):
            for assignment in space.assignments(variables):
                key = canonical(assignment)
                if key not in seen:
                    seen.add(key)
                    out.append(assignment)
        return out


def requires_experiment(graph: CausalGraph, treatment: Iterable[str], outcome: str) -> bool:
    """Whether learning ``P(outcome | do(treatment))`` *requires* experimentation (Task 2, "when").

    Returns ``True`` exactly when the effect is **not** identifiable from observational (L1) data,
    so an online experiment (an L2 intervention) is necessary, and ``False`` when offline data
    already suffices. This is the "when to intervene" companion to POMIS's "where": :func:`pomis`
    narrows *which* intervention sets could be optimal, while this decides *whether* you must
    intervene at all. Delegates to the complete ID algorithm
    (:func:`causalrl.identification.id_algorithm.is_identifiable_effect`).
    """
    return not is_identifiable_effect(graph, treatment, {outcome})
