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
