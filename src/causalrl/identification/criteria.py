from __future__ import annotations

from causalrl.scm.graph import CausalGraph


def backdoor_adjustment_set(graph: CausalGraph, treatment: str, outcome: str) -> set[str]:
    """Return a valid back-door adjustment set.

    Adjusting for the observed parents of `treatment` blocks all back-door paths when
    those parents are observed. Returns the parent set, or an empty set when treatment
    has no parents.
    """
    return set(graph.parents(treatment))


def is_identifiable(graph: CausalGraph, treatment: str, outcome: str) -> bool:
    """True if P(outcome | do(treatment)) is identifiable.

    Conditions checked for v0.1 (full ID/sID/gID deferred):
    - If treatment and outcome are joined by a bidirected edge AND a direct edge
      (a bow arc), the effect is NOT identifiable.
    - Otherwise it is identifiable via back-door adjustment on observed parents.
    """
    return not (graph.is_confounded(treatment, outcome) and outcome in graph.children(treatment))
