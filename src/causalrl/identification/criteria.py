from __future__ import annotations

from causalrl.scm.graph import CausalGraph


def backdoor_adjustment_set(graph: CausalGraph, treatment: str, outcome: str) -> set[str]:
    """Return the observed-parent back-door adjustment set for `treatment`.

    SCOPE / PRECONDITION (v0.1): this returns ``parents(treatment)``, which is a *valid*
    back-door set ONLY when the treatment's parents are all observed and there is no
    latent confounding of the treatment. It does NOT search for minimal sets, does NOT
    verify that the returned set blocks every back-door path, and does NOT handle
    front-door identification. For example, on the front-door graph ``X->M->Y, X<->Y`` it
    returns ``set()`` (the correct estimand there is the front-door formula, not back-door
    adjustment). Do not feed the result into adjustment without confirming the
    precondition holds. Full ID/sID/gID is deferred to a later version.
    """
    return set(graph.parents(treatment))


def is_identifiable(graph: CausalGraph, treatment: str, outcome: str) -> bool:
    """Whether P(outcome | do(treatment)) is identifiable.

    SCOPE (v0.1): this detects only the canonical *bow arc* non-identifiable structure
    (treatment and outcome joined by both a direct edge and a bidirected edge) and returns
    ``False`` for it. It returns ``True`` for everything else, which means it can be
    OPTIMISTIC: graphs that are non-identifiable for reasons other than a direct bow arc
    (e.g. a hedge over a longer path) are not yet detected. Full ID/sID/gID is deferred.
    """
    return not (graph.is_confounded(treatment, outcome) and outcome in graph.children(treatment))
