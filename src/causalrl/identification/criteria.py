from causalrl.exceptions import NotIdentifiableError
from causalrl.scm.graph import CausalGraph


def backdoor_adjustment_set(graph: CausalGraph, treatment: str, outcome: str) -> set[str]:
    """Return the observed-parent back-door adjustment set for `treatment`.

    SCOPE / PRECONDITION: this returns ``parents(treatment)``, which is a valid parent
    adjustment set when the treatment is not incident to latent confounding. It does not
    implement front-door identification or the general ID/sID/gID algorithms. Graphs outside
    that supported contract raise rather than returning an invalid adjustment set.
    """
    parents = set(graph.parents(treatment))
    if graph.has_incident_bidirected_edges(treatment):
        raise NotIdentifiableError(
            "parent adjustment is unsupported when treatment has latent confounding; "
            "a front-door or general ID algorithm may be required"
        )
    return parents


def is_identifiable(graph: CausalGraph, treatment: str, outcome: str) -> bool | None:
    """Conservative status for whether ``P(outcome | do(treatment))`` is identifiable.

    Returns ``True`` for DAGs, ``False`` for the canonical direct bow-arc failure, and
    ``None`` for other ADMG cases whose identification status requires algorithms not yet
    implemented in this package. ``None`` deliberately avoids false positive claims.
    """
    if graph.is_confounded(treatment, outcome) and outcome in graph.children(treatment):
        return False
    if graph.has_bidirected_edges():
        return None
    return True
