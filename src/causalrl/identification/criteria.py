from causalrl.exceptions import NotIdentifiableError
from causalrl.identification.id_algorithm import is_identifiable_effect
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


def is_identifiable(graph: CausalGraph, treatment: str, outcome: str) -> bool:
    """Whether ``P(outcome | do(treatment))`` is identifiable from observational data.

    Delegates to the sound and complete ID algorithm
    (:func:`causalrl.identification.id_algorithm.is_identifiable_effect`), so it returns a definite
    boolean for any ADMG — including front-door-style cases the earlier scoped heuristic could only
    report as unknown. For the estimand itself, see
    :func:`~causalrl.identification.id_algorithm.identify_effect` and
    :func:`~causalrl.identification.id_algorithm.estimate_effect`.
    """
    return is_identifiable_effect(graph, {treatment}, {outcome})
