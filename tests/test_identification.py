import pytest

from causalrl.exceptions import NotIdentifiableError
from causalrl.identification.criteria import backdoor_adjustment_set, is_identifiable
from causalrl.scm.graph import CausalGraph


def test_backdoor_set_for_confounded_triple():
    # Z -> X, Z -> Y, X -> Y. Adjusting for Z identifies X -> Y.
    g = CausalGraph(directed_edges=[("Z", "X"), ("Z", "Y"), ("X", "Y")])
    s = backdoor_adjustment_set(g, treatment="X", outcome="Y")
    assert s == {"Z"}


def test_identifiable_when_no_unobserved_confounding():
    g = CausalGraph(directed_edges=[("Z", "X"), ("Z", "Y"), ("X", "Y")])
    assert is_identifiable(g, "X", "Y") is True


def test_bow_arc_not_identifiable():
    # X -> Y with X <-> Y (unobserved confounder): the canonical non-identifiable bow arc.
    g = CausalGraph(directed_edges=[("X", "Y")], bidirected_edges=[("X", "Y")])
    assert is_identifiable(g, "X", "Y") is False


def test_no_backdoor_path_empty_set():
    g = CausalGraph(directed_edges=[("X", "Y")])
    assert backdoor_adjustment_set(g, "X", "Y") == set()


def test_frontdoor_has_no_backdoor_set_but_is_identifiable_by_general_id():
    # X -> M -> Y with X <-> Y has no back-door adjustment set, yet the effect IS identifiable
    # (front-door). is_identifiable now delegates to the complete ID algorithm and says so.
    g = CausalGraph(directed_edges=[("X", "M"), ("M", "Y")], bidirected_edges=[("X", "Y")])
    with pytest.raises(NotIdentifiableError, match="latent confounding"):
        backdoor_adjustment_set(g, "X", "Y")
    assert is_identifiable(g, "X", "Y") is True
