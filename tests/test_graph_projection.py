from causalrl.scm.graph import CausalGraph


def _fig2a() -> CausalGraph:
    # Lee & Bareinboim 2019 (R-40) Fig. 2a: A->C, B->C, A->Y, C->Y; A<->B, B<->Y.
    return CausalGraph(
        directed_edges=[("A", "C"), ("B", "C"), ("A", "Y"), ("C", "Y")],
        bidirected_edges=[("A", "B"), ("B", "Y")],
    )


def test_frontdoor_projects_to_bow_arc():
    # X->Z->Y, X<->Y ; project out Z  ->  bow arc X->Y, X<->Y.
    g = CausalGraph(directed_edges=[("X", "Z"), ("Z", "Y")], bidirected_edges=[("X", "Y")])
    h = g.latent_projection({"X", "Y"})
    assert set(h.nodes) == {"X", "Y"}
    assert h.parents("Y") == ["X"]
    assert h.is_confounded("X", "Y") is True


def test_project_out_A_from_fig2a_induces_bidirected():
    h = _fig2a().latent_projection({"B", "C", "Y"})
    assert set(h.nodes) == {"B", "C", "Y"}
    assert set(h.parents("C")) == {"B"}        # A->C dropped with A
    assert set(h.parents("Y")) == {"C"}        # A->Y dropped with A
    assert h.is_confounded("B", "C") is True   # induced via removed A (B<->A->C)
    assert h.is_confounded("C", "Y") is True   # induced via removed A (C<-A->Y)
    assert h.is_confounded("B", "Y") is True    # original B<->Y preserved


def test_projection_keeping_all_is_identity_like():
    h = _fig2a().latent_projection({"A", "B", "C", "Y"})
    assert set(h.parents("C")) == {"A", "B"}
    assert set(h.parents("Y")) == {"A", "C"}
    assert h.is_confounded("A", "B") is True
    assert h.is_confounded("B", "Y") is True
    assert h.is_confounded("A", "C") is False  # never originally confounded


def test_removing_a_collider_induces_no_confounding():
    # X->M<-W (M is a collider). Removing M must NOT confound X and W.
    g = CausalGraph(
        directed_edges=[("X", "M"), ("W", "M"), ("M", "Y"), ("X", "Y"), ("W", "Y")]
    )
    h = g.latent_projection({"X", "W", "Y"})
    assert h.is_confounded("X", "W") is False
    assert set(h.parents("Y")) == {"X", "W"}
