"""Task 2 'when to intervene': you must experiment exactly when the effect is not identifiable."""

from __future__ import annotations

from causalrl.identification.intervention_sets import requires_experiment
from causalrl.scm.graph import CausalGraph


def test_no_experiment_needed_when_observationally_identifiable() -> None:
    # Back-door: adjusting for the observed confounder Z identifies the effect offline.
    graph = CausalGraph(directed_edges=[("Z", "X"), ("Z", "Y"), ("X", "Y")])
    assert requires_experiment(graph, ["X"], "Y") is False


def test_no_experiment_needed_for_frontdoor() -> None:
    # Latent confounding, but the mediator makes the effect front-door identifiable from L1 data.
    graph = CausalGraph(directed_edges=[("X", "M"), ("M", "Y")], bidirected_edges=[("X", "Y")])
    assert requires_experiment(graph, ["X"], "Y") is False


def test_experiment_required_when_not_identifiable() -> None:
    # Bow arc: the effect is not identifiable from observation, so an experiment is necessary.
    graph = CausalGraph(directed_edges=[("X", "Y")], bidirected_edges=[("X", "Y")])
    assert requires_experiment(graph, ["X"], "Y") is True
