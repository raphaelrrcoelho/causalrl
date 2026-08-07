"""fit_scm_bounded: bound the confounded mechanisms instead of refusing the graph."""

from __future__ import annotations

import numpy as np
import pytest

from causalrl import CausalGraph, Kind, NotIdentifiableError, fit_scm
from causalrl.scm.bounded_fit import fit_scm_bounded


def _confounded_data(n: int = 4000, seed: int = 0) -> dict[str, np.ndarray]:
    """A latent U drives both the treatment A and the outcome Y.

    ``Y = 0.3 A + 0.5 U + eps`` with ``U ~ Bernoulli(0.5)`` and ``P(A=1 | U) = 0.2 + 0.6 U``, so
    ``E[Y | do(A=1)] = 0.3 + 0.5 * 0.5 = 0.55`` and ``E[Y | do(A=0)] = 0.25`` -- while the naive
    conditional means are pulled apart by U.
    """
    rng = np.random.default_rng(seed)
    u = rng.integers(0, 2, size=n)
    a = (rng.random(n) < 0.2 + 0.6 * u).astype(np.int_)
    y = np.clip(0.3 * a + 0.5 * u + rng.normal(scale=0.02, size=n), 0.0, 1.0)
    return {"A": a.astype(np.float64), "Y": y}


def _graph() -> CausalGraph:
    return CausalGraph(directed_edges=[("A", "Y")], bidirected_edges=[("A", "Y")], nodes=["A", "Y"])


def test_fit_scm_still_refuses_a_confounded_graph() -> None:
    """The point-fitting entry point keeps its honest refusal; only the return type differs."""
    with pytest.raises(NotIdentifiableError):
        fit_scm(_confounded_data(n=200), graph=_graph())


def test_the_bound_contains_the_truth() -> None:
    """The interval is the point of the exercise: it must cover the real do-effect."""
    fit = fit_scm_bounded(_confounded_data(), graph=_graph(), value_ranges={"Y": (0.0, 1.0)})

    treated = fit.interval("Y", {"A": 1.0})
    untreated = fit.interval("Y", {"A": 0.0})

    assert treated.lower <= 0.55 <= treated.upper
    assert untreated.lower <= 0.25 <= untreated.upper
    # A real bound, not a degenerate point masquerading as one.
    assert treated.upper - treated.lower > 0.05


def test_only_nodes_confounded_with_their_own_parents_are_bounded() -> None:
    """A bidirected edge to a non-parent does not break the regression, so it must not bound."""
    rng = np.random.default_rng(1)
    n = 500
    data = {
        "A": rng.integers(0, 2, size=n).astype(np.float64),
        "B": rng.normal(size=n),
        "Y": rng.normal(size=n),
    }
    # A <-> B, but B is not a parent of A, and Y's only parent A is unconfounded with it.
    graph = CausalGraph(
        directed_edges=[("A", "Y")], bidirected_edges=[("A", "B")], nodes=["A", "B", "Y"]
    )
    fit = fit_scm_bounded(data, graph=graph)

    assert fit.bounded_nodes == ()
    assert set(fit.identified_nodes) == {"A", "B", "Y"}
    assert fit.certificate().kind is Kind.IDENTIFIED
    # An identified node answers with a degenerate interval, so callers need not branch.
    point = fit.interval("Y", {"A": 1.0})
    assert point.lower == point.upper


def test_a_bounded_node_needs_a_value_range() -> None:
    """A Manski bound is built from the range the unobserved values occupy; there is no default."""
    with pytest.raises(ValueError, match="value_ranges is missing bounded node"):
        fit_scm_bounded(_confounded_data(n=200), graph=_graph())


def test_the_certificate_is_bounded_and_names_the_nodes() -> None:
    fit = fit_scm_bounded(_confounded_data(n=800), graph=_graph(), value_ranges={"Y": (0.0, 1.0)})
    cert = fit.certificate()

    assert cert.kind is Kind.BOUNDED
    assert "Y" in cert.claim
    assert fit.bounded_nodes == ("Y",)
    assert fit.is_identified("A") is True
    assert fit.is_identified("Y") is False
    assert "BOUNDED" in fit.summary()


def test_missing_parent_assignment_is_refused() -> None:
    fit = fit_scm_bounded(_confounded_data(n=400), graph=_graph(), value_ranges={"Y": (0.0, 1.0)})
    with pytest.raises(KeyError, match="missing parent"):
        fit.interval("Y", {})
