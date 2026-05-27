"""Complete transportability (sID + mz + meta) via the general ``Domain`` engine.

M1: the general engine reproduces single-source transportability exactly (a behaviour-preserving
generalization of the c-factor routing) and reports a transport-hedge when no domain can supply a
needed c-factor. At c-factor granularity invariance is exactly "touches no selection-marked
variable", so single-source observational transport was already complete.

M2 (mz): a surrogate experiment in a *source* domain supplies a c-factor that no observational
distribution can, validated against a simulation oracle. M3 (meta) follows with multiple domains.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import Tensor
from torch.distributions import Bernoulli, Distribution, Uniform

from causalrl.exceptions import CausalGraphError
from causalrl.identification.id_algorithm import (
    Domain,
    estimate_transport_general,
    identify_transport,
    identify_transport_general,
    is_identifiable_effect,
    is_transportable_general,
)
from causalrl.scm.graph import CausalGraph
from causalrl.scm.mechanisms import FunctionalMechanism, Mechanism
from causalrl.scm.scm import StructuralCausalModel

_N = 40_000


def _flip(u: Tensor, p: float) -> Tensor:
    return (u < p).float()


def _cols(samples: dict[str, Tensor], keep: list[str]) -> dict[str, np.ndarray]:
    return {name: samples[name].long().numpy() for name in keep}


# --- M1: the general engine reproduces single-source sID and reports hedges ---------------------
def test_covariate_shift_formula_mixes_domains() -> None:
    g = CausalGraph(directed_edges=[("Z", "X"), ("Z", "Y"), ("X", "Y")])
    formula = identify_transport(g, {"X"}, {"Y"}, ["Z"]).render()
    assert "do(" not in formula
    assert "P_target(" in formula and "P_source(" in formula  # shifted Z vs invariant Y


@pytest.mark.parametrize(
    ("edges", "bidirected", "ok"),
    [
        ([("Z", "X"), ("Z", "Y"), ("X", "Y")], [], True),
        ([("X", "M"), ("M", "Y")], [("X", "Y")], True),  # front-door
        ([("X", "Y")], [("X", "Y")], False),  # bow arc
    ],
)
def test_empty_selection_reduces_to_id(
    edges: list[tuple[str, str]], bidirected: list[tuple[str, str]], ok: bool
) -> None:
    g = CausalGraph(directed_edges=edges, bidirected_edges=bidirected)
    assert is_transportable_general(g, {"X"}, {"Y"}, [Domain("source")]) is ok
    assert is_identifiable_effect(g, {"X"}, {"Y"}) is ok


def test_non_transportable_hedge_is_reported() -> None:
    # Y is confounded with X (bow) AND its mechanism shifts (S->Y): neither source nor target
    # observational data supplies Q[Y]. A real transport-hedge.
    g = CausalGraph(directed_edges=[("X", "Y")], bidirected_edges=[("X", "Y")])
    assert is_transportable_general(g, {"X"}, {"Y"}, [Domain("source", frozenset({"Y"}))]) is False


def test_errors() -> None:
    g = CausalGraph(directed_edges=[("X", "Y")])
    with pytest.raises(CausalGraphError):
        is_transportable_general(g, {"X"}, {"Q"}, [Domain("source")])  # unknown outcome
    with pytest.raises(CausalGraphError):
        is_transportable_general(g, {"X"}, {"X"}, [Domain("source")])  # overlap


# --- M2: mz-transportability (a surrogate experiment in a source domain) ------------------------
def _bow_scm() -> StructuralCausalModel:
    graph = CausalGraph(directed_edges=[("U", "X"), ("U", "Y"), ("X", "Y")])
    mechanisms: dict[str, Mechanism] = {
        "U": FunctionalMechanism([], lambda pa, u: u),
        "X": FunctionalMechanism(["U"], lambda pa, u: (pa["U"] + _flip(u, 0.2)) % 2),
        "Y": FunctionalMechanism(
            ["X", "U"], lambda pa, u: ((((pa["X"] + pa["U"]) > 0).float()) + _flip(u, 0.05)) % 2
        ),
    }
    exogenous: dict[str, Distribution] = {
        "U": Bernoulli(0.5),
        "X": Uniform(0.0, 1.0),
        "Y": Uniform(0.0, 1.0),
    }
    return StructuralCausalModel(graph, mechanisms, exogenous)


def _randomized(scm: StructuralCausalModel, target: str, keep: list[str]) -> dict[str, np.ndarray]:
    low = scm.do({target: 0.0}).see(_N, seed=1)
    high = scm.do({target: 1.0}).see(_N, seed=2)
    return _cols({name: torch.cat([low[name], high[name]]) for name in low}, keep)


def test_mz_source_experiment_breaks_a_hedge() -> None:
    # Bow arc: not transportable from observation alone; a source experiment do(X) supplies Q[Y].
    g = CausalGraph(directed_edges=[("X", "Y")], bidirected_edges=[("X", "Y")])
    obs_only = Domain("source")
    with_exp = Domain("source", experiments=frozenset({frozenset({"X"})}))
    assert is_transportable_general(g, {"X"}, {"Y"}, [obs_only]) is False
    assert is_transportable_general(g, {"X"}, {"Y"}, [with_exp]) is True


def test_mz_estimand_references_the_source_experiment() -> None:
    g = CausalGraph(directed_edges=[("X", "Y")], bidirected_edges=[("X", "Y")])
    dom = Domain("source", experiments=frozenset({frozenset({"X"})}))
    formula = identify_transport_general(g, {"X"}, {"Y"}, [dom]).render()
    assert "P_source(" in formula and "do(X)" in formula


def test_mz_estimate_matches_simulation() -> None:
    scm = _bow_scm()
    g = CausalGraph(directed_edges=[("X", "Y")], bidirected_edges=[("X", "Y")])
    dom = Domain("source", experiments=frozenset({frozenset({"X"})}))
    exp_data = _randomized(scm, "X", ["X", "Y"])
    target_obs = _cols(scm.see(_N, seed=0), ["X", "Y"])
    for value in (0, 1):
        est = estimate_transport_general(
            g,
            {"X"},
            {"Y"},
            [dom],
            domain_data={"target": target_obs, "source": target_obs},
            experiment_data={("source", frozenset({"X"})): exp_data},
            do={"X": value},
        )[(1,)]
        truth = float(scm.do({"X": float(value)}).see(_N, seed=7)["Y"].float().mean())
        assert est == pytest.approx(truth, abs=0.03)


# --- M3: meta-transportability (multiple source domains) ----------------------------------------
def _two_cov_scm(p_z1: float, p_z2: float) -> StructuralCausalModel:
    """``Z1,Z2`` are independent common causes of ``X`` and ``Y``; only the marked prior changes."""
    graph = CausalGraph(
        directed_edges=[("Z1", "X"), ("Z2", "X"), ("Z1", "Y"), ("Z2", "Y"), ("X", "Y")]
    )
    mechanisms: dict[str, Mechanism] = {
        "Z1": FunctionalMechanism([], lambda pa, u: u),
        "Z2": FunctionalMechanism([], lambda pa, u: u),
        "X": FunctionalMechanism(
            ["Z1", "Z2"], lambda pa, u: (u < (0.2 + 0.3 * pa["Z1"] + 0.3 * pa["Z2"])).float()
        ),
        "Y": FunctionalMechanism(
            ["X", "Z1", "Z2"],
            lambda pa, u: ((((pa["X"] + pa["Z1"] + pa["Z2"]) >= 2).float()) + _flip(u, 0.05)) % 2,
        ),
    }
    exogenous: dict[str, Distribution] = {
        "Z1": Bernoulli(p_z1),
        "Z2": Bernoulli(p_z2),
        "X": Uniform(0.0, 1.0),
        "Y": Uniform(0.0, 1.0),
    }
    return StructuralCausalModel(graph, mechanisms, exogenous)


def test_meta_combines_invariant_factors_from_multiple_sources() -> None:
    # Source a differs in Z1, source b differs in Z2. Each invariant covariate factor is drawn from
    # the source where it is invariant, so the estimand combines BOTH source domains.
    g = CausalGraph(directed_edges=[("Z1", "X"), ("Z2", "X"), ("Z1", "Y"), ("Z2", "Y"), ("X", "Y")])
    doms = [Domain("a", frozenset({"Z1"})), Domain("b", frozenset({"Z2"}))]
    assert is_transportable_general(g, {"X"}, {"Y"}, doms) is True
    formula = identify_transport_general(g, {"X"}, {"Y"}, doms).render()
    assert "do(" not in formula
    assert "P_a(" in formula and "P_b(" in formula


def test_meta_estimate_matches_target_simulation() -> None:
    target, src_a, src_b = _two_cov_scm(0.5, 0.5), _two_cov_scm(0.85, 0.5), _two_cov_scm(0.5, 0.2)
    g = CausalGraph(directed_edges=[("Z1", "X"), ("Z2", "X"), ("Z1", "Y"), ("Z2", "Y"), ("X", "Y")])
    doms = [Domain("a", frozenset({"Z1"})), Domain("b", frozenset({"Z2"}))]
    cols = ["Z1", "Z2", "X", "Y"]
    domain_data = {
        "target": _cols(target.see(_N, seed=0), cols),
        "a": _cols(src_a.see(_N, seed=1), cols),
        "b": _cols(src_b.see(_N, seed=2), cols),
    }
    for value in (0, 1):
        est = estimate_transport_general(
            g, {"X"}, {"Y"}, doms, domain_data=domain_data, do={"X": value}
        )[(1,)]
        truth = float(target.do({"X": float(value)}).see(_N, seed=7)["Y"].float().mean())
        assert est == pytest.approx(truth, abs=0.03)


def test_meta_experiment_available_in_one_domain() -> None:
    # Bow arc: source a (mechanism shifts, no experiment) cannot supply Q[Y]; source b can run
    # do(X). The engine finds the experiment across domains.
    g = CausalGraph(directed_edges=[("X", "Y")], bidirected_edges=[("X", "Y")])
    a = Domain("a", selection=frozenset({"X"}))
    b = Domain("b", experiments=frozenset({frozenset({"X"})}))
    assert is_transportable_general(g, {"X"}, {"Y"}, [a]) is False
    assert is_transportable_general(g, {"X"}, {"Y"}, [b]) is True
    assert is_transportable_general(g, {"X"}, {"Y"}, [a, b]) is True


def test_general_transport_symbols_exported() -> None:
    import causalrl

    for name in (
        "Domain",
        "estimate_transport_general",
        "identify_transport_general",
        "is_transportable_general",
    ):
        assert hasattr(causalrl, name)
