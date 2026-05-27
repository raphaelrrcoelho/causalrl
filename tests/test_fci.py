"""FCI: causal discovery with latent confounders, and the PAG it returns.

The PAG is validated against the **true MAG** computed from the data-generating DAG-with-latents:
adjacency by m-separation over observed conditioning sets, endpoint marks by ancestry. The learned
PAG must have the MAG's adjacencies and carry no non-circle mark that contradicts the MAG
(soundness; circles are always allowed). A few hand-confident cases additionally pin exact marks.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pytest
from torch import Tensor
from torch.distributions import Bernoulli, Distribution, Uniform

from causalrl.discovery import (
    PAG,
    _discriminating_path,
    _rule1,
    _rule2,
    _rule3,
    _rule4,
    _rule5,
    _rule6,
    _rule7,
    _rule8,
    _rule9,
    _rule10,
    discover,
    discover_latent,
)
from causalrl.exceptions import CausalGraphError
from causalrl.identification._separation import d_separated
from causalrl.scm.graph import CausalGraph
from causalrl.scm.mechanisms import FunctionalMechanism, Mechanism
from causalrl.scm.scm import StructuralCausalModel

_N = 40_000


def _flip(u: Tensor, p: float) -> Tensor:
    return (u < p).float()


def _data(scm: StructuralCausalModel, observed: list[str], *, seed: int) -> dict[str, np.ndarray]:
    sample = scm.see(_N, seed=seed)
    return {v: sample[v].long().numpy() for v in observed}


def _true_mag(dag: CausalGraph, observed: list[str]) -> dict[tuple[str, str], str]:
    """The MAG over ``observed`` induced by ``dag`` (which may include latent nodes).

    Adjacent iff no observed subset m-separates them; the mark at an endpoint is a tail if that
    endpoint is an ancestor of the other, else an arrowhead.
    """
    marks: dict[tuple[str, str], str] = {}
    for a, b in combinations(sorted(observed), 2):
        others = [v for v in observed if v not in (a, b)]
        separable = any(
            d_separated(dag, {a}, {b}, set(cond))
            for k in range(len(others) + 1)
            for cond in combinations(others, k)
        )
        if separable:
            continue
        marks[(a, b)] = "-" if b in dag.ancestors(a) else ">"
        marks[(b, a)] = "-" if a in dag.ancestors(b) else ">"
    return marks


def _assert_sound(pag: PAG, mag: dict[tuple[str, str], str]) -> None:
    pag_adj = {frozenset(e) for e in pag.marks}
    mag_adj = {frozenset(e) for e in mag}
    assert pag_adj == mag_adj, f"adjacency mismatch: PAG {pag_adj} vs MAG {mag_adj}"
    for endpoint, mark in pag.marks.items():
        if mark != "o":
            assert mark == mag[endpoint], f"PAG mark {mark!r} at {endpoint} contradicts MAG {mag}"


# --- PAG data structure (M1) --------------------------------------------------------------------
def test_pag_directed_and_bidirected() -> None:
    marks = {("A", "B"): ">", ("B", "A"): "-", ("A", "C"): ">", ("C", "A"): ">"}
    pag = PAG(("A", "B", "C"), marks)
    assert pag.adjacent("A", "B") and pag.adjacent("B", "A")
    assert pag.is_directed("A", "B")
    assert not pag.is_directed("B", "A")
    assert pag.is_bidirected("A", "C") and pag.is_bidirected("C", "A")
    assert not pag.is_directed("A", "C")
    assert not pag.adjacent("B", "C")


def test_pag_edges_and_render() -> None:
    marks = {("A", "B"): ">", ("B", "A"): "-", ("B", "C"): "o", ("C", "B"): "o"}
    pag = PAG(("A", "B", "C"), marks)
    assert pag.edges() == [("A", "B", "-", ">"), ("B", "C", "o", "o")]
    rendered = pag.render()
    assert "A --> B" in rendered and "B o-o C" in rendered


def test_pag_validates_marks() -> None:
    with pytest.raises(CausalGraphError):
        PAG(("A", "B"), {("A", "B"): ">"})
    with pytest.raises(CausalGraphError):
        PAG(("A", "B"), {("A", "B"): "x", ("B", "A"): "-"})


# --- FCI core (M2): colliders + Possible-D-SEP + R1-R3, validated against the MAG ----------------
def _chain_scm() -> tuple[StructuralCausalModel, CausalGraph]:
    dag = CausalGraph(directed_edges=[("X", "Y"), ("Y", "Z")])
    mechanisms: dict[str, Mechanism] = {
        "X": FunctionalMechanism([], lambda pa, u: u),
        "Y": FunctionalMechanism(["X"], lambda pa, u: (pa["X"] + _flip(u, 0.15)) % 2),
        "Z": FunctionalMechanism(["Y"], lambda pa, u: (pa["Y"] + _flip(u, 0.15)) % 2),
    }
    exo: dict[str, Distribution] = {
        "X": Bernoulli(0.5),
        "Y": Uniform(0.0, 1.0),
        "Z": Uniform(0.0, 1.0),
    }
    return StructuralCausalModel(dag, mechanisms, exo), dag


def _collider_scm() -> tuple[StructuralCausalModel, CausalGraph]:
    dag = CausalGraph(directed_edges=[("X", "Z"), ("Y", "Z")])
    mechanisms: dict[str, Mechanism] = {
        "X": FunctionalMechanism([], lambda pa, u: u),
        "Y": FunctionalMechanism([], lambda pa, u: u),
        "Z": FunctionalMechanism(
            ["X", "Y"], lambda pa, u: ((((pa["X"] + pa["Y"]) >= 1).float()) + _flip(u, 0.05)) % 2
        ),
    }
    exo: dict[str, Distribution] = {
        "X": Bernoulli(0.5),
        "Y": Bernoulli(0.5),
        "Z": Uniform(0.0, 1.0),
    }
    return StructuralCausalModel(dag, mechanisms, exo), dag


def _latent_confounder_scm() -> tuple[StructuralCausalModel, CausalGraph]:
    # C -> A, D -> B, and a latent L -> A, L -> B confounds A and B. Two unshielded colliders
    # (C->A<-B and A->B<-D) force arrowheads at both ends of A-B, so FCI must report A <-> B.
    dag = CausalGraph(directed_edges=[("C", "A"), ("D", "B"), ("L", "A"), ("L", "B")])
    mechanisms: dict[str, Mechanism] = {
        "C": FunctionalMechanism([], lambda pa, u: u),
        "D": FunctionalMechanism([], lambda pa, u: u),
        "L": FunctionalMechanism([], lambda pa, u: u),
        "A": FunctionalMechanism(
            ["C", "L"], lambda pa, u: ((((pa["C"] + pa["L"]) >= 1).float()) + _flip(u, 0.05)) % 2
        ),
        "B": FunctionalMechanism(
            ["D", "L"], lambda pa, u: ((((pa["D"] + pa["L"]) >= 1).float()) + _flip(u, 0.05)) % 2
        ),
    }
    exo: dict[str, Distribution] = {
        "C": Bernoulli(0.5),
        "D": Bernoulli(0.5),
        "L": Bernoulli(0.5),
        "A": Uniform(0.0, 1.0),
        "B": Uniform(0.0, 1.0),
    }
    return StructuralCausalModel(dag, mechanisms, exo), dag


def test_fci_chain_is_unoriented_no_latent() -> None:
    scm, dag = _chain_scm()
    pag = discover_latent(_data(scm, ["X", "Y", "Z"], seed=0), ["X", "Y", "Z"])
    _assert_sound(pag, _true_mag(dag, ["X", "Y", "Z"]))
    assert {frozenset(e) for e in pag.marks} == {frozenset(("X", "Y")), frozenset(("Y", "Z"))}
    assert all(mark == "o" for mark in pag.marks.values())  # chain direction is undetermined


def test_fci_orients_collider_no_latent() -> None:
    scm, dag = _collider_scm()
    pag = discover_latent(_data(scm, ["X", "Y", "Z"], seed=0), ["X", "Y", "Z"])
    _assert_sound(pag, _true_mag(dag, ["X", "Y", "Z"]))
    assert pag.marks[("X", "Z")] == ">" and pag.marks[("Y", "Z")] == ">"  # arrowheads at Z
    assert not pag.is_bidirected("X", "Z")  # the X end stays a circle


def test_fci_reports_latent_confounder_as_bidirected() -> None:
    scm, dag = _latent_confounder_scm()
    observed = ["A", "B", "C", "D"]
    pag = discover_latent(_data(scm, observed, seed=0), observed)
    _assert_sound(pag, _true_mag(dag, observed))
    assert pag.is_bidirected("A", "B")  # the latent L is detected as A <-> B


def test_fci_reduces_to_pc_skeleton_without_latents() -> None:
    scm, _ = _collider_scm()
    data = _data(scm, ["X", "Y", "Z"], seed=1)
    pag = discover_latent(data, ["X", "Y", "Z"])
    cpdag = discover(data, ["X", "Y", "Z"])
    pag_adj = {frozenset(e) for e in pag.marks}
    cpdag_adj = {e for e in cpdag.undirected_edges} | {
        frozenset(e) for e in cpdag.directed_edges
    }
    assert pag_adj == cpdag_adj
    assert not any(pag.is_bidirected(*sorted(e)) for e in pag_adj)  # no spurious confounders


def _mbias_scm() -> tuple[StructuralCausalModel, CausalGraph]:
    # Classic M-bias: L1 -> X, L1 -> Z <- L2, L2 -> Y (L1, L2 latent), no X-Y edge. Z is a collider;
    # FCI must mark arrowheads at Z (the "do not condition on Z" structure).
    dag = CausalGraph(directed_edges=[("L1", "X"), ("L1", "Z"), ("L2", "Z"), ("L2", "Y")])
    mechanisms: dict[str, Mechanism] = {
        "L1": FunctionalMechanism([], lambda pa, u: u),
        "L2": FunctionalMechanism([], lambda pa, u: u),
        "X": FunctionalMechanism(["L1"], lambda pa, u: (pa["L1"] + _flip(u, 0.15)) % 2),
        "Y": FunctionalMechanism(["L2"], lambda pa, u: (pa["L2"] + _flip(u, 0.15)) % 2),
        "Z": FunctionalMechanism(
            ["L1", "L2"], lambda pa, u: (((pa["L1"] + pa["L2"]) >= 1).float() + _flip(u, 0.05)) % 2
        ),
    }
    exo: dict[str, Distribution] = {
        "L1": Bernoulli(0.5),
        "L2": Bernoulli(0.5),
        "X": Uniform(0.0, 1.0),
        "Y": Uniform(0.0, 1.0),
        "Z": Uniform(0.0, 1.0),
    }
    return StructuralCausalModel(dag, mechanisms, exo), dag


def test_fci_mbias_orients_collider_at_z() -> None:
    scm, dag = _mbias_scm()
    pag = discover_latent(_data(scm, ["X", "Y", "Z"], seed=0), ["X", "Y", "Z"])
    _assert_sound(pag, _true_mag(dag, ["X", "Y", "Z"]))
    assert pag.marks[("X", "Z")] == ">" and pag.marks[("Y", "Z")] == ">"  # collider at Z
    assert not pag.adjacent("X", "Y")  # m-separated by the empty set


# --- Per-rule unit fixtures for the orientation rules R1-R10 (minimal mark configs) -------------
def test_rule1_orients_away_from_collider() -> None:
    marks = {("A", "B"): ">", ("B", "A"): "o", ("B", "C"): "o", ("C", "B"): "o"}
    assert _rule1(marks)
    assert marks[("C", "B")] == "-" and marks[("B", "C")] == ">"  # B -> C


def test_rule2_orients_arrowhead() -> None:
    marks = {("A", "B"): ">", ("B", "A"): "-", ("B", "C"): ">", ("C", "B"): "o",
             ("A", "C"): "o", ("C", "A"): "o"}
    assert _rule2(marks)
    assert marks[("A", "C")] == ">"


def test_rule3_orients_arrowhead_at_b() -> None:
    marks = {("A", "B"): ">", ("B", "A"): "o", ("C", "B"): ">", ("B", "C"): "o",
             ("A", "D"): "o", ("D", "A"): "o", ("C", "D"): "o", ("D", "C"): "o",
             ("D", "B"): "o", ("B", "D"): "o"}
    assert _rule3(marks)
    assert marks[("D", "B")] == ">"


def test_rule4_discriminating_path_bidirects() -> None:
    marks = {("T", "A"): ">", ("A", "T"): "o", ("A", "C"): ">", ("C", "A"): "-",
             ("B", "A"): ">", ("A", "B"): "o", ("C", "B"): "o", ("B", "C"): "o"}
    assert _discriminating_path(marks, "B", "C") == ["T", "A", "B", "C"]
    assert _rule4(marks, {})  # B not in sepset(T, C): orient A <-> B <-> C
    assert marks[("A", "B")] == ">" and marks[("B", "A")] == ">"
    assert marks[("B", "C")] == ">" and marks[("C", "B")] == ">"


def test_rule5_undirects_circle_path() -> None:
    marks = {("A", "B"): "o", ("B", "A"): "o", ("A", "G"): "o", ("G", "A"): "o",
             ("G", "H"): "o", ("H", "G"): "o", ("H", "B"): "o", ("B", "H"): "o"}
    assert _rule5(marks)
    assert marks[("A", "B")] == "-" and marks[("B", "A")] == "-"
    assert marks[("A", "G")] == "-" and marks[("H", "B")] == "-"


def test_rule6_propagates_tail() -> None:
    marks = {("A", "B"): "-", ("B", "A"): "-", ("C", "B"): "o", ("B", "C"): "o"}
    assert _rule6(marks)
    assert marks[("C", "B")] == "-"


def test_rule7_propagates_tail_unshielded() -> None:
    marks = {("B", "A"): "-", ("A", "B"): "o", ("C", "B"): "o", ("B", "C"): "o"}
    assert _rule7(marks)
    assert marks[("C", "B")] == "-"


def test_rule8_orients_tail() -> None:
    marks = {("A", "B"): ">", ("B", "A"): "-", ("B", "C"): ">", ("C", "B"): "-",
             ("A", "C"): ">", ("C", "A"): "o"}
    assert _rule8(marks)
    assert marks[("C", "A")] == "-"  # A -> C


def test_rule9_orients_tail_via_pd_path() -> None:
    marks = {("A", "C"): ">", ("C", "A"): "o", ("A", "B"): "o", ("B", "A"): "o",
             ("B", "D"): "o", ("D", "B"): "o", ("D", "C"): "o", ("C", "D"): "o"}
    assert _rule9(marks)
    assert marks[("C", "A")] == "-"


def test_rule10_orients_tail_via_two_pd_paths() -> None:
    marks = {("A", "C"): ">", ("C", "A"): "o", ("B", "C"): ">", ("C", "B"): "-",
             ("D", "C"): ">", ("C", "D"): "-", ("A", "B"): "o", ("B", "A"): "o",
             ("A", "D"): "o", ("D", "A"): "o"}
    assert _rule10(marks)
    assert marks[("C", "A")] == "-"
