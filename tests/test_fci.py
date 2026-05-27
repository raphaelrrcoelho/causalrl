"""FCI: causal discovery with latent confounders, and the PAG it returns.

M1 covers the PAG data structure. M2/M3 add the FCI algorithm itself, validated against the true
MAG computed from the data-generating DAG-with-latents (adjacency by m-separation, marks by
ancestry) plus hand-confident canonical PAGs.
"""

from __future__ import annotations

import pytest

from causalrl.discovery import PAG
from causalrl.exceptions import CausalGraphError


def test_pag_directed_and_bidirected() -> None:
    # A -> B (tail at A, arrow at B); A <-> C (arrowheads both ends, a latent confounder).
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
        PAG(("A", "B"), {("A", "B"): ">"})  # missing the (B, A) endpoint
    with pytest.raises(CausalGraphError):
        PAG(("A", "B"), {("A", "B"): "x", ("B", "A"): "-"})  # invalid mark
