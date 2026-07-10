"""Phase 1 §7.5: transport estimation + the mechanism-swap suite (acceptance g).

All numpy/local. Two environments identical but for one mechanism: swapping the treatment assignment
keeps the effect transportable (do() overrides it); swapping the outcome mechanism makes it
non-transportable (hedge); a covariate-distribution shift transports by reweighting to the target.
"""

from __future__ import annotations

import numpy as np

from causalrl.certify.certificate import Certificate, Kind
from causalrl.identification.transport import SelectionDiagram
from causalrl.scm.graph import CausalGraph
from causalrl.transport.estimate import certify_transported_effect

B0, BX, BW = 1.0, 2.0, 1.5


def _confounded_graph() -> CausalGraph:
    return CausalGraph([("W", "X"), ("W", "Y"), ("X", "Y")], [], nodes=["W", "X", "Y"])


def _gen(rng: np.random.Generator, n: int, *, a: float) -> dict[str, np.ndarray]:
    """W -> X -> Y, W -> Y; ``a`` sets how W drives X's assignment (the swappable mechanism)."""
    w = rng.binomial(1, 0.5, n).astype(float)
    x = rng.binomial(1, 1.0 / (1.0 + np.exp(-a * (w - 0.5)))).astype(float)
    y = B0 + BX * x + BW * w + 0.3 * rng.standard_normal(n)
    return {"W": w, "X": x, "Y": y}


def test_direct_transport_recovers_target_effect() -> None:
    """Swap the treatment assignment -> still transportable; source do-effect transfers."""
    rng = np.random.default_rng(0)
    source = _gen(rng, 8000, a=2.0)
    target = _gen(rng, 8000, a=-2.0)  # X mechanism differs; Y mechanism and P(W) unchanged
    diagram = SelectionDiagram(_confounded_graph(), frozenset({"X"}))
    cert = certify_transported_effect(
        diagram, source, target, treatment="X", outcome="Y", treated_value=1.0
    )
    assert cert.kind is Kind.IDENTIFIED and cert.hedge is None
    assert cert.value is not None
    truth = B0 + BX + BW * 0.5  # E*[Y | do(X=1)]
    assert abs(cert.value - truth) < 0.1
    assert cert.witness is not None and cert.witness.kind == "direct-transport"


def test_nontransportable_outcome_swap_hedges() -> None:
    """Swap the outcome mechanism -> not transportable -> hedge, never a silent point."""
    rng = np.random.default_rng(1)
    source = _gen(rng, 500, a=2.0)
    target = _gen(rng, 500, a=2.0)
    diagram = SelectionDiagram(_confounded_graph(), frozenset({"Y"}))
    cert = certify_transported_effect(diagram, source, target, treatment="X", outcome="Y")
    assert cert.value is None
    assert cert.hedge is not None and cert.hedge.reason == "non-transportable"


def test_adjustment_transport_uses_target_marginal() -> None:
    """Covariate-distribution shift transports by reweighting to the TARGET marginal, not source."""
    rng = np.random.default_rng(2)

    def gen_pop(gen: np.random.Generator, n: int, pw: float) -> dict[str, np.ndarray]:
        w = gen.binomial(1, pw, n).astype(float)
        x = gen.binomial(1, 0.5, n).astype(float)  # X exogenous
        y = B0 + BX * x + BW * w + 0.3 * gen.standard_normal(n)
        return {"W": w, "X": x, "Y": y}

    source = gen_pop(rng, 8000, 0.3)
    target = gen_pop(rng, 8000, 0.7)
    diagram = SelectionDiagram(
        CausalGraph([("W", "Y"), ("X", "Y")], [], nodes=["W", "X", "Y"]), frozenset({"W"})
    )
    cert = certify_transported_effect(
        diagram, source, target, treatment="X", outcome="Y", treated_value=1.0
    )
    assert cert.kind is Kind.IDENTIFIED and cert.value is not None
    assert abs(cert.value - (B0 + BX + BW * 0.7)) < 0.1  # target marginal
    assert abs(cert.value - (B0 + BX + BW * 0.3)) > 0.1  # NOT the source marginal
    assert cert.witness is not None and cert.witness.detail["adjustment_set"] == ["W"]


def test_transport_certificate_roundtrip() -> None:
    rng = np.random.default_rng(3)
    source = _gen(rng, 2000, a=2.0)
    target = _gen(rng, 2000, a=-2.0)
    diagram = SelectionDiagram(_confounded_graph(), frozenset({"X"}))
    cert = certify_transported_effect(diagram, source, target, treatment="X", outcome="Y")
    assert Certificate.from_json(cert.to_json()).to_dict() == cert.to_dict()
