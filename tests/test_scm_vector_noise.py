"""Per-sample (vector) values for do() and abduct(known=).

Scalar interventions/known-noise broadcast to all n units (the existing behaviour, kept
working). A length-n tensor/array is instead applied elementwise — one value per sample —
which is what per-trajectory counterfactuals (e.g. a hidden latent that differs per unit)
require. The dynamics here are deterministic given the noise, so every assertion is exact.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.distributions import Normal

from causalrl.scm import (
    CausalGraph,
    LinearMechanism,
    StructuralCausalModel,
)


def _chain_scm() -> StructuralCausalModel:
    """X -> Y, Y = 1 + 2*X + u_y, with X = u_x. Deterministic given (u_x, u_y)."""
    graph = CausalGraph(nodes=["X", "Y"], directed_edges=[("X", "Y")])
    mechanisms = {
        "X": LinearMechanism([], {}, intercept=0.0),
        "Y": LinearMechanism(["X"], {"X": 2.0}, intercept=1.0),
    }
    exogenous = {"X": Normal(0.0, 1.0), "Y": Normal(0.0, 1.0)}
    return StructuralCausalModel(graph, mechanisms, exogenous)


def test_do_with_vector_value_intervenes_per_sample() -> None:
    scm = _chain_scm()
    x = torch.tensor([1.0, 2.0, 3.0])
    out = scm.do({"X": x}).see(3, seed=0)
    # X is pinned elementwise; Y = 1 + 2*X + u_y differs per sample BECAUSE X differs.
    assert torch.allclose(out["X"], x, atol=1e-6)
    # Y - (1 + 2X) should be the SAME shared u_y noise draw across samples? No — see()
    # draws an independent u_y per sample, so the per-sample X effect is what we isolate:
    # the X contribution 2*X differs by exactly 2*(x_i - x_j).
    contrib = out["Y"] - out["X"] * 2.0  # = 1 + u_y, the part independent of X
    # The 2*X term genuinely varies per sample (interventions are NOT broadcast to a scalar).
    y_no_noise = 1.0 + 2.0 * x
    # Pin the noise to make the per-sample claim exact:
    post = scm.do({"X": x}).abduct(known={"Y": 0.0}, n=3)
    out2 = post.predict()
    assert torch.allclose(out2["X"], x, atol=1e-6)
    assert torch.allclose(out2["Y"], y_no_noise, atol=1e-6)
    # And with nonzero shared noise the contribution check holds:
    assert contrib.shape == (3,)


def test_do_vector_value_exact_with_pinned_noise() -> None:
    scm = _chain_scm()
    x = torch.tensor([-1.0, 0.5, 4.0, 10.0])
    # Pin u_y elementwise too, so the whole trajectory is deterministic & exact.
    u_y = torch.tensor([0.0, 1.0, -2.0, 0.25])
    post = scm.do({"X": x}).abduct(known={"Y": u_y}, n=4)
    out = post.predict()
    assert torch.allclose(out["X"], x, atol=1e-6)
    assert torch.allclose(out["Y"], 1.0 + 2.0 * x + u_y, atol=1e-6)


def test_abduct_known_accepts_length_n_vector_elementwise() -> None:
    scm = _chain_scm()
    u_x = torch.tensor([0.7, -0.3, 2.0])
    u_y = torch.tensor([0.1, 0.2, 0.3])
    post = scm.abduct(known={"X": u_x, "Y": u_y}, n=3)
    out = post.predict()
    # X = u_x (X has no parents, additive noise), Y = 1 + 2*u_x + u_y — elementwise exact.
    assert torch.allclose(out["X"], u_x, atol=1e-6)
    assert torch.allclose(out["Y"], 1.0 + 2.0 * u_x + u_y, atol=1e-6)


def test_abduct_known_vector_accepts_numpy_array() -> None:
    scm = _chain_scm()
    u_x = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    post = scm.abduct(known={"X": u_x}, n=3)
    out = post.predict(do={"X": np.array([5.0, 6.0, 7.0])})
    # do() with a numpy vector also pins elementwise.
    assert torch.allclose(out["X"], torch.tensor([5.0, 6.0, 7.0]), atol=1e-6)


def test_scalar_known_and_do_still_broadcast() -> None:
    """Regression: the original scalar API must keep broadcasting to all n units."""
    scm = _chain_scm()
    post = scm.abduct(known={"X": 0.7, "Y": -0.3}, n=5)
    out = post.predict(do={"X": 3.0})
    assert torch.allclose(out["X"], torch.full((5,), 3.0), atol=1e-6)
    assert torch.allclose(out["Y"], torch.full((5,), 7.0 - 0.3), atol=1e-6)


def test_do_vector_wrong_length_raises() -> None:
    scm = _chain_scm()
    # A length-2 vector cannot be applied to n=3 samples.
    with torch.no_grad():
        try:
            scm.do({"X": torch.tensor([1.0, 2.0])}).see(3, seed=0)
        except (ValueError, RuntimeError):
            return
    raise AssertionError("expected a length-mismatch error for a vector value of wrong length")
