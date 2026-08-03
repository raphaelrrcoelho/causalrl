"""Phase 1 §7.1: continuous mechanisms + posterior abduction (torch; CI-verified).

torch.nn is unavailable in some dev environments, so this whole module skips unless the compiled
torch.nn is importable — it runs in CI (real torch). Acceptance (h) rests on EXACT noise recovery
for the invertible location-scale mechanism (algebraic, deterministic); the amortized-VI test only
checks the machinery runs and reconstructs (not an exactness claim).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch.nn")

from causalrl.certify.certificate import Kind  # noqa: E402
from causalrl.scm.continuous.abduction import (  # noqa: E402
    AmortizedGaussianAbduction,
    abduct_invertible,
    abduct_location_scale,
    certify_counterfactual,
    posterior_predictive_check,
)
from causalrl.scm.continuous.mechanisms import (  # noqa: E402
    ConditionalFlowMechanism,
    LocationScaleMechanism,
    MLPMechanism,
)


def test_mlp_mechanism_shapes() -> None:
    torch.manual_seed(0)
    mech = MLPMechanism(["A", "B"])
    a, b, u = torch.randn(16), torch.randn(16), torch.randn(16)
    y = mech({"A": a, "B": b}, u)
    assert tuple(y.shape) == (16,)


def test_location_scale_forward_invert_roundtrip() -> None:
    torch.manual_seed(0)
    mech = LocationScaleMechanism(["X"])
    x, u = torch.randn(64), torch.randn(64)
    y = mech({"X": x}, u)
    assert torch.allclose(mech.invert({"X": x}, y), u, atol=1e-4)


def test_exact_abduction_reproduces_known_counterfactual() -> None:
    """Acceptance (h): black-box (inversion) abduction reproduces the exact-known counterfactual."""
    torch.manual_seed(0)
    mech = LocationScaleMechanism(["X"])
    x, u_true = torch.randn(500), torch.randn(500)
    y = mech({"X": x}, u_true)  # factual
    posterior = abduct_location_scale(mech, {"X": x}, y)  # recovered without u_true
    u_hat = posterior.sample(500)["U"]
    assert torch.allclose(u_hat, u_true, atol=1e-4)  # noise recovered exactly
    x_cf = torch.randn(500)
    y_cf_known = mech({"X": x_cf}, u_true)  # exact-known CF
    y_cf_recovered = mech({"X": x_cf}, u_hat)  # CF from recovered noise
    assert torch.allclose(y_cf_recovered, y_cf_known, atol=1e-4)


def test_amortized_abduction_runs_and_reconstructs() -> None:
    torch.manual_seed(0)
    mech = MLPMechanism(["X"])
    x, u = torch.randn(400), torch.randn(400)
    y = mech({"X": x}, u)
    vi = AmortizedGaussianAbduction(mech, ["X"]).fit({"X": x}, y, steps=300, seed=0)
    posterior = vi.posterior({"X": x}, y)
    draw = posterior.sample(1, seed=0)["U"].reshape(-1)
    assert bool(torch.isfinite(draw).all())
    ppc = posterior_predictive_check(mech, {"X": x}, y, posterior.mean)
    assert ppc["ppc_rmse"] < 5.0  # loose: the VI machinery reconstructs; not an exactness claim


def test_flow_forward_invert_roundtrip() -> None:
    torch.manual_seed(0)
    mech = ConditionalFlowMechanism(["X"], n_blocks=3)
    x, u = torch.randn(64), torch.randn(64)
    y = mech({"X": x}, u)
    assert torch.allclose(mech.invert({"X": x}, y), u, atol=1e-4)


def test_flow_is_monotone_and_nonaffine() -> None:
    """The flow is strictly increasing in the noise (invertible) yet not a pure affine map."""
    torch.manual_seed(1)
    mech = ConditionalFlowMechanism([], n_blocks=3, negative_slope=0.2)
    grid = torch.linspace(-3.0, 3.0, 200)  # no parents; the zero-column root path is used
    y = mech({}, grid).detach()  # a shape assertion, not a gradient one
    diffs = y[1:] - y[:-1]
    assert bool((diffs > 0).all())  # strictly increasing => invertible
    # Second differences are non-constant => the map is genuinely non-affine.
    second = diffs[1:] - diffs[:-1]
    assert float(second.abs().max()) > 1e-3


def test_flow_exact_abduction_reproduces_counterfactual() -> None:
    """Acceptance (h) extended to the flow family: exact noise recovery => exact CF."""
    torch.manual_seed(0)
    mech = ConditionalFlowMechanism(["X"], n_blocks=2)
    x, u_true = torch.randn(500), torch.randn(500)
    y = mech({"X": x}, u_true)
    posterior = abduct_invertible(mech, {"X": x}, y)
    u_hat = posterior.sample(500)["U"]
    assert torch.allclose(u_hat, u_true, atol=1e-4)
    x_cf = torch.randn(500)
    assert torch.allclose(mech({"X": x_cf}, u_hat), mech({"X": x_cf}, u_true), atol=1e-4)


def test_abduct_location_scale_delegates_to_invertible() -> None:
    torch.manual_seed(0)
    mech = LocationScaleMechanism(["X"])
    x, u = torch.randn(32), torch.randn(32)
    y = mech({"X": x}, u)
    a = abduct_location_scale(mech, {"X": x}, y).sample(32)["U"]
    b = abduct_invertible(mech, {"X": x}, y).sample(32)["U"]
    assert torch.allclose(a, b)


def test_certify_counterfactual_kinds() -> None:
    exact = certify_counterfactual(
        "cf via inversion", {"ppc_rmse": 0.0, "ppc_bias": 0.0}, exact=True
    )
    assert exact.kind is Kind.IDENTIFIED and exact.method == "exact-inversion"
    approx = certify_counterfactual("cf via vi", {"ppc_rmse": 0.1, "ppc_bias": 0.0}, exact=False)
    assert approx.kind is Kind.EMPIRICAL and approx.method == "amortized-vi"
