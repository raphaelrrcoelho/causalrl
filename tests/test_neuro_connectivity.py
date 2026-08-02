"""Functional connectivity and its common-input sensitivity certificate."""

from __future__ import annotations

import numpy as np
import pytest

from causalrl.certify.certificate import Certificate, Kind
from causalrl.neuro.connectivity import (
    certify_functional_edge,
    common_input_tipping_point,
    functional_connectivity,
    observed_shared_variance,
)
from causalrl.neuro.simulate import SpikingCorticalSimulator, two_area_microcircuit
from causalrl.neuro.timeseries import lagged_frame


def test_tipping_point_equals_the_partial_correlation() -> None:
    assert common_input_tipping_point(0.4) == pytest.approx(0.4)
    assert common_input_tipping_point(-0.4) == pytest.approx(0.4)
    assert common_input_tipping_point(0.0) == 0.0
    assert common_input_tipping_point(3.0) == 1.0


def test_tipping_point_matches_the_simulated_confounding_it_describes() -> None:
    """A symmetric latent explaining a fraction R² of both units induces rho = R²."""
    rng = np.random.default_rng(0)
    n = 200_000
    for r2 in (0.1, 0.3, 0.5):
        a = np.sqrt(r2 / (1.0 - r2))
        z = rng.standard_normal(n)
        x = a * z + rng.standard_normal(n)
        y = a * z + rng.standard_normal(n)
        rho = float(np.corrcoef(x, y)[0, 1])
        assert common_input_tipping_point(rho) == pytest.approx(r2, abs=0.01)


def test_observed_shared_variance_finds_the_strongest_pair() -> None:
    rng = np.random.default_rng(1)
    n = 4000
    shared = rng.standard_normal(n)
    data = {
        "A": 2.0 * shared + rng.standard_normal(n),
        "B": 2.0 * shared + rng.standard_normal(n),
        "C": rng.standard_normal(n),
    }
    scale, pair = observed_shared_variance(data, ["A", "B", "C"])
    assert scale > 0.3
    assert pair is not None and set(pair) == {"A", "B"}


def test_observed_shared_variance_is_near_zero_for_independent_channels() -> None:
    rng = np.random.default_rng(2)
    data = {k: rng.standard_normal(4000) for k in ("A", "B", "C")}
    scale, _ = observed_shared_variance(data, ["A", "B", "C"])
    assert scale < 0.01


def _edge_frame(rho_target: float, n: int = 20000, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    y = rho_target * x + np.sqrt(1.0 - rho_target**2) * rng.standard_normal(n)
    return lagged_frame({"X": x, "Y": y}, ["X", "Y"], 1)


def test_certificate_is_bounded_and_carries_the_tipping_point() -> None:
    cert = certify_functional_edge(
        _edge_frame(0.5), source="X", target="Y", lag=0, benchmark=0.05
    )
    assert isinstance(cert, Certificate)
    assert cert.kind is Kind.BOUNDED
    assert cert.value is not None and cert.value.upper == pytest.approx(0.5, abs=0.03)
    assert cert.witness is not None and cert.witness.kind == "exceeds-observed-common-input"
    assert cert.hedge is None


def test_certificate_hedges_when_a_plausible_common_input_would_erase_the_edge() -> None:
    cert = certify_functional_edge(
        _edge_frame(0.05), source="X", target="Y", lag=0, benchmark=0.4
    )
    assert cert.kind is Kind.BOUNDED
    assert cert.witness is None
    assert cert.hedge is not None
    assert "not distinguishable from common input" in cert.hedge.reason


def test_certificate_round_trips_through_json() -> None:
    cert = certify_functional_edge(
        _edge_frame(0.5), source="X", target="Y", lag=0, benchmark=0.05, seeds=(7,)
    )
    restored = Certificate.from_dict(cert.to_dict())
    assert restored.kind is cert.kind
    assert restored.claim == cert.claim
    assert restored.provenance.seeds == (7,)


def test_certificate_records_the_conditioning_set_it_is_conditional_on() -> None:
    frame = _edge_frame(0.5)
    cert = certify_functional_edge(
        frame, source="X", target="Y", lag=1, conditioning=["Y@t-1"], benchmark=0.0
    )
    names = {a.name for a in cert.assumptions}
    assert names == {"linear-gaussian-common-input", "benchmark-shared-variance"}
    params = next(a.params for a in cert.assumptions if a.name == "linear-gaussian-common-input")
    assert params["conditioning"] == ["Y@t-1"]
    assert params["lag"] == 1


def test_functional_connectivity_certifies_every_edge_at_the_micro_scale() -> None:
    spec = two_area_microcircuit(n_per_area=3, latent_gain=0.4, n_latent=1, seed=0)
    rec = SpikingCorticalSimulator(spec, seed=1).simulate(8000)
    fc = functional_connectivity(rec, scale="micro", max_lag=2, max_conditioning_size=1)
    assert fc.scale == "micro"
    assert len(fc.certificates) == len(fc.sensitivities)
    assert all(c.kind is Kind.BOUNDED for c in fc.certificates)
    assert set(fc.robust_edges()) & set(fc.fragile_edges()) == set()
    assert "lagged edges" in fc.summary()


def test_functional_connectivity_runs_at_the_meso_scale() -> None:
    spec = two_area_microcircuit(n_per_area=4, latent_gain=0.6, n_latent=1, seed=0)
    rec = SpikingCorticalSimulator(spec, seed=1).simulate(6000)
    fc = functional_connectivity(rec, scale="meso", max_lag=2, max_conditioning_size=1)
    assert fc.scale == "meso"
    assert set(fc.graph.variables) == set(rec.meso_names)


def test_functional_connectivity_rejects_an_unknown_scale() -> None:
    spec = two_area_microcircuit(n_per_area=2, latent_gain=0.0, n_latent=0, seed=0)
    rec = SpikingCorticalSimulator(spec, seed=0).simulate(500)
    with pytest.raises(ValueError, match="micro"):
        functional_connectivity(rec, scale="nope")
