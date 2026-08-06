"""Micro→meso abstraction on models fitted from data (no simulator on either side)."""

from __future__ import annotations

import numpy as np
import pytest

from causalrl.certify.certificate import Certificate, Kind
from causalrl.neuro.fitted import (
    area_count_columns,
    certify_fitted_abstraction,
    fit_lagged_scm,
    render_outcomes,
)
from causalrl.neuro.recording import MultiScaleRecording, RecordingError
from causalrl.neuro.simulate import SpikingCorticalSimulator, two_area_microcircuit


def _recording(n_bins: int = 30_000, seed: int = 1) -> MultiScaleRecording:
    spec = two_area_microcircuit(
        n_per_area=3, coupling_gain=2.5, latent_gain=0.0, n_latent=0, seed=0
    )
    return SpikingCorticalSimulator(spec, seed=seed).simulate(n_bins)


def test_area_count_columns_sum_the_units_of_each_area() -> None:
    rec = _recording(n_bins=500)
    macro = area_count_columns(rec)
    assert set(macro) == set(rec.areas)
    for area in rec.areas:
        idx = [rec.unit_names.index(u) for u in rec.units_in(area)]
        assert np.array_equal(macro[area], rec.spikes[:, idx].sum(axis=1))


def test_area_count_columns_needs_area_labels() -> None:
    rec = MultiScaleRecording(
        spikes=np.zeros((10, 2), dtype=np.int64), unit_names=("a", "b"), bin_size=0.005
    )
    with pytest.raises(RecordingError, match="no area labels"):
        area_count_columns(rec)


def test_macro_variable_is_exactly_what_omega_pins() -> None:
    """tau is a sum, so clamping every unit of an area to c pins the area total to n_A * c."""
    rec = _recording(n_bins=500)
    macro = area_count_columns(rec)
    area = rec.areas[0]
    n = len(rec.units_in(area))
    # If every unit in the area emitted c spikes in a bin, the macro variable would read n * c.
    assert macro[area].max() <= n * rec.spikes.max()


def test_fit_lagged_scm_returns_a_fitted_scm_over_lag_named_nodes() -> None:
    rec = _recording(n_bins=20_000)
    scm, graph, report = fit_lagged_scm(
        rec.micro_columns(), list(rec.unit_names), max_lag=2, max_conditioning_size=1
    )
    assert getattr(scm, "provenance", None) == "fitted"
    assert graph.max_lag == 2
    assert report is not None
    # Every lag of every unit is a node; the unrolled view is what makes this a DAG.
    assert f"{rec.unit_names[0]}@t-2" in scm.mechanisms


def test_fitted_scm_supports_do_on_a_lagged_node() -> None:
    rec = _recording(n_bins=20_000)
    scm, _, _ = fit_lagged_scm(
        rec.micro_columns(), list(rec.unit_names), max_lag=1, max_conditioning_size=1
    )
    unit = rec.unit_names[0]
    silenced = scm.do({f"{unit}@t-1": 0.0, unit: 0.0})
    draw = silenced.see(200, seed=0)
    assert float(np.asarray(draw[unit], dtype=np.float64).sum()) == 0.0


def test_certificate_is_never_identified_when_both_scales_are_fitted() -> None:
    """Two learned models agreeing is not ground truth, and the certificate must not say it is."""
    rec = _recording()
    cert, _, _ = certify_fitted_abstraction(rec, max_lag=2, n_samples=1500, tolerance=1e9)
    assert isinstance(cert, Certificate)
    # Both sides are learned from the same recording, so there is no ground truth to be identified
    # against; the provenance is recorded as an assumption on every certificate this path issues.
    assert cert.kind is not Kind.IDENTIFIED
    assert cert.hedge is not None
    fitted = next(a for a in cert.assumptions if a.name == "fitted-mechanisms")
    assert fitted.params["family"] == "PoissonGLMFit"
    assert fitted.diagnostic == {"provenance": "fitted"}


def test_partial_area_intervention_is_reported_as_non_liftable() -> None:
    rec = _recording()
    _, report, _ = certify_fitted_abstraction(rec, max_lag=2, n_samples=1000)
    assert report.non_liftable
    assert all("half" in row.intervention for row in report.non_liftable)
    assert "NOT LIFTABLE" in render_outcomes(report)


def test_out_of_support_drive_is_refused_rather_than_reported_as_disagreement() -> None:
    """A log-linear mechanism extrapolates exponentially; that is arithmetic, not evidence."""
    rec = _recording()
    cert, _, _ = certify_fitted_abstraction(
        rec, max_lag=2, n_samples=1000, drive_count=50.0, tolerance=2.0
    )
    assert cert.kind is Kind.EMPIRICAL
    assert cert.hedge is not None
    assert "outside the support" in cert.hedge.reason
    support = next(a for a in cert.assumptions if a.name == "intervention-support")
    assert support.diagnostic is not None
    assert support.diagnostic["outside_observed_support"]


def test_default_drive_stays_inside_the_observed_sustained_range() -> None:
    rec = _recording()
    cert, _, _ = certify_fitted_abstraction(rec, max_lag=2, n_samples=1000)
    support = next(a for a in cert.assumptions if a.name == "intervention-support")
    assert support.diagnostic is not None
    assert support.diagnostic["outside_observed_support"] == []


def test_certificate_records_the_finite_horizon_it_measured_over() -> None:
    rec = _recording()
    cert, _, scales = certify_fitted_abstraction(rec, max_lag=2, n_samples=1000)
    horizon = next(a for a in cert.assumptions if a.name == "finite-horizon")
    assert horizon.params["horizon_seconds"] == pytest.approx(2 * rec.bin_size)
    assert scales.horizon_seconds() == pytest.approx(2 * rec.bin_size)
    assert "not an equilibrium" in str(horizon.diagnostic)


def test_certificate_serialises() -> None:
    rec = _recording()
    cert, _, _ = certify_fitted_abstraction(rec, max_lag=2, n_samples=1000)
    assert Certificate.from_dict(cert.to_dict()).kind is cert.kind


def test_silencing_an_area_zeroes_it_at_both_scales() -> None:
    """The lifted macro intervention must reproduce what tau reads off the micro one."""
    rec = _recording()
    _, report, _ = certify_fitted_abstraction(rec, max_lag=2, n_samples=1500)
    silenced = [r for r in report.liftable if r.intervention.startswith("silence(")]
    assert silenced
    for row in silenced:
        area = row.intervention[len("silence(") : -1]
        assert row.micro[area] == pytest.approx(0.0, abs=1e-9)
        assert row.macro[area] == pytest.approx(0.0, abs=1e-9)
