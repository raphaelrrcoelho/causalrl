"""Micro→meso causal abstraction: liftability, commutation error, and the certificate ladder."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from causalrl.certify.certificate import Certificate, Kind
from causalrl.neuro.abstraction import (
    AreaRateAbstraction,
    MicroIntervention,
    area_rates,
    certify_abstraction,
    default_interventions,
    mean_field_stability_margin,
)
from causalrl.neuro.simulate import (
    MeanFieldAreaModel,
    SpikingCorticalSimulator,
    two_area_microcircuit,
)


def _stable_spec(**kwargs: object) -> object:
    defaults: dict[str, object] = {
        "n_per_area": 60,
        "connection_density": 0.2,
        "latent_gain": 0.5,
        "n_latent": 2,
        "target_loop_gain": 0.3,
        "seed": 0,
    }
    defaults.update(kwargs)
    return two_area_microcircuit(**defaults)  # type: ignore[arg-type]


def test_stability_margin_matches_the_cyclic_scm_definition() -> None:
    """Same quantity as LinearCyclicSCM.stability_margin, without importing experimental."""
    from causalrl.experimental.cyclic import LinearCyclicSCM

    b = np.array([[0.2, 0.1], [-0.3, 0.4]])
    scm = LinearCyclicSCM(b, ["a", "b"])
    assert mean_field_stability_margin(b) == pytest.approx(scm.stability_margin())


def test_stability_margin_of_an_empty_system_is_one() -> None:
    assert mean_field_stability_margin(np.zeros((0, 0))) == 1.0


def test_tau_maps_a_recording_to_area_firing_rates() -> None:
    spec = _stable_spec(n_per_area=6, target_loop_gain=None)
    rec = SpikingCorticalSimulator(spec, seed=1).simulate(2000)
    rates = area_rates(rec)
    assert set(rates) == {"M1", "PMd"}
    assert all(0.0 < v < 300.0 for v in rates.values())


def test_omega_lifts_a_whole_area_intervention() -> None:
    spec = _stable_spec(n_per_area=5, target_loop_gain=None)
    abstraction = AreaRateAbstraction(spec)
    units = [u for u in spec.unit_names if spec.unit_area[u] == "M1"]
    macro = abstraction.omega(MicroIntervention(dict.fromkeys(units, 0.0)))
    assert macro == {"M1": 0.0}


def test_omega_refuses_a_partial_area_intervention() -> None:
    """Silencing half an area has no mesoscopic counterpart; omega must not invent one."""
    spec = _stable_spec(n_per_area=5, target_loop_gain=None)
    abstraction = AreaRateAbstraction(spec)
    units = [u for u in spec.unit_names if spec.unit_area[u] == "M1"]
    assert abstraction.omega(MicroIntervention(dict.fromkeys(units[:2], 0.0))) is None


def test_omega_refuses_heterogeneous_clamps_within_an_area() -> None:
    spec = _stable_spec(n_per_area=4, target_loop_gain=None)
    abstraction = AreaRateAbstraction(spec)
    units = [u for u in spec.unit_names if spec.unit_area[u] == "M1"]
    targets = dict.fromkeys(units, 0.0)
    targets[units[0]] = 0.5
    assert abstraction.omega(MicroIntervention(targets)) is None


def test_omega_lifts_the_observational_case_to_an_empty_macro_intervention() -> None:
    spec = _stable_spec(n_per_area=4, target_loop_gain=None)
    assert AreaRateAbstraction(spec).omega(MicroIntervention({})) == {}


def test_default_battery_includes_a_non_liftable_intervention() -> None:
    spec = _stable_spec(n_per_area=4, target_loop_gain=None)
    labels = [i.label for i in default_interventions(spec)]
    assert any("half" in label for label in labels)


# A 2 Hz tolerance is roughly 10% of this network's operating rate — the scale at which a
# mesoscopic prediction is useful for the questions a population model is asked.
TOLERANCE_HZ = 2.0


def test_stable_regime_certifies_the_abstraction_as_identified() -> None:
    spec = _stable_spec()
    liftable = [i for i in default_interventions(spec) if "half" not in i.label]
    cert, report = certify_abstraction(
        spec, interventions=liftable, tolerance=TOLERANCE_HZ, n_bins=6000, seed=0
    )
    assert isinstance(cert, Certificate)
    assert cert.kind is Kind.IDENTIFIED
    assert cert.witness is not None and cert.hedge is None
    assert report.max_error < TOLERANCE_HZ
    assert report.stability_margin > 0.0
    assert not report.non_liftable


def test_partial_area_intervention_downgrades_the_certificate_to_bounded() -> None:
    spec = _stable_spec()
    cert, report = certify_abstraction(spec, tolerance=TOLERANCE_HZ, n_bins=6000, seed=0)
    assert cert.kind is Kind.BOUNDED
    assert cert.hedge is not None
    assert "no mesoscopic counterpart" in cert.hedge.reason
    assert report.non_liftable


def test_past_the_fold_the_certificate_refuses_the_abstraction() -> None:
    """Past the saddle-node the mean field predicts a runaway the network never takes."""
    base = two_area_microcircuit(
        n_per_area=100, connection_density=0.2, latent_gain=0.6, n_latent=2, seed=0
    )
    spec = dataclasses.replace(
        base, weights=base.weights * 1.15, max_log_intensity=6.0
    )
    liftable = [i for i in default_interventions(spec) if "half" not in i.label]
    cert, report = certify_abstraction(
        spec, interventions=liftable, tolerance=TOLERANCE_HZ, n_bins=4000, seed=0
    )
    assert cert.kind is Kind.EMPIRICAL
    assert cert.hedge is not None
    assert report.max_error > 20.0
    # The point of the test: the mean-field mean dynamics are *stable* here, so a stability check
    # alone would have passed this model. Only the commutation measurement catches it.
    assert report.stability_margin > 0.0


def test_certificate_reports_the_worst_intervention_and_serialises() -> None:
    spec = _stable_spec()
    cert, report = certify_abstraction(spec, n_bins=4000, seed=0)
    assert report.worst() is not None
    assert Certificate.from_dict(cert.to_dict()).kind is cert.kind
    assert "abstraction over 2 macro variables" in report.render()


def test_report_renders_non_liftable_rows_explicitly() -> None:
    spec = _stable_spec()
    _, report = certify_abstraction(spec, n_bins=3000, seed=0)
    assert "NOT LIFTABLE" in report.render()


def test_macro_do_pins_the_area_it_intervenes_on() -> None:
    spec = _stable_spec()
    model = MeanFieldAreaModel(spec)
    assert model.equilibrium(do={"M1": 0.0})["M1"] == pytest.approx(0.0)
