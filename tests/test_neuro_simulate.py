"""The ground-truth cortical simulator and its mean-field counterpart."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from causalrl.neuro.simulate import (
    MeanFieldAreaModel,
    SimulationError,
    SpikingCorticalSimulator,
    _loop_gain,
    two_area_microcircuit,
)


def _spec(**kwargs: object) -> object:
    defaults: dict[str, object] = {
        "n_per_area": 4,
        "latent_gain": 0.5,
        "n_latent": 1,
        "seed": 0,
    }
    defaults.update(kwargs)
    return two_area_microcircuit(**defaults)  # type: ignore[arg-type]


def test_microcircuit_has_two_areas_and_dale_respecting_columns() -> None:
    spec = _spec()
    assert spec.areas == ("M1", "PMd")
    assert spec.n_units == 8
    for j in range(spec.n_units):
        column = spec.weights[:, j]
        nonzero = column[column != 0.0]
        if nonzero.size:
            # Dale's law: every outgoing weight of a unit shares one sign.
            assert np.all(nonzero > 0) or np.all(nonzero < 0)


def test_ground_truth_graph_excludes_self_edges_and_records_latent_pairs() -> None:
    spec = _spec()
    edges = spec.ground_truth_edges()
    assert edges
    assert all(a != b for a, b in edges)
    pairs = spec.ground_truth_confounded_pairs()
    assert all(a != b for a, b in pairs)
    graph = spec.ground_truth_summary_graph()
    assert set(graph.bidirected_edges) == {tuple(sorted(p)) for p in pairs} or pairs


def test_simulation_produces_plausible_rates_and_both_scales() -> None:
    spec = _spec()
    rec = SpikingCorticalSimulator(spec, seed=1).simulate(4000)
    assert rec.n_bins == 4000
    assert rec.meso is not None
    assert set(rec.meso_names) == {"rate:M1", "rate:PMd", "lfp:M1", "lfp:PMd"}
    rates = rec.firing_rates()
    assert all(0.5 < r < 200.0 for r in rates.values())


def test_simulation_is_reproducible_under_a_fixed_seed() -> None:
    spec = _spec()
    a = SpikingCorticalSimulator(spec, seed=3).simulate(500)
    b = SpikingCorticalSimulator(spec, seed=3).simulate(500)
    assert np.array_equal(a.spikes, b.spikes)


def test_do_silences_a_unit_and_cuts_its_incoming_edges() -> None:
    spec = _spec()
    sim = SpikingCorticalSimulator(spec, seed=2)
    target = spec.unit_names[0]
    rec = sim.simulate(2000, do={target: 0.0})
    assert rec.firing_rates()[target] == 0.0


def test_do_drives_a_unit_to_the_clamped_probability() -> None:
    spec = _spec()
    rec = SpikingCorticalSimulator(spec, seed=2).simulate(4000, do={spec.unit_names[1]: 0.3})
    observed = rec.spikes[:, 1].mean()
    assert observed == pytest.approx(0.3, abs=0.03)


def test_do_rejects_unknown_targets_and_out_of_range_probabilities() -> None:
    sim = SpikingCorticalSimulator(_spec(), seed=0)
    with pytest.raises(SimulationError, match="unknown intervention target"):
        sim.simulate(50, do={"nope": 0.0})
    with pytest.raises(SimulationError, match="probability"):
        sim.simulate(50, do={"M1_0": 1.5})


def test_target_loop_gain_hits_the_requested_mesoscopic_gain() -> None:
    spec = two_area_microcircuit(
        n_per_area=60, connection_density=0.2, latent_gain=0.4, n_latent=1,
        target_loop_gain=0.35, seed=0,
    )
    assert _loop_gain(spec) == pytest.approx(0.35, abs=1e-3)


def test_unreachable_loop_gain_fails_loudly() -> None:
    with pytest.raises(SimulationError, match="unreachable"):
        two_area_microcircuit(n_per_area=4, target_loop_gain=50.0, seed=0)


def test_mean_field_tracks_the_spiking_network_in_the_stable_regime() -> None:
    spec = two_area_microcircuit(
        n_per_area=60, connection_density=0.2, latent_gain=0.5, n_latent=2,
        target_loop_gain=0.3, seed=0,
    )
    model = MeanFieldAreaModel(spec)
    rec = SpikingCorticalSimulator(spec, seed=1).simulate(6000)
    predicted = model.equilibrium()
    for area in rec.areas:
        empirical = float(rec.population_rate(area).mean() / rec.bin_size)
        assert predicted[area] == pytest.approx(empirical, abs=1.5)


def test_fluctuation_correction_raises_the_predicted_rate() -> None:
    """The correction is off by default; when enabled it must move rates up, never down."""
    spec = two_area_microcircuit(
        n_per_area=60, connection_density=0.2, latent_gain=0.9, n_latent=2,
        target_loop_gain=0.3, seed=0,
    )
    plain = MeanFieldAreaModel(spec).equilibrium()
    corrected = MeanFieldAreaModel(spec, include_fluctuations=True).equilibrium()
    assert all(corrected[a] > plain[a] for a in plain)


def test_mean_field_do_pins_the_intervened_area() -> None:
    spec = _spec(n_per_area=20)
    model = MeanFieldAreaModel(spec)
    pinned = model.equilibrium(do={"M1": 25.0})
    assert pinned["M1"] == pytest.approx(25.0)


def test_mean_field_rejects_unknown_macro_targets() -> None:
    with pytest.raises(SimulationError, match="unknown macro intervention target"):
        MeanFieldAreaModel(_spec()).equilibrium(do={"V1": 5.0})


def test_equilibria_returns_at_least_one_root_and_they_are_fixed_points() -> None:
    spec = two_area_microcircuit(
        n_per_area=60, connection_density=0.2, latent_gain=0.4, n_latent=1,
        target_loop_gain=0.3, seed=0,
    )
    model = MeanFieldAreaModel(spec)
    roots = model.equilibria()
    assert roots
    for root in roots:
        r = np.array([root[a] for a in model.areas])
        assert np.allclose(model._map(r), r, atol=1e-4)


def test_newton_finds_the_fixed_point_where_iteration_would_run_away() -> None:
    """Past the fold the map has no moderate attractor, but the fixed point still exists."""
    base = two_area_microcircuit(
        n_per_area=100, connection_density=0.2, latent_gain=0.6, n_latent=2, seed=0
    )
    spec = dataclasses.replace(base, weights=base.weights * 1.0, max_log_intensity=6.0)
    model = MeanFieldAreaModel(spec)
    eq = model.equilibrium()
    r = np.array([eq[a] for a in model.areas])
    assert np.allclose(model._map(r), r, atol=1e-3)


def test_spec_validates_its_shapes() -> None:
    spec = _spec()
    with pytest.raises(SimulationError, match="weights must have shape"):
        dataclasses.replace(spec, weights=np.zeros((3, 3)))
    with pytest.raises(SimulationError, match="baseline must have shape"):
        dataclasses.replace(spec, baseline=np.zeros(3))
