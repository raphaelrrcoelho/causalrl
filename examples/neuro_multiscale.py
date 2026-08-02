"""Multi-scale cortical causality end to end: spikes → functional graph → certified abstraction.

Runs the whole `causalrl.neuro` pipeline on a simulated microcircuit whose synaptic graph and
unrecorded common input are known, so every claim can be scored:

1. simulate a recurrent spiking network with latent common input;
2. recover its functional connectivity with lagged point-process discovery, and score it;
3. certify each edge against unrecorded common input;
4. ask whether the mesoscopic population model licenses micro-level interventional claims.

Run with::

    uv run python examples/neuro_multiscale.py
"""

from __future__ import annotations

from causalrl.neuro import (
    SpikingCorticalSimulator,
    certify_abstraction,
    default_interventions,
    functional_connectivity,
    two_area_microcircuit,
)


def main() -> None:
    # 1. A two-area microcircuit with one unrecorded oscillatory common input.
    spec = two_area_microcircuit(
        n_per_area=4, coupling_gain=2.5, latent_gain=1.0, n_latent=1, seed=0
    )
    recording = SpikingCorticalSimulator(spec, seed=1).simulate(120_000)  # 10 min at 5 ms bins
    print(f"recording: {recording.n_bins} bins, {recording.n_units} units, "
          f"{recording.duration / 60:.1f} min, areas={recording.areas}")
    rates = recording.firing_rates()
    print("mean rate: " + ", ".join(f"{u}={r:.1f} Hz" for u, r in list(rates.items())[:4]) + ", …")

    # 2. Functional connectivity at the micro scale, scored against the true synaptic graph.
    fc = functional_connectivity(recording, scale="micro", max_lag=2, max_conditioning_size=2)
    true_edges = set(spec.ground_truth_edges())
    found = set(fc.graph.lagged_edges())
    tp, fp = len(true_edges & found), len(found - true_edges)
    print()
    print(fc.summary())
    print(f"scored vs ground truth: {tp} true positives, {fp} false positives, "
          f"{len(true_edges - found)} missed of {len(true_edges)} real synapses")
    print(f"contemporaneous common-input candidates: {fc.graph.common_input_candidates()}")

    # 3. Per-edge sensitivity to unrecorded common input.
    print()
    for sensitivity in sorted(fc.sensitivities, key=lambda s: -s.tipping_point)[:3]:
        mark = "real" if (sensitivity.source, sensitivity.target) in true_edges else "SPURIOUS"
        print(f"[{mark}] {sensitivity.summary()}")

    # 4. Does the mesoscopic model license micro-level causal claims?
    #    A larger network, because the mesoscopic loop gain is what decides this.
    meso_spec = two_area_microcircuit(
        n_per_area=100, connection_density=0.2, latent_gain=0.6, n_latent=2,
        target_loop_gain=0.3, seed=0,
    )
    print()
    for label, interventions in (
        ("whole-area perturbations only", [
            i for i in default_interventions(meso_spec) if "half" not in i.label
        ]),
        ("including a partial-area silencing", list(default_interventions(meso_spec))),
    ):
        certificate, report = certify_abstraction(
            meso_spec, interventions=interventions, tolerance=2.0, n_bins=8000, seed=0
        )
        print(f"--- {label}")
        print(f"    {certificate}")
        print(f"    stability margin={report.stability_margin:.3g}, "
              f"max error={report.max_error:.3g} Hz, "
              f"{len(report.non_liftable)} intervention(s) with no mesoscopic counterpart")


if __name__ == "__main__":
    main()
