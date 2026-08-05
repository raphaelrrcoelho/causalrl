# Multi-scale neural causality

`causalrl.neuro` is the library's front-end for electrophysiology: spike trains and population
signals in, certified causal claims out. It exists because cortical recordings break three
assumptions the core identification machinery makes.

| What recordings are | What the core assumes | What `neuro` adds |
|---|---|---|
| Autocorrelated time series | i.i.d. draws | `discover_lagged` — PCMCI-style lagged discovery with a contemporaneous FCI slice |
| Point processes and continuous LFP | Discrete tables, CMI tests | `PoissonGLMTest`, `PartialCorrelationTest`, `KnnCMITest` behind the `CITest` protocol |
| Two scales at once | One variable set | `certify_abstraction` — does the mesoscopic model's `do()` match the network's? |

Everything is validated against `causalrl.neuro.simulate`, a recurrent spiking network whose
synaptic graph, unrecorded common input and true interventional distributions are known by
construction — the answer key that real recordings do not come with.

!!! warning "Stability"
    Like `causalrl.experimental`, this package is **not API-frozen** and sits outside causalrl's
    semver guarantees until promoted.

## Quickstart

```python
from causalrl.neuro import (
    SpikingCorticalSimulator, functional_connectivity, two_area_microcircuit,
)

spec = two_area_microcircuit(n_per_area=4, coupling_gain=2.5, latent_gain=1.0, seed=0)
recording = SpikingCorticalSimulator(spec, seed=1).simulate(120_000)   # 10 min at 5 ms bins

fc = functional_connectivity(recording, scale="micro", max_lag=2)
print(fc.summary())
for certificate in fc.certificates[:3]:
    print(certificate)
```

Every reported edge carries a certificate saying how much unrecorded common input it would take to
explain it away. On real data you would swap the first two lines for
`causalrl.neuro.io.load_dataset(...)`.

## Lagged discovery

`discover_lagged` runs the PCMCI scheme of Runge et al. (*Sci. Adv.* 2019) over an explicit lag
embedding:

1. **PC₁ condition selection** prunes each target's candidate parents `{(source, lag)}`.
2. **MCI** re-tests each surviving link conditioning on the parents of *both* endpoints. This is
   the step that stops autocorrelation from manufacturing edges — without it, two independent but
   strongly autocorrelated spike trains look connected.
3. **Orientation.** Links at lag ≥ 1 are oriented past → present by time order — no Meek
   ambiguity, no equivalence class. The lag-0 slice goes to the shipped FCI with the lagged parents
   pinned into every conditioning set, returning a PAG.

### Reading the contemporaneous slice

At bin width `dt`, an interaction faster than one bin cannot be resolved in time. A lag-0 edge
therefore means *"common input, or an interaction faster than `dt`"* — never a resolved synaptic
effect. Three accessors keep that honest:

- `latent_pairs()` — definite `<->`; FCI committed to a latent common cause.
- `contemporaneous_ambiguous()` — `o-o` or `o->`; direction undetermined by the data.
- `common_input_candidates()` — the union, and the one to use for connectivity claims.

Narrowing `dt` moves interactions from the lag-0 slice into the lagged slice; reporting that
sensitivity alongside a result is good practice.

### Into the rest of the library

```python
graph = discover_lagged(recording.micro_columns(), list(recording.unit_names), max_lag=3)
admg = graph.unrolled_admg()      # acyclic CausalGraph over X@t-2, X@t-1, X
summary = graph.summary_graph()   # cyclic, as cortex is
```

The **unrolled** view is acyclic by construction — every edge points forward in time — so
`identify_effect`, POMIS intervention selection, transportability and the certificate layer all
apply to spike-train functional connectivity through it. The summary view keeps recurrence and is
the right object for the mesoscopic question.

## Conditional-independence tests

| Test | Use for | Basis |
|---|---|---|
| `PoissonGLMTest` | binned spike counts | likelihood-ratio between nested point-process GLMs |
| `PartialCorrelationTest` | LFP, population rates | Fisher-z on the partial correlation |
| `KnnCMITest` | nonlinear dependence, no binning | Frenzel–Pompe k-NN conditional mutual information |

All are pure NumPy — the required special functions (digamma, regularised incomplete gamma) are
implemented in `causalrl.neuro.citests`, since the project has no SciPy dependency. Any of them can
be handed to the core `discover` / `discover_latent` via `ci_test=`.

## Common-input sensitivity certificates

A functional edge is only as good as the assumption that nothing unrecorded drives both units — an
assumption a 100-electrode array falsifies by construction. Rather than assume it,
`certify_functional_edge` computes the **tipping point**: the fraction of variance an unrecorded
common input must explain *in both units* to erase the edge.

For the linear-Gaussian null `X = a·Z + e`, `Y = b·Z + e`, the least demanding confounder
reproducing a partial correlation `rho` is the symmetric one, needing `R² = |rho|` in each unit —
so the tipping point is `|rho|`, and a weaker common input cannot do it whatever its asymmetry.

That number is benchmarked against the shared variance actually observed among recorded pairs
(after Cinelli & Hazlett, *JRSS-B* 2020). An edge is reported as robust only when explaining it
away would need a hidden drive stronger than anything visible in the recording. The certificate is
always `BOUNDED` — a sensitivity statement under an explicit budget, never a point-identified
effect.

!!! warning "The benchmark is not yet calibrated for spike data"
    Experiment E2 is a negative result. The tipping point itself is exact for the linear-Gaussian
    null and verified against simulated confounding, but on binned spike trains it does **not**
    separate genuine synapses from confounding-induced ones, and the `robust` / `fragile` split is
    not currently trustworthy for point-process data.

    The problem is a units mismatch: the tipping point is a requirement on the *log-intensity*
    scale, while the benchmark measures association on the *spike-count* scale, where pairwise
    correlations are of order 1e-4 because point-process noise dominates. Almost everything clears
    such a benchmark. A point-process-calibrated benchmark — the shared drive a latent of a given
    intensity-scale strength implies — is the fix, and it is not implemented. Use
    `common_input_tipping_point` as the reportable quantity; treat the robust/fragile verdict on
    spikes as provisional. See [`docs/neuro/RESULTS.md`](neuro/RESULTS.md).

## Micro→meso abstraction

Using a population model as though intervening on it answered questions about neurons is a *causal
abstraction* claim. It holds only when the diagram commutes:

```
tau( P_micro^{do(i)} )  =  P_macro^{do(omega(i))}
```

with `tau` mapping units to area rates and `omega` mapping micro interventions to macro ones
(Rubenstein et al., *UAI* 2017; Beckers & Halpern, *UAI* 2019). `certify_abstraction` measures the
commutation error against the simulator and returns one of three verdicts:

```python
from causalrl.neuro import certify_abstraction, two_area_microcircuit

spec = two_area_microcircuit(n_per_area=100, connection_density=0.2, target_loop_gain=0.3, seed=0)
certificate, report = certify_abstraction(spec, tolerance=2.0)
print(certificate)
print(report.render())
```

- **IDENTIFIED** — commutes within tolerance, mean dynamics stable, every intervention liftable.
- **BOUNDED** — commutes only up to a measured error, or some intervention has no macro
  counterpart, or the macro model has several equilibria (an equilibrium-selection hedge: which one
  the network occupies depends on history, which no plain macro SCM represents — Blom, Bongers &
  Mooij, UAI 2019).
- **EMPIRICAL** — the mean dynamics are unstable, or commutation fails badly. The macro model is
  answering a different question than the one asked.

### Two results worth knowing

**Stability is not sufficient.** Sweeping the mesoscopic loop gain, the commutation error grows
from under 1 Hz to tens of Hz while the stability margin stays positive throughout. A model
selected by the usual criterion — is the mean-field fixed point stable? — passes settings where the
population model is wrong by 30 Hz. The error has to be measured, not inferred from the Jacobian.

**Some perturbations have no mesoscopic counterpart at all.** Silencing *part* of an area leaves
`omega` undefined, because the macro state does not resolve which units were clamped. That is the
precise sense in which a population model cannot answer a targeted optogenetic or microstimulation
question, and the certificate says so rather than substituting an area-wide intervention.

Full numbers: [`docs/neuro/RESULTS.md`](neuro/RESULTS.md).

## Loading real recordings

```python
from causalrl.neuro.io import DATASETS, load_dataset

print(DATASETS["multielectrode_grasp"].doi)   # 10.12751/g-node.f83565
recording = load_dataset("multielectrode_grasp", "/path/to/local/copy", bin_size=0.005)
```

`from_neo_block` converts any Neo `Block`: spike trains become the micro scale, analog signals are
block-averaged onto the same bin grid to become the meso scale. Neo is never imported by causalrl —
the adapters are duck-typed, so a real `neo.Block` and a stand-in with the same attributes both
work.

For NWB sorted-spike sessions (Allen, IBL, most Neuropixels pipelines) use `from_nwb_ecephys`,
which reads with `h5py` alone and needs neither pynwb nor Neo:

```python
from causalrl.neuro.io import from_nwb_ecephys

rec = from_nwb_ecephys("session.nwb", bin_size=0.005, t_start=1000.0, t_stop=1600.0,
                       areas=["VISp", "VISl", "VISal", "VISrl"], max_units_per_area=6,
                       lfp_path="probe_lfp.nwb")
```

Each unit's brain area is resolved through its `peak_channel_id` and the electrodes table's
`location`, so `unit_area` — and hence the abstraction's `tau` — comes from the recording's own
anatomy rather than being supplied by hand. Units are filtered by the published `ALLEN_QUALITY`
thresholds by default: a unit with refractory violations is partly another neuron's spikes, which
is a *measurement-induced* dependence no downstream causal machinery can undo.

`load_dataset` reads a **local** copy and never downloads. These datasets are large and versioned
by DOI; when nothing is found it raises `DatasetUnavailableError` naming the DOI and source.

## References

- Runge, Nowack, Kretschmer, Flaxman & Sejdinovic, *Detecting and quantifying causal associations
  in large nonlinear time series datasets*, Sci. Adv. 5(11), 2019 — PCMCI.
- Zhang, *On the completeness of orientation rules for causal discovery*, AIJ 172, 2008 — FCI R1–R10.
- Truccolo, Eden, Fellows, Donoghue & Brown, J. Neurophysiol. 93(2), 2005 — point-process GLMs.
- Frenzel & Pompe, Phys. Rev. Lett. 99, 2007; Kraskov, Stögbauer & Grassberger, Phys. Rev. E 69,
  2004 — k-NN (conditional) mutual information.
- Rubenstein, Weichwald, Bongers, Mooij, Janzing, Grosse-Wentrup & Schölkopf, *Causal consistency
  of structural equation models*, UAI 2017; Beckers & Halpern, *Abstracting causal models*, UAI 2019.
- Blom, Bongers & Mooij, *Beyond structural causal models: causal constraints models*, UAI 2019 —
  why equilibrium selection escapes plain SCM semantics.
- Cinelli & Hazlett, *Making sense of sensitivity*, JRSS-B 82(1), 2020 — benchmarked sensitivity.
- Brochier, Zehl, Hao, Duret, Sprenger, Denker, Grün & Riehle, *Massively parallel recordings in
  macaque motor cortex during an instructed delayed reach-to-grasp task*, Sci. Data 4:170055, 2018.
- Mazzoni, Panzeri, Logothetis & Brunel, PLoS Comput. Biol. 4(12), 2008; Einevoll, Kayser, Logothetis
  & Panzeri, Nat. Rev. Neurosci. 14, 2013 — LFP forward proxies.
