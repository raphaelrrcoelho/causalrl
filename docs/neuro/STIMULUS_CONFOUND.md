# The stimulus is a confounder: ~40% of inferred connectivity during gratings is the stimulus

Functional connectivity between cortical neurons is routinely estimated while a stimulus is on
screen. A stimulus drives every responsive neuron at once, so it is a **common cause** of their
joint activity: two neurons that both like the same grating are dependent whether or not anything
connects them. Conditioning on other *recorded neurons* cannot remove it, because the driver is not
one of them.

Unlike the unrecorded common input this pipeline usually worries about, this confounder is
**exactly known** — the experimenter chose it and the NWB file records its timing, contrast and
orientation. That makes it the one case on real data where the right answer is available.

**Result: on one Allen Neuropixels session, conditioning on the stimulus removes 39% and 48% of
inferred functional connections in two independent blocks, and removes substantially the same
edges in both (p = 0.005).**

## Design

Three contiguous blocks that run back to back in session 835479236, so unit yield, spike sorting
and drift are held fixed:

```
[ gratings block 1 ] [ ----- spontaneous ----- ] [ gratings block 2 ]
      900 s                   1802 s                    900 s
```

| arm | data | stimulus in the conditioning set? |
|---|---|---|
| **A** spontaneous | no stimulus at all | — |
| **B** driven, naive | gratings on screen | **no** (standard practice) |
| **C** driven, adjusted | *the same spikes as B* | **yes** |

12 units from VISp / VISl / VISal, 20 ms bins, `max_lag=4` (an 80 ms window, which spans mouse V1's
~40–80 ms visual latency), point-process GLM tests at α=1e-3. Stimulus regressors: on/off,
contrast, orientation as `contrast·cos 2θ` and `contrast·sin 2θ`, and the within-trial drift phase
at the grating's own 2 Hz temporal frequency.

**B versus C is the whole experiment.** A and B differ in firing rate (4.36 vs 2.50 Hz), and more
spikes means more statistical power means more detected edges at fixed α — so an A/B difference has
several explanations and settles nothing. B and C are the *identical spike trains*: same rates, same
power, same units. The only thing that changes is whether the analysis knows what was on the screen.

## Results

| | edges | common-input candidates |
|---|---|---|
| A spontaneous | 50 | 5 |
| B1 driven, naive | 110 | 24 |
| **C1 driven, adjusted** | **69** | 17 |
| B2 driven, naive | 106 | 18 |
| **C2 driven, adjusted** | **56** | 16 |

| block | removed by adjustment | created | fraction of B removed |
|---|---|---|---|
| 1 | 43 | 2 | **39%** |
| 2 | 51 | 1 | **48%** |

Three things make this more than an edge count moving:

- **One-directional.** 94 edges removed against 3 created across both blocks. Adjustment deletes
  structure; it does not reshuffle it.
- **The same edges, both times.** Of 100 edges present in both naive graphs, 30 are removed in both
  blocks against 18.8 expected if the two blocks removed edges independently (one-sided binomial
  **p = 0.005**). Adjustment targets a consistent set, not a random subset.
- **It moves the graph toward the stimulus-free one.** Jaccard against the spontaneous graph rises
  0.379 → 0.417 after adjustment.

## What is not claimed

- **Anatomy of the removed edges.** They were 81% inter-areal against 76% in B overall, which fits
  the mechanism — a full-field grating is a global driver, so it should manufacture long-range
  coupling preferentially. Fisher exact gives **p = 0.37**. That is a direction, not a finding.
- **A general figure.** Units were selected for stimulus responsiveness, so this is the population
  where the effect should be *largest*. 39–48% is not an estimate for arbitrary populations.
- **That adjustment removes all of it.** Residual confounding is likely: these regressors describe
  the stimulus, not the retinal or thalamic response to it.
- **Generality.** One session, 12 units, one stimulus class, two blocks. Nothing here is
  established beyond that.

## The first version of this experiment was wrong

Worth recording, because the failure mode is the interesting part.

Run once with `max_units_per_area=4`, the result was **0 of 54 edges removed** — an apparently clean
null. It was an artefact of unit selection: `max_units_per_area` takes units in **file order**,
which selects on nothing in particular, and it returned units whose rate barely moved between
stimulus and blank (4.73 vs 5.08 Hz). The experiment asked whether adjusting for a confounder
removes edges, in a population where the confounder was not acting. Correlations with drift phase
were 0.002–0.013.

A second error compounded it: `max_lag=3` at 10 ms bins is a **30 ms** window, shorter than mouse V1
visual latency, so the stimulus response could not fit inside the analysis window even in principle.

Across all 298 quality visual units, 97 have |modulation index| > 0.2 — driven units were plentiful,
they simply were not the ones selected. Selecting on stimulus response (and only on stimulus
response, which cannot bias a B-vs-C comparison that uses the same units in both arms) and widening
the window to 80 ms produced the result above.

`causalrl.neuro.io.from_nwb_ecephys` now takes `unit_ids` for explicit selection, and its docstring
says plainly what "first N per area" selects on.

## Reproducing

```bash
uv run python experiments/neuro/run_stimulus_confound.py
```

Needs a local copy of DANDI:000022 session 835479236 under `experiments/neuro/data/`
(CC-BY-4.0; see `causalrl.neuro.io.DATASETS`).

## References

- Siegle et al., *Survey of spiking in the mouse visual system reveals functional hierarchy*,
  Nature 592, 2021 — the session and the area hierarchy.
- Runge et al., *Detecting and quantifying causal associations in large nonlinear time series
  datasets*, Sci. Adv. 5(11), 2019 — PCMCI.
- Truccolo et al., J. Neurophysiol. 93(2), 2005 — point-process GLMs for spike trains.
