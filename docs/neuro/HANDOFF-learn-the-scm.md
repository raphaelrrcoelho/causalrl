# Request to the `learn-the-scm` agent, from the `claude/causalrl-multiscale-cortical-q3v9lu` branch

I've built `causalrl.neuro` (multi-scale electrophysiology: lagged causal discovery, point-process
CI tests, certified micro→meso causal abstraction). I want to merge your branch into mine, and
`fit_scm` is the piece that unblocks the most important limitation in my work. This lists what I
need from you, in priority order, with the reasoning.

## Context: why I need `fit_scm`

My `certify_abstraction` answers "does a mesoscopic population model's `do()` agree with the
spiking network's interventional behaviour?" It currently requires a `CorticalNetworkSpec` — i.e.
**the simulator** — because comparing the two scales needs a *micro* model you can intervene on.
That confines the result to simulation.

`fit_scm` is the way out: fit the micro model from recorded data, then certify the abstraction
against the fitted model. The composition already works — I test-merged and ran it:

```
discover_lagged  →  LaggedGraph.unrolled_admg()  →  fit_scm(...)  →  SCM with do()
```

On simulated spikes: 18 nodes, 24 directed edges, 0 bidirected → fitted SCM, `provenance='fitted'`,
`do()` returns a mutilated model. It works *today*. The reason it works at all is that
`unrolled_admg()` is acyclic by construction — a lag-unrolled view is the only view of a recurrent
cortical circuit that is a DAG, which is exactly what `fit_scm` needs.

So: please don't change what already works, and please add the fitter that makes it usable on real
spike trains.

---

## P0 — A point-process mechanism fitter (`PoissonGLMFit`)

**This is the ask that matters.** Binned spike counts are integer-valued with few levels (0, 1, 2
per 5 ms bin), so `_is_discrete` routes them to `TabularCPT`. Two problems:

1. **It blows up combinatorially.** `TabularCPT` builds a table of `∏(len(levels[p]) for p in
   parents)` rows. A lag-unrolled spike node routinely has 6–12 lagged parents; at ~3 levels each
   that is 3⁶ = 729 to 3¹² = 531,441 configurations, most of them never observed. Laplace smoothing
   then dominates the estimate.
2. **It throws away the point-process structure.** A spike train's conditional intensity is
   log-linear in its inputs. A CPT models it as an unstructured lookup, which is the wrong
   inductive bias and needs orders of magnitude more data.

The right mechanism is the standard point-process GLM (Truccolo et al., *J. Neurophysiol.* 93(2),
2005; Pillow et al., *Nature* 455, 2008):

```
lambda_i(t) = exp(b_i + sum_j sum_tau w_ij,tau * s_j(t - tau))
N_i(t) ~ Poisson(lambda_i(t) * dt)          # or Bernoulli(1 - exp(-lambda*dt)) at fine bins
```

Cost per node is **linear** in the number of parents, not exponential.

**You don't have to write the estimator.** I already have IRLS for exactly this model, on my
branch, in `PoissonGLMTest._fit` (`src/causalrl/neuro/citests.py`) — a Poisson log-link GLM fitted
by iteratively reweighted least squares with a ridge term for collinear regressors, pure NumPy,
tested. Please lift it (or import it) and wrap it in your `MechanismFitter` protocol. I'm happy for
it to move into `causalrl/scm/fitters.py` and for my CI test to import it from there instead —
one estimator, one home.

Specifics I need:

- `invertible=False`. The Poisson inverse-CDF coupling is one of many reproducing the same
  `P(N | parents)`, so counterfactuals at this node must be an interval, not a point — the same
  reasoning your `TabularCPT` docstring already gives. Please keep that guard.
- Route to it from `fit_scm`'s dtype heuristic when a column is **non-negative integer-valued**,
  ahead of `TabularCPT`. If you'd rather not change the default routing, that's fine — but then
  please make sure passing `families={node: PoissonGLMFit()}` works for every node, which is all I
  strictly need.
- Report an **offset/exposure** term if it's cheap (`log(dt)`), so bin width is explicit rather
  than absorbed into the intercept. Nice-to-have, not a blocker.

## P0 — Don't break `fit_scm(data, graph=...)`

I depend on this exact call, with:

- a plain `Mapping[str, np.ndarray]` of columns (I pass a lag-embedded frame from
  `causalrl.neuro.timeseries.lagged_frame`, so node names look like `M1_0@t-2`);
- a `CausalGraph` positional-or-keyword `graph=`;
- the returned object being an ordinary `StructuralCausalModel` that `do()` accepts.

Node names containing `@` and `-` must keep working — they're how I encode lags.

## P1 — Poisson-appropriate holdout scoring

`evaluate_holdout` uses mean log-likelihood for discrete/non-invertible and R² for
continuous/invertible. For a Poisson node please report **mean log-likelihood** (or mean deviance,
stated either way), not R². R² on count data with a log link is misleading, and I want to put this
number in a certificate's `Assumption.diagnostic`, so it needs to mean what it says.

## P1 — Keep torch off the required path (or tell me it's staying)

`scm/fit.py` and `scm/fitters.py` both `import torch` at module level (`fit.py:15`,
`fitters.py:14`). `causalrl.neuro` is currently **pure NumPy** and works on the core install; the
project ships torch only under the `[torch]` extra, and the README is explicit that "the core
graph, POMIS, tabular-agent and tabular-environment surfaces do not require PyTorch".

If I take a dependency on `fit_scm`, everything downstream of `discover_lagged` inherits torch.
Preferred, in order:

1. Make the torch import lazy so `TabularCPT` / `LinearGaussianFit` / `ANMFit` / `PoissonGLMFit`
   work without it, and only `NeuralFit` requires it.
2. Or leave it, and I'll gate my fitted-SCM path behind the `[torch]` extra and document it.

Either is workable — I just need to know which, so I document the right thing. Option 1 is
noticeably better for a spike-train user who wants a GLM and nothing else.

## P2 — The confounded case is where real recordings live

`fit_scm` refuses bidirected edges, and the message is good:

> `fit_scm cannot fit a graph with bidirected edges: under latent confounding a mechanism is not
> identified by regression on observed parents.`

That refusal is correct and I don't want it weakened. But it is also the **common case for my
data**: a Utah array samples a vanishing fraction of the local network, so unrecorded common input
is the default, and my contemporaneous FCI slice produces `<->` edges precisely to record it. Right
now the composition only runs when that slice happens to be empty.

I'm not asking you to solve confounded fitting. I'm asking for one of:

- a documented, sound path for the partially-confounded case — e.g. fit the sub-DAG over nodes not
  incident to any bidirected edge and mark the rest unfitted in the `FitReport`; or
- confirmation that the neural-causal-model construction your error message mentions is planned,
  so I can write my docs against it rather than around it.

Please don't "solve" it by dropping bidirected edges silently. An unsound fit is worse than a
refusal here — the whole point of my certificate layer is that common input is the thing you must
not assume away.

## P2 — `fit_scm_mec` over PAGs, not just CPDAGs

`fit_scm_mec` enumerates the MEC of a **CPDAG**. My contemporaneous slice is a **PAG** (FCI output,
because latents are possible), where `o-o` means "could be `->`, `<-`, or `<->`". A PAG-aware
version — enumerating the MAGs in the equivalence class — would let me turn my
`common_input_candidates()` into an actual set-valued belief instead of a list of caveats.

Lower priority than the fitter, but it is the natural pairing with my `contemporaneous_ambiguous()`
accessor, and it fits your existing "fit every member rather than pick one" design.

## P3 — Merge mechanics (no action needed from you)

We both touched `src/causalrl/discovery.py`. I test-merged: **one conflict, and it is only the
`__all__` list** — your `"orient"` vs my `"CITest"`. Union of the two; everything else auto-merged
cleanly, and `test_neuro_timeseries`, `test_scm_fit`, `test_orient` and `test_scm_fitters` all pass
together (55 tests) once resolved. I'll do the resolution on merge.

One thing to be aware of: I added a `CITest` protocol and an optional `ci_test=` parameter to
`discover`, `discover_latent` and `discover_interventional`, so PC/FCI can run on continuous and
point-process data. Default behaviour is unchanged. Your `orient` doesn't touch that path. If you
restructure `discovery.py`'s public surface further, a heads-up would help.

Also note `orient(cpdag, tiers=...)` and my lagged orientation are complementary, not redundant:
lagged links are oriented past→present by construction, and the lag-0 slice goes to FCI as a PAG,
which `orient` doesn't consume. Nothing to deduplicate.

---

## What "done" looks like for me

```python
from causalrl.neuro import discover_lagged, PoissonGLMTest
from causalrl.neuro.timeseries import lagged_frame
from causalrl.scm.fit import fit_scm
from causalrl.scm.fitters import PoissonGLMFit

graph = discover_lagged(rec.micro_columns(), units, max_lag=3, ci_test=PoissonGLMTest())
frame = lagged_frame(rec.micro_columns(), units, 3)
scm = fit_scm(frame, graph=graph.unrolled_admg(),
              families={n: PoissonGLMFit() for n in frame})
# -> a micro model fitted from real spikes that I can do() and hand to certify_abstraction
```

Everything except `PoissonGLMFit` already works.
