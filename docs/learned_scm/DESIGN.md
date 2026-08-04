# `fit_scm` — learning the SCM from data

- **Date:** 2026-07-31
- **Status:** Approved design (brainstorming). Next step: implementation plan (writing-plans).
- **Branch:** `learn-the-scm`
- **Program position:** sub-project 1 of 4. Sequence: **1 → 4 → 3 → 2**.
- **Parent program:** `docs/causal_mbrl_agent/DESIGN.md` — this builds the SCM-belief half that
  program specified but never implemented.

## 1. Motivation

`causalrl` can *specify* SCMs and *certify* decisions, but it has no path from data to an SCM.
Every `StructuralCausalModel` in the library is hand-written by the user. `discover` returns a
CPDAG that `DiscoveryBackdoorAgent` mines for a back-door adjustment set and then discards.
`NeuralMechanism` (`src/causalrl/scm/mechanisms.py:60`) is documented as making the SCM a neural
causal model and is trained nowhere — it appears in one unit test.

`docs/causal_mbrl_agent/DESIGN.md:74` promised an SCM belief of "graph + mechanisms + confounding".
What shipped is the graph → adjustment-set half. The mechanism-learning half was never built, which
is why the agent's only distinctive output is a robustness band: with no learned model, a
sensitivity certificate is the *only* thing it can add to a point estimate.

This sub-project builds the missing primitive. Its consumers are the remaining three sub-projects:
planning inside the learned model (4), active interventional discovery that supplies and refines
the graph (3), and the neural-causal-model min/max that turns a fit into a tight bound (2).

### The design claim this rests on

L1 data plus a true, causally-sufficient DAG identifies the L2 mechanisms but **not L3**. The
coupling from exogenous noise to a discrete variable's value is a modelling choice: many couplings
reproduce the same observational *and* interventional distributions while disagreeing on
counterfactuals. A fitted SCM must therefore not report point counterfactuals. The honest answer is
an interval over admissible couplings.

That is the thread the whole program hangs on: bounds stop being a bolt-on module and become a
*consequence of having learned a model*.

## 2. Goals / non-goals

**Goals**

- `fit_scm(data, graph) → StructuralCausalModel` — a learned model that is rollout-able,
  interveneable, and honest about what it cannot answer.
- Learn into the **existing** `StructuralCausalModel` type, so `do`, `abduct`, `see`,
  `CausalEnvWrapper`, transport, `build_unrolled_scm`, and the certify layer all accept learned
  models with zero new code.
- Four mechanism families, auto-selected by dtype and per-node overridable.
- A CPDAG → DAG orientation seam, so `discover → orient → fit_scm` is a documented pipeline.
- Counterfactual queries on a learned SCM return a **valid interval**, labelled tight or not.
- A falsifiable oracle gate, plus a real-data cross-check.

**Non-goals (phase 1)**

- Fitting under latent confounding (ADMG / bidirected edges). Needs the NCM one-latent-per-
  c-component construction — sub-project 2.
- Tight multi-node counterfactual bounds — sub-project 2's constrained min/max.
- Causal representation learning from raw perception. `DESIGN.md` §2 already defers this.
- Beating targeted estimators on a single estimand. See §7.

## 3. Public API

Eleven new names, exported top-level alongside `discover`: four entry points (`orient`, `fit_scm`,
`fit_scm_mec`, `counterfactual_interval`), the `CounterfactualBound` return type, the `FitReport` /
`NodeFit` provenance records, and the four mechanism fitters (`TabularCPT`, `LinearGaussianFit`,
`ANMFit`, `NeuralFit`). The snippet below shows the entry points.

```python
from causalrl import discover, orient, fit_scm, fit_scm_mec, counterfactual_interval

cpdag = discover(data, variables=["Z", "A", "Y"])
dag   = orient(cpdag, tiers=[["Z"], ["A"], ["Y"]])
scm   = fit_scm(data, graph=dag)

scm.do({"A": 1}).see(1000)        # L2, for free
scm.provenance                     # "fitted"
scm.fit_report                     # per-node family, holdout score, invertible

counterfactual_interval(scm, evidence={"A": 0.0, "Y": 0.0},
                        interventions={"A": 1}, target="Y")
# CounterfactualBound(lower=..., upper=..., tight=True)
```

### Signatures

```python
def orient(cpdag: CPDAG, *, tiers: Sequence[Sequence[str]] | None = None) -> CausalGraph: ...

def fit_scm(
    data: Mapping[str, np.ndarray],
    *,
    graph: CausalGraph,
    families: Mapping[str, MechanismFitter] | None = None,
    holdout: float = 0.2,
    seed: int = 0,
) -> StructuralCausalModel: ...

def fit_scm_mec(
    data: Mapping[str, np.ndarray],
    *,
    cpdag: CPDAG,
    max_members: int = 32,
    **kwargs: Any,
) -> list[StructuralCausalModel]: ...

def counterfactual_interval(
    scm: StructuralCausalModel,
    *,
    evidence: Mapping[str, float],
    interventions: Mapping[str, Value],
    target: str,
    n: int = 20_000,
    seed: int | None = None,
) -> CounterfactualBound: ...
```

`fit_scm_mec` is a separate function rather than a `mec=True` flag on `fit_scm`, so the return type
does not depend on an argument value. Above `max_members` it raises `ValueError` naming the actual
equivalence-class size — Markov-equivalence classes are exponential and silent truncation would
misrepresent the belief. That returned list *is* the SCM belief sub-project 3 shrinks by
intervening.

## 4. Architecture

| Unit | Interface | File | Depends on |
|---|---|---|---|
| Orientation | `CPDAG (+ tiers) → CausalGraph` | `discovery.py` | torch-free |
| Mechanism fitters | `(parent columns, child column) → (Mechanism, noise dist, invertible)` | `scm/fit.py` | numpy / torch |
| SCM fitting | `data + DAG → StructuralCausalModel` | `scm/fit.py` | fitters |
| Counterfactual bound | `(SCM, evidence, do, target) → CounterfactualBound` | `identification/counterfactual_bounds.py` | fitted conditionals + `residual` |

`orient` lives in `discovery.py` because it is a structure operation on a CPDAG and must stay
torch-free — `discovery.py` has no torch import and the top-level lazy loader relies on that.
`fit_scm` lives under `scm/` and therefore inherits the optional `[torch]` extra gate
(`src/causalrl/__init__.py:542`), which is already the rule for the whole SCM surface.

### Provenance

`StructuralCausalModel` gains two read-only attributes:

- `provenance: Literal["specified", "fitted"]`, defaulting to `"specified"` so every existing
  construction site is unchanged.
- `fit_report: FitReport | None` — `None` for specified models.

```python
class NodeFit(NamedTuple):
    node: str
    family: str            # "tabular_cpt" | "linear_gaussian" | "anm" | "neural"
    parents: tuple[str, ...]
    holdout_score: float   # log-likelihood (discrete) or R^2 (continuous)
    invertible: bool

class FitReport(NamedTuple):
    nodes: tuple[NodeFit, ...]
    n_samples: int
    def summary(self) -> str: ...
```

`invertible` is the per-node flag that decides whether a node contributes counterfactual ambiguity.
It is a property of the fitted mechanism, not of the family alone (a `NeuralFit` built with an
additive-noise head is invertible; a general net is not).

### Orientation rules

`orient` resolves undirected edges in this order, and raises `CausalGraphError` if any edge remains:

1. **Tiers** — an edge between different tiers is oriented earlier-tier → later-tier. This reuses
   the temporal-tier assumption `DiscoveryBackdoorAgent` already relies on.
2. **Acyclicity** — an orientation that would close a cycle takes the only remaining direction.
3. **Unresolved** — raise, listing the edges, and name `fit_scm_mec` in the message.

Rule 3 is deliberate: silently picking an orientation would commit to an unidentified choice, which
is the failure mode the library rejects everywhere else.

## 5. Mechanism families

Selected per node by dtype, overridable via `families={"Y": ANMFit(...)}`.

```python
class MechanismFitter(Protocol):
    def fit(self, parents: dict[str, np.ndarray], child: np.ndarray
            ) -> tuple[Mechanism, Distribution, bool]: ...
```

| Family | Node type | Fit | Noise | `invertible` |
|---|---|---|---|---|
| `TabularCPT` | discrete | conditional counts, Laplace-smoothed | `Uniform(0,1)` + inverse CDF | `False` |
| `LinearGaussianFit` | continuous | closed-form OLS | `Normal(0, σ̂)` from residuals | `True` |
| `ANMFit` | continuous | duck-typed `fit`/`predict` estimator, numpy-ridge default | empirical residual distribution | `True` |
| `NeuralFit` | continuous | torch MLP into `NeuralMechanism` | `Normal(0, σ̂)`, additive head | `True` |
| `PoissonGLMFit` | count | log-link GLM via IRLS, opt-in | `Uniform(0,1)` + inverse CDF | `False` |
| `BayesianLinearFit` | continuous | NUTS posterior over `(intercept, w, σ)`, opt-in, `[numpyro]` extra | `Normal(0, E[σ])` from the posterior mean | `True` |

`ANMFit` mirrors `GFormulaBackdoorAgent`'s `outcome_model=` factory (`agents/mbrl.py:409`): a
callable returning a fresh sklearn-style estimator, so sklearn stays optional.

**Default selection:** integer dtype with ≤ 20 distinct values → `TabularCPT`; otherwise `ANMFit`.
`LinearGaussianFit` and `NeuralFit` are opt-in per node. Root nodes (no parents) fit their marginal
and take noise directly, matching the `FunctionalMechanism([], lambda pa, u: u)` idiom already used
across `envs/suite/`.

**Torch note:** the local toolchain works (torch 2.12.0, `pyright src` clean, checked 2026-08-01),
so `NeuralFit` is verified locally *and* in CI. Its test stays a smoke test on a small fixed
problem, not a convergence benchmark, so it remains fast enough for CI.

**Guard:** `fit_scm` raises `NotIdentifiableError` when `graph.has_bidirected_edges()`, naming
sub-project 2 as the path for confounded fitting. Regression on parents is not a valid mechanism
estimate under latent confounding, and failing loudly is the only honest option.

## 6. Counterfactual intervals

### Why an interval

For a discrete node the fitted object is `P(V | Pa)`. Turning it into a structural equation
`V = f(Pa, U)` requires choosing a coupling, and the choice is invisible to L1 and L2 data. Two
SCMs with identical `P(V|Pa)` for every `Pa` disagree on `P(V_{a=1} = 1, V_{a=0} = 0)`.

### What makes it computable

A counterfactual query touches only **finitely many parent configurations** — the factual one plus
one per `do` contrast, typically two. So the ambiguity at each discrete node is a coupling between
two known conditionals, whose extremes are the closed-form Fréchet–Hoeffding bounds:

```
P(V_{pa₁} = v, V_{pa₀} = v) ∈ [max(0, p₁ + p₀ − 1), min(p₁, p₀)]
```

No LP, no scipy dependency. Invertible nodes contribute zero width: their noise is recoverable from
`(parents, value)`, so their counterfactual is a point. On the canonical binary cases this
reproduces the known Tian–Pearl probability-of-necessity / probability-of-sufficiency bounds, which
is the test oracle in §8.

### Tightness — stated, not hidden

When several discrete nodes contribute ambiguity, composing per-node Fréchet intervals is **valid
but not necessarily tight**: the extremes may not be jointly achievable by a single SCM. The return
type carries this:

```python
class CounterfactualBound(NamedTuple):
    lower: float
    upper: float
    tight: bool            # the slot for sub-project 2's valid-but-loose bounds
    @property
    def interval(self) -> Interval: ...
```

**Refined during planning:** rather than *composing* a loose bound when a second non-invertible node
contributes, phase 1 **refuses** — it raises and names the NCM min/max. A non-invertible node the
intervention cannot reach costs nothing (it keeps its factual value), so the refusal fires only when
one sits strictly between the intervention and the target. `tight` therefore reads `True` throughout
phase 1; the field exists so sub-project 2 can return valid-but-loose bounds without a breaking type
change. An honest refusal beats a loose answer, and both beat a tight-looking one that is wrong.

**`CounterfactualBound` is a new type, not an extension of `Interval`.** `Interval`
(`identification/bounds.py:17`) documents tuple-compatibility — `lo, hi = interval` — as a public
guarantee, and `causalrl` is released at 2.1.0. Adding a third field would break that unpacking for
every existing caller. `.interval` converts for interop with the existing bounds surface.

### Guarding the existing L3 paths

`StructuralCausalModel.abduct` (`scm/scm.py:153`) is the single choke point for every L3 entry
point — `counterfactual`, `counterfactual_expectation`, `effect_of_treatment_on_treated`,
`regret_decision_table` all route through it. It raises `NotIdentifiableError` when
`provenance == "fitted"` **and the model contains any non-invertible node**, with a message naming
`counterfactual_interval`.

The check is whole-model rather than scoped to a query's ancestry, and it has to be: `abduct`
returns an `ExogenousPosterior` that the caller may reuse for *any* subsequent `predict(do=...)`,
so no target exists at abduct time to compute an ancestry from. The guard therefore over-refuses —
a fitted SCM with one discrete covariate refuses abduction even for a wholly continuous query — and
that is the safe direction: it never lets a coupling choice through unlabelled, and
`counterfactual_interval` is the precise path. Ancestry scoping does apply there, where the target
*is* known (§6).

Because the guard keys off provenance, hand-written SCMs are completely unaffected: the user
asserted those mechanisms, so their couplings are given, not inferred.

The guard needs an unguarded twin. `abduct` becomes a thin wrapper over a private `_abduct`, so a
caller that has already established what its query licenses can bypass it — sub-project 4's
counterfactual data augmentation replays invertible mechanisms on a fitted model and must not be
blocked.

`counterfactual_interval` does not use either: rejection-sampled abduction never matches continuous
evidence, so invertibility is made operational instead. Every invertible fitter attaches a
`residual(parent_values, value)` inverse map, and the bound recovers noise exactly by solving
`U = V − g(parents)`. That is what `invertible=True` buys.

## 7. Gates

### Oracle gate (kill gate, runs in CI)

Fit on observational samples from a known `envs/suite/` SCM — `build_discovery_scm()` is the
primary world: `X → Z ← Y`, `Z → W`, all binary, noisy-OR and noisy-copy mechanisms, so every
`do`-query has an exact oracle value.

The baseline is the sharp one. A **correlational world model** is the same `fit_scm` machinery run
on an L1-equivalent but causally wrong DAG (a reversed ordering / a fully connected alternative in
the same Markov equivalence class). It matches the observational distribution just as well and
implies different interventions. This isolates the claim: **structure, not fit quality, is what
buys L2 correctness.**

- **PASS:** mean absolute error of the true-graph fit on held-out `do`-queries → ≈ 0 (within Monte-
  Carlo tolerance) at n = 20k, while the L1-matched wrong-structure model's error stays materially
  higher, with a per-seed gap whose 95% CI over 10 seeds excludes 0.
- **FAIL / kill:** the wrong-structure model's `do`-query error is statistically indistinguishable
  from the true-graph fit's. That would mean the graph carries no information the fit doesn't
  already have in this regime, and the sub-project's premise is wrong.

This finally runs the model half of `DESIGN.md`'s Verdict 1, which was replaced by skeleton
discovery in M1a and never executed.

### Real-data cross-check (credibility, runnable example)

Fit on NHEFS and Coat; check `scm.do()` **agrees within CI** with the g-formula / AIPW numbers the
existing real-data suite already produced (`docs/causal_mbrl_agent/REAL_DATA.md`).

The criterion is agreement, not superiority, and that is deliberate. The suite's own meta-lesson is
that a general causal model does not reliably beat a targeted estimator on a single estimand. A
gate demanding superiority would be a gate designed to be failed. Agreement is the win; the surplus
is the set of queries the estimators structurally cannot answer from the same fitted object —
other interventions, rollouts, transported effects, counterfactual intervals.

## 8. Testing

TDD, per the project's practice. Tests before implementation, each failing first.

**Unit**

- Round-trip: sample from a known linear-Gaussian SCM, `fit_scm`, recover weights within tolerance.
- Round-trip discrete: sample from `build_discovery_scm()`, `fit_scm`, recover `P(V|Pa)` within
  sampling error, and `do`-query values within Monte-Carlo tolerance.
- Each family in isolation, including `invertible` correctness.
- `orient`: tiers resolve; a cycle-closing orientation is rejected; an unresolvable edge raises and
  the message names `fit_scm_mec`.
- `fit_scm` raises `NotIdentifiableError` on a graph with bidirected edges.
- `fit_scm_mec` returns one SCM per member on a small CPDAG; raises above `max_members`.
- Provenance: fitted models report `"fitted"`; hand-written ones still report `"specified"` and
  their `abduct` behaviour is byte-identical to today's.

**Counterfactual bounds**

- Binary treatment / binary outcome: `counterfactual_interval` reproduces the analytic
  Tian–Pearl PN and PS bounds.
- An all-invertible (continuous ANM) SCM returns a degenerate interval with `tight=True`, matching
  the exact `abduct(known=...)` counterfactual.
- A two-discrete-node query **refuses**, per the refinement in §6: a non-invertible node lying
  strictly between the intervention and the target raises `NotIdentifiableError` naming the NCM
  min/max, rather than composing a valid-but-loose interval. (Superseded the original plan for
  this test, which expected `tight=False` and an interval containing a known-coupling truth;
  `tight` reads `True` throughout phase 1.) A non-invertible node the intervention cannot reach,
  or one on a sibling branch off the path to the target, does not block the query.
- The bound always contains the true counterfactual across randomized couplings (property test).

**Gate**

- `run_oracle_fit_gate` in CI at reduced n and seeds; the full 10-seed version in
  `examples/learned_scm_oracle_gate.py`.

## 9. Files

```
src/causalrl/scm/fitters.py                  new — MechanismFitter protocol + the four families
src/causalrl/scm/fit.py                      new — fit_scm, fit_scm_mec, FitReport, NodeFit
src/causalrl/scm/scm.py                      edit — provenance, fit_report, abduct guard
src/causalrl/discovery.py                    edit — orient
src/causalrl/identification/counterfactual_bounds.py  new — counterfactual_interval, CounterfactualBound
src/causalrl/__init__.py                     edit — lazy exports for the five new names
tests/test_scm_fit.py                        new
tests/test_orient.py                         new
tests/test_counterfactual_interval.py        new
examples/learned_scm_oracle_gate.py          new — the kill gate
examples/learned_scm_nhefs.py                new — real-data cross-check
docs/                                        edit — tour entry, architecture map
```

## 10. Risks

| Risk | Mitigation |
|---|---|
| Fitted `do()` worse than targeted estimators on real data | §7 sets the criterion to agreement, and states the surplus is multi-query capability, not accuracy |
| Non-tight composed bounds read as tight | `tight: bool` in the return type, not in prose |
| `NeuralFit` unverifiable locally | CI is the verifier; smoke test only |
| MEC enumeration blows up | Hard `max_members` cap with an error naming the real size |
| Adding fields to released types | `CounterfactualBound` is new; `Interval` is untouched; `provenance` defaults to `"specified"` |

## 11. What this unlocks

- **Sub-project 4 (next):** plan inside the fitted SCM — `do()` rollouts, counterfactual data
  augmentation via `abduct`, policy improvement in the model. Makes "model-based" in
  `CausalMBRLAgent` literally true.
- **Sub-project 3:** active interventional discovery — the agent spends an intervention budget
  shrinking the `fit_scm_mec` belief set.
- **Sub-project 2:** NCM + identification by optimization — replaces §6's Fréchet composition with
  a constrained min/max, delivering tight bounds and latent-confounded fitting, and cross-checked
  against the library's symbolic `id_algorithm`.
