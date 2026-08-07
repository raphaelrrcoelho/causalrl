# kaggle-integration rebase: gate report

Worktree `.claude/worktrees/kaggle-rebase`, branch `kaggle-integration`.
Four commits developed from `29ec740`, cherry-picked onto `main` (`a73e0e7`) as
`1e57400 → 8251668 → 71b9a16 → 2482a41`. Fix commit on top: **`1a9ef4c`**.

**Status: green.** Two integration failures, both the same root cause, both fixed.
Nothing was found that needs a design decision from you.

---

## 1. Gate results

Environment: this worktree's own `.venv`, created once with
`uv sync --extra dev --extra data` (Python 3.14, matching `.python-version` and one leg
of CI's `["3.11", "3.14"]` matrix). Everything after that ran under `uv run --no-sync`.

| Gate | Result | Exit |
|---|---|---|
| `ruff check . --exclude examples/causal_corr2cause_prompted.py` | All checks passed | 0 |
| `ruff format --check .` (same exclude) | 353 files already formatted | 0 |
| `pyright src` | **0 errors, 0 warnings, 0 informations** | 0 |
| `pytest --cov-fail-under=90` | **1052 passed, 4 skipped** in 435.69s | 0 |
| coverage | **97.04%** (7690 statements, 228 missed) | — |

Against the pre-rebase reference from `main` (ruff clean, pyright 0/0/0, green, 96.79%):
ruff and pyright match exactly; the suite grew by the branch's new tests and coverage
rose 96.79% → 97.04%.

The 4 skips are all module-scope `importorskip`s for optional extras that the `dev`+`data`
matrix deliberately does not install:

```
tests/test_bayesian_fit.py:9    no module named 'numpyro'
tests/test_scm_nuts.py:14       no module named 'numpyro'
tests/test_backends_jax.py:12   no module named 'jax'
tests/test_cgfa_example_smoke.py:33  no module named 'stable_baselines3'
```

NumPyro and JAX publish no py3.14 wheels and `pyproject.toml` markers exclude them there by
design; CI exercises each on its own dedicated py3.11 lane. Not a rebase effect — the same
skips occur on `main`. Worth flagging only because `tests/test_bayesian_fit.py` is what
pins `family == "bayesian_linear"`, so that assertion did not run here (see §3, where I
verified it directly instead).

---

## 2. What the rebase broke, and the fix

### Both failures: phase 2's exports were never placed or documented

The branch's fourth commit (`2482a41`) added two surface-curation gates. Both failed on
the rebase, and both named exactly the same two symbols:

```
tests/test_public_api.py::test_api_tiers_partition_the_public_surface
    set(tiered) != set(causalrl.__all__) - {"__version__", "API_TIERS"}
    in __all__ but untiered: ['BayesianLinearFit', 'PoissonGLMFit']

tests/test_public_api.py::test_every_export_appears_in_the_api_reference
    AssertionError: 2 exported name(s) absent from docs/api.md:
    ['BayesianLinearFit', 'PoissonGLMFit']
```

Cause: `PoissonGLMFit` and `BayesianLinearFit` arrived on `main` in the learn-the-SCM
phase-2 merge, *after* `29ec740`. `API_TIERS` and the completeness section of
`docs/api.md` were both written on the branch against the 253-name surface that predated
them, so the two names reached `causalrl.__all__` with no tier and no `:::` entry.

This is the gates doing their job, not a false alarm — the branch built them precisely to
make that omission loud, and they caught the first export that slipped through. The fix is
therefore to place and document the names, not to touch the tests:

- `src/causalrl/__init__.py` — `BayesianLinearFit` and `PoissonGLMFit` added to
  `API_TIERS["modelling"]`, in the tuple's existing ASCII order, beside the other five
  shipped fitters (`ANMFit`, `LinearGaussianFit`, `NeuralFit`, `PinnedMechanism`,
  `TabularCPT`).
- `docs/api.md` — two entries in `## Structural Models & Data — Complete Reference`, in
  that section's ASCII order:
  `::: causalrl.scm.continuous.bayesian_fit.BayesianLinearFit` and
  `::: causalrl.scm.fitters.PoissonGLMFit`. `BayesianLinearFit` is documented at the
  attribute its lazy export map entry points at; that module imports without numpyro
  (only `fit()` raises), so mkdocstrings resolves it on the main matrix.

No test was weakened, no assertion removed, and no export changed meaning.

### Collateral: the CHANGELOG's `[Unreleased]` sectioning

Not caught by any gate, but a genuine cherry-pick casualty. `main` had no `### Fixed`
heading under `[Unreleased]`; the branch introduced one. The merge placed that heading at
the *top* of `main`'s existing `### Added` run, so nine Added entries ended up filed under
Fixed — including `main`'s own `cce_polytope` and `PoissonGLMFit`/`BayesianLinearFit`
bullets, and the branch's `FunctionalManskiBounds`, `BoundedFittedQIteration`,
`StateEncoder`, `FittedQIteration`, `InterventionSpace`, `AdmissibleInterventions`,
`ExposureMapping`, `PinnedMechanism` and `Deadline` entries.

Fixed by moving the `### Fixed` heading below the Added block, carrying its three genuine
entries with it. Two counts that described the pre-rebase surface were also updated to the
post-rebase reality: "253-name public surface" → 255, and "127 of 253 were missing" → 129
of 255 (the two names above raise both).

---

## 3. The two hand-resolved conflicts: verified

### `src/causalrl/scm/scm.py` — the L3 abduction guard

The merge is correct. It keeps the branch's widened predicate and provenance-aware message
*and* `main`'s lean single `sorted(...)` binding:

```python
if self.provenance in ("fitted", "mixed"):
    ambiguous = sorted(self.non_invertible_nodes())
    if ambiguous:
        raise NotIdentifiableError(
            f"counterfactuals on a {self.provenance} SCM are not identified: node(s) "
            f"{ambiguous} have a non-invertible mechanism, ...",
            witness=ambiguous,
        )
```

Exercised directly against the built library (not only via the suite):

| Case | Observed |
|---|---|
| `mixed` model, `X` learned via `TabularCPT` (non-invertible), `Y` pinned | `provenance="mixed"`, `non_invertible_nodes()==['X']`, **refuses**: "counterfactuals on a **mixed** SCM are not identified: node(s) `['X']` …", `witness=['X']` |
| all-pinned model | `provenance="specified"`, guard skipped, `abduct(known=...)` returns an `ExogenousPosterior` — **not** refused |
| plain `fitted` model, three non-invertible nodes | refuses with `witness=['A', 'Y', 'Z']` — sorted, which a multi-node case actually discriminates |

`witness` carries the same sorted list object the message interpolates, so the two can
never disagree. `tests/test_scm_pinned.py::test_a_mixed_model_is_still_gated_for_point_counterfactuals`
matches on `"mixed SCM"` and passes; the two `main`-side guard regressions in
`tests/test_scm_fit.py` match on `"counterfactual_interval"` and still pass, so widening
the predicate did not disturb the `fitted` path.

### `src/causalrl/scm/fit.py` — `_FAMILY_NAMES`

Keeping all three additions is correct and complete. There are exactly seven concrete
`fit(...) -> FittedMechanism` implementations in the tree (six in `scm/fitters.py`, plus
`BayesianLinearFit` in `scm/continuous/bayesian_fit.py`; the eighth match is the
`MechanismFitter` Protocol stub). `_family_name` returns the snake_case name for every one
— none falls through to the class-name fallback:

```
TabularCPT        -> tabular_cpt      NeuralFit         -> neural
LinearGaussianFit -> linear_gaussian  PoissonGLMFit     -> poisson_glm
ANMFit            -> anm              BayesianLinearFit -> bayesian_linear
                                      PinnedMechanism   -> pinned
```

Each is pinned by a test: `poisson_glm` in `tests/test_scm_fit.py:324`, `bayesian_linear`
in `tests/test_bayesian_fit.py:128` (numpyro lane), `pinned` and `anm` in
`tests/test_scm_pinned.py:36,124`.

Also checked in passing, since the same conflict region touches it: `_provenance` maps
no-pinned → `"fitted"`, all-pinned → `"specified"`, mixed → `"mixed"`, which is what the
guard above relies on and what `tests/test_scm_pinned.py` asserts for all three.

---

## 4. Reported, not fixed — pre-existing, not rebase casualties

These are branch-side or main-side issues that the rebase merely made visible. No gate
fails on any of them and none was touched.

1. **`PinnedMechanism` is missing from `causalrl.scm`'s re-export.** It is exported
   top-level from `causalrl` and documented, but `src/causalrl/scm/__init__.py` lists
   every *other* shipped fitter (`ANMFit`, `BayesianLinearFit`, `LinearGaussianFit`,
   `NeuralFit`, `PoissonGLMFit`, `TabularCPT`) in its `_LAZY` map and `__all__` and omits
   this one. Verified against the built library: `from causalrl.scm import TabularCPT`
   works, `from causalrl.scm import PinnedMechanism` raises
   `ImportError: cannot import name 'PinnedMechanism' from 'causalrl.scm'`, while
   `from causalrl import PinnedMechanism` and `from causalrl.scm.fitters import
   PinnedMechanism` both work. Branch-side omission in `1e57400`, present before the
   rebase; nothing tests the subpackage surface, so nothing caught it.

2. **`FunctionalManskiBounds` annotates its actions array inconsistently.**
   `2482a41` corrected `fit(..., actions: NDArray[np.int_], ...)` — "what it has always
   required and coerced". But the two helpers it hands that array to,
   `_fit_fold(self, x: FloatArray, a: FloatArray, r: FloatArray)` and the `a` it derives,
   still say `FloatArray`. Pyright accepts it (numpy's `asarray(..., dtype=int)` widens),
   so this is cosmetic — but it contradicts the annotation the same commit just fixed one
   call up.

3. **`_provenance([])` returns `"fitted"` for an empty fit list.** The `pinned == 0` branch
   is checked before the all-pinned branch, so a zero-node model is reported as learned
   rather than specified. Degenerate and unreachable through `fit_scm` (a graph with no
   nodes), so it costs nothing today; noted only because the same function is the guard's
   input.

4. **The local venv is py3.14, so the numpyro-backed assertions do not run here.**
   `.python-version` pins 3.14 and `pyproject.toml` markers exclude numpyro/jax there, so
   `tests/test_bayesian_fit.py` (which is what pins `family == "bayesian_linear"`) skips
   in this environment and only runs on CI's py3.11 lane. Pre-existing by design, not a
   rebase effect — I verified `_family_name(BayesianLinearFit())` directly instead of
   relying on the skipped test. Consequence for whoever merges: the `bayesian_linear`
   entry in `_FAMILY_NAMES`, one of the three names the `fit.py` conflict had to keep, is
   *only* covered by a lane this run could not exercise. It is confirmed correct here by
   direct call, but the merge should still go green on CI's py3.11 numpyro lane before the
   branch lands.

---

## 5. Also checked, clean

- No conflict markers anywhere in `src/`, `tests/`, `docs/`.
- No `:::` entry present on `main`'s `docs/api.md` was lost in the merge (102 before,
  258 after, zero dropped).
- `causalrl.__all__` (257) and `_EXPORTS` (255 + `__version__` + `API_TIERS`) agree exactly
  in both directions; no duplicate tier entries; `core` is 14 names, within the ≤ 20 the
  branch's own gate allows.
- `tools/generality_lint.py` (§12.4, its own CI lane): clean, no domain-noun leakage from
  the branch's new public surface.
