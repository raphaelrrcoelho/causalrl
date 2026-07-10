# Architecture map — Phase 0 audit (v1.2.0 → v1.3.0)

**Audited:** package `v1.2.0` (pyproject; CHANGELOG top = `[1.2.0]` 2026-07-03), on branch
`v1.3-phase0-consolidation` off `main`, 2026-07-10. Per the plan's §6.1 and risk register,
**CHANGELOG / PyPI / the code are authoritative**; this document reconciles the 2.0 engineering
plan (§4 target layout, §5 protocols) against the real tree. Nothing shipped after `v1.2.0`.

This is the plan's Phase 0 Task 1 deliverable. It is a living reference: later phases re-audit and
amend it rather than trusting it blindly.

## 1. Current layout vs. plan §4 target

`src/` layout, 7.7k LOC, `py.typed`, pyright-strict on `src`, lazy public API (§6).

| Plan §4 module | Exists today? | Current home | Phase-0 action |
|---|---|---|---|
| `certify/` | ✗ | — (3 bespoke cert types scattered) | **NEW** — unified `Certificate` + adapters |
| `data/` | ✓ (partial) | `data/dataset.py` (`ConfoundedTrajectoryDataset`) | **ADD** `data/trajectory.py` (`TrajectoryLog`) + bridge |
| `backends/` | ✗ (private `_backend/` torch seam exists) | `_backend/__init__.py` | **NEW** array-API skeleton (numpy); leave `_backend` as-is |
| `graphs/` | ✗ | graph types under `scm/graph.py`, `identification/` | **NEW** re-export shim + `graph_hash()` (no code move) |
| `identify/` | ✓ (as `identification/`) | `identification/` | keep; graph-level, type-agnostic |
| `bounds/` | ✓ | `identification/bounds.py` | keep; `continuous/` is Phase 1 |
| `estimate/` | ✗ (seam in `identification/estimate.py`) | `identification/estimate.py` | Phase 1 |
| `scm/` | ✓ | `scm/` (`graph`, `mechanisms`, `scm`, `unrolled`) | keep; `continuous/` is Phase 1 |
| `learners/` | ✓ (as `agents/`) | `agents/` | keep (name differs; no rename in Phase 0) |
| `policies/` | ✓ (scattered) | `agents/counterfactual.py`, `imitation.py`, `shaping.py` | keep |
| `transport/` | ✓ (in `identification/`) | `identification/transport*.py` | keep; estimation layer is Phase 1 |
| `interop/` | ✓ | `interop/` (`dowhy`, `econml`) | keep; pettingzoo/sbi are Phase 2/4 |
| `scale/` | ✓ | `scale/d3rlpy.py` | keep; extend in Phase 4 |
| `magames/`, `meanfield/` | ✗ | — | Phase 2 |
| `experimental/` | ✓ | `experimental/ope.py` | keep; Phase 5 `cyclic/` later |

**Dependency rule (§4):** `certify`, `graphs`, `data`, `backends` are leaves anyone may import;
`interop`/`scale` may import anything; nothing imports `interop`/`scale`/`experimental`.

## 2. Shipped certificate types → unified `Certificate` (§5.2)

Three bespoke types today (I1: unify by **generalization, not replacement** — adapters, no field
changes; two are `NamedTuple`, one a plain class, so inheritance is out):

| Type | Home | Fields | → unified `Certificate` |
|---|---|---|---|
| `DecisionCertificate` (`NamedTuple`) | `identification/decision.py:29` | `decision, naive_contrast, certified, pivotality, tipping_gamma, msm_certified, summary` + `.recommendation` (`"act"/"abstain"`) | `kind=BOUNDED`; `value=naive_contrast`; `hedge` **absent iff** `recommendation=="act"`; `Assumption("MSM", {gamma_max})` |
| `PivotalityCertificate` (`NamedTuple`) | `identification/bounds.py:268` | `certified, naive, bias_bound, mi_flip, mi_measured` | `kind=BOUNDED`; `Assumption("mi-cap", {mi_flip, mi_measured})`; `value=bias_bound`; hedge if not `certified` |
| `TransportRegretCertificate` (class) | `identification/transport_regret.py:72` | `transportable, formula, non_transportable_witness, reweight_required, decision_dependence, value_range, regret_bound: Interval` | `kind=BOUNDED`; `witness`=selection-marked mechanism (`non_transportable_witness`); `value=regret_bound`; `Assumption("selection-nodes-S")` |

Mechanism: additive `as_certificate()` (method on each + a dispatch function in `certify/`).
Serialization round-trip through the unified type is Phase-0 acceptance #2.

## 3. Naming collisions — shipped names are frozen (I9)

The plan's §5 signatures are "normative in shape, not exact naming." Three names are already taken;
adapt the new types to avoid clobbering:

| Plan name (§5) | Already shipped as | Phase-0 name chosen |
|---|---|---|
| `Estimand` (§5.2, mean\|quantile\|tail target) | `identification.id_algorithm.Estimand` (ID query target; top-level export) | **`EstimandSpec`** in `certify/` (field still named `estimand`) |
| `CausalEnv` (§5.1 Protocol) | `envs.base.CausalEnv(gym.Env)` (env base class; top-level export) | **`CausalEnvProtocol`** (new module; distinct from the gym base) |
| `Domain` (§5.3) | `identification.id_algorithm.Domain` (top-level export) | reuse — `Regime` **builds on** `Domain`/`SelectionDiagram`, not beside |
| `Interval` (§5.2 `value`) | `identification.bounds.Interval` (`NamedTuple`; top-level export) | reuse directly |

## 4. Routines returning bare numerics (certified-variant candidates — Task 3/7)

Already certificate-returning: `certify_decision`, `certify_estimate`, `certify_policy`,
`pivotality_certificate`, `transport_regret_certificate`.

Bare-return (candidates for additive `return_certificate=True` variants, deprecation per I9):
`identify_effect`/`is_identifiable_effect`/`estimate_effect`, `manski_bounds`, `causal_q_bounds`,
`ipw_sensitivity_bounds`, `msm_policy_value_bounds`, `msm_contribution_bounds`,
`msm_per_step_bounds`, `msm_stratified_bounds`, `confounding_bias_bound`,
`confounding_bias_per_step_bounds`, `tipping_gamma`, `mi_flip_threshold`, `transported_effect`,
`estimate_transported_effect`, `ipw_value`.

**Phase-0 representative trio (acceptance #3 — one ID, one bound, one policy-value):**
`identify_effect` → `IDENTIFIED`; `ipw_sensitivity_bounds` → `BOUNDED`;
`msm_policy_value_bounds` → `BOUNDED`. Remaining routines get variants in later 1.x minors.

## 5. `ConfoundedTrajectoryDataset` → `TrajectoryLog` bridge (§5.4, acceptance #4)

Dataset (`data/dataset.py`): an ordered `list[Transition(state:int, action:int, reward:float,
next_state:int, done:bool)]` + `n_states`, `n_actions` (single-agent, discrete/tabular). Derived
stats: `behavior_propensity(s,a)`, `mean_reward(s,a)` from `counts`/`reward_sums`.

Lossless mapping to the §5.4 long schema (`entity_id, episode_id, t, kind, name, value, regime,
observed`): each transition → a `(entity_id=0, episode_id, t)` row-group with `state`(obs),
`action`(action), `reward`(reward), `next_state`(obs), `done`(done) as `name`d values;
`episode_id`/`t` reconstructed from the `done` flags; `regime="observed"`, `observed=True`;
`n_states`/`n_actions` carried as log-level metadata. Reverse reads the ordered groups back into
`Transition`s. Round-trip preserves the transition list exactly (⇒ derived stats identical),
pinned by a test on the d3rlpy example dataset.

## 6. Public API mechanics

`src/causalrl/__init__.py`: `_EXPORTS: dict[name -> (module, attr)]` + lazy `__getattr__`, with a
`ModuleNotFoundError`→torch handler emitting *"install the 'causalrl[torch]' extra"*. New Phase-0
exports (`Certificate`, `Kind`, `Assumption`, `EstimandSpec`, `Witness`, `Hedge`, `Provenance`,
`as_certificate`, `TrajectoryLog`, `graph_hash`, `Regime`) follow this exact pattern; add an
analogous pyarrow handler for the `[data]` extra. `__version__ = importlib.metadata.version(...)`.

## 7. CI gates + local verification reality

Gates (must stay green): pyright-strict on `src`, ruff, `pytest --cov` (≥90% gate), nbmake
executed examples, `certify_decision` byte-for-byte regression pin, `bench_causal_core` fast-path
guards (≥5× floor), equilibrium ε-verification. CI matrix Linux/macOS/Windows; trusted PyPI
publish on GitHub Release.

**Local env is unreliable (my notes + probe):** `.venv` (py3.14) has `torch` and `pyarrow` as
**empty namespace dirs** (import succeeds, no real module), and editable metadata reports a stale
`__version__` (`1.0.0`). Strategy: numpy-core work is TDD'd locally (30/30 torch-free core tests
pass in ~17s); pyarrow Parquet-IO tests are gated with `pytest.importorskip("pyarrow.parquet")` and
exercised in **CI** (where the `[data]` extra installs real pyarrow); provenance tests assert *a*
version is recorded, never a specific string.

## 8. `experimental/`

`experimental/ope.py` — `confounding_sensitivity_bounds(point, gamma)`, explicitly documented as a
*qualitative, non-validated* sensitivity interval (not the published MSM bound). Phase 5's
`experimental/cyclic/` lands here first.
