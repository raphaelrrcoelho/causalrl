# RL recentering: conversion plan

**Status:** plan, 2026-08-04. Target release **v3.0.0** (breaking — removals and renames permitted by
the owner on 2026-08-04).

## Why

A library-wide audit (import graph over all 122 modules, reachability computed in both directions
from `agents/` and `envs/`) found that `causalrl` is **two disjoint halves touching at one point**:

| Half | Hub | Contents | LOC |
|---|---|---|---|
| A — agent half | `ConfoundedTrajectoryDataset` | `agents/`, `envs/`, `identification/`, `scm/`, `certify/`, `discovery`, `eval/` | ~11.4k |
| B — columnar half | `TrajectoryLog` | `estimate/`, `bounds/`, `conformal/`, `backends/`, `interop/`, `magames/`, `meanfield/` | ~6.4k |

The only cross-edge in `src/` is `transport/estimate.py:36`. No test exercises both halves together.
`agents/` never imports `envs/`. **37 of 228 exported symbols (16%) are RL.**

The worst finding is inside the RL half, not outside it: **five of six back-door planners' `act()`
return a constant and discard the observation entirely** — `return int(self._best_action)`, identical
at `agents/mbrl.py:181,246,305,388,495`. Nine of fifteen public agent classes never update from
experience. (Correction to `docs/learned_scm/PROGRESS.md:162`: it is 5 of 6, not 6 — the `"sequential"`
route dispatches to `DOVI`, which does genuine backward induction at `agents/dovi.py:100-111`.)

## Global constraints

- **Branch:** `rl-recentering`. Breaking changes are permitted; the release is **v3.0.0**.
- Every removal or rename gets a **CHANGELOG migration entry** naming the old symbol, the new one (or
  the reason for removal), and what a 2.1.0 user must change.
- **A conversion must be real.** Renaming a module to say "off-policy evaluation" while nothing under
  `agents/` can call it is the same deviation this plan exists to fix, better disguised. For every
  rename, name the code path that makes it true.
- No `Co-Authored-By` / `Generated with` trailer. `git add` by explicit path.
- Full CI gate before every commit (`ruff check`, `ruff format --check`, `pyright src`,
  `pytest --cov-fail-under=90`), `--no-sync` on every `uv run`, never `uv sync`/`uv venv`/`uv run --python X`.
- Never delete or overwrite untracked files belonging to other projects.
- Every test must discriminate: mutate the behaviour it names, confirm failure, revert.

---

### Task 1 — `act()` returns a policy, not a constant  *(highest leverage)*

`GFormulaBackdoorAgent` computes per-unit CATE at `agents/mbrl.py:473-486` (`mu1 - mu0`) and then
discards it: `act` returns `self._best_action` at `:495`. The library already performs exactly this
conversion **for a third-party model** — `interop/econml.py:34-35` does `tau = effect(X); pi = (tau > 0)`
→ `PolicyValueContrast`. Do it for ourselves.

- Cache the per-action ridge weights and standardisation stats in `fit()` (`:432-444`).
  `_action_predictions` (`:445-461`) currently refits on every call because it takes the training
  arrays as parameters — that is why `act` cannot use it.
- `act(observation)` builds a one-row design and returns `argmax_a mu_a(x)`.
- Apply the same pattern to `FunctionApproxBackdoorAgent` (already has per-action RBF features) and
  `BackdoorAdjustedAgent` (stratum lookup keyed by the observation).
- **Discriminating test:** two observations whose CATE signs differ must produce different actions.
  *Mutation: return `self._best_action` — the test must fail.* A test that only checks `act` returns a
  valid action passes against the constant and is worthless.
- Where an agent genuinely cannot condition on an observation, say so in its docstring rather than
  accepting a parameter it ignores.

### Task 2 — the real-data examples call the RL front door

- `examples/causal_mbrl_obd.py:65-70` binarises 80 bandit arms into `treated`/`control` to fit
  `certify_decision`'s signature. Use `certify_policy` on the 80-arm policy value the script already
  computes at `:46`.
- `examples/causal_mbrl_coat.py` **imports no causalrl function at all** and re-implements the
  certificate layer by hand at `:81-89`. Replace with `certify_policy` / `msm_contribution_bounds`.
- `causal_mbrl_twins.py`: assignment is synthetically 0.5 (`:48`), so it *is* a two-armed bandit with
  known propensity — score the CATE-induced policy's off-policy value, not PEHE.
- `causal_mbrl_lalonde.py:52` already computes an assign/kill decision; make that the headline as a
  policy value and score regret against the RCT ground truth.
- `.act()` is called nowhere in the real-data suite. After Task 1 it should be.

### Task 3 — `causalrl.ope`

Move the OPE surface, today scattered across five packages, into one: `eval/ope.py::ipw_value`,
`estimate/streaming.py::stream_policy_value`, all four of `estimate/sequential.py`,
`identification/bounds.py::{msm_policy_value_bounds, msm_contribution_bounds, msm_per_step_bounds,
causal_q_bounds}`, `scale/__init__.py::certify_policy`. Deprecation shims at the old paths.

**Honest only after Task 1**, which gives the package a caller. The existing path is
`certify_policy` ← `agents/mbrl.py:133` (`CertifiedPolicyAgent`).

### Task 4 — delete the statistics tooling

No RL concept these are instances of. Internal references measured at 0-3 each.

- `bounds/continuous.py:56-146,230-313` — `certify_mean`, `certify_quantile`, `moment_diagnostic`,
  `tail_index_hill`, `weighted_quantile`, `bootstrap_quantile_ci`, `MomentDiagnostic`. `certify_mean`
  emits `EstimandSpec(query="do", …)` at `:303` for a computation containing **no intervention** —
  actively misleading provenance.
- `estimate/streaming.py:140` `stream_quantile_certificate` — its own docstring at `:155` says
  *"not a causal effect"*.
- `backends/quantile_sketch.py:37` `GKQuantileSketch` — exists only to support the above.
- `experimental/ope.py` — 19 lines, explicitly *not* the published MSM bound (`:11-12`), its own test
  asserts it is not public. Superseded by `identification/bounds.py`.
- `meanfield/` — *"Evaluation-only (no learning)"* (`__init__.py:8`), zero consumers.
- `estimate/nuisance.py:25-78`, `_stats.py:48` — a hand-rolled mini-sklearn. Keep as **internal**
  nuisance models; drop from `__all__`.

### Task 5 — rename the CI-named parameters on the decision front door

The docstrings already say the RL thing; only the parameter names disagree.

- `certify_decision(outcomes=, treated=)` → `(rewards=, action=)`. Docstring `decision.py:147` already
  says *"`outcomes` are logged rewards"* and `:158` *"the off-policy (IPS) value contrast"*.
- `DecisionCertificate.decision` labels `"prefer treated"/"prefer control"` (`decision.py:39`) → action
  labels. An 80-arm bandit log currently gets a "prefer treated" verdict.
- `estimate_sequential_value(treatments=, outcome=)` → `(actions=, reward=)` (`sequential.py:253,255`).
- `interop`: `from_dowhy_estimate` → `policy_contrast_from_dowhy`, `from_econml_cate` →
  `policy_from_econml_cate`. Neither is top-level exported: free.
- `certify_sequential_transport` → `certify_transported_policy_value` (it already emits
  `EstimandSpec(query="policy_value")` at `transport/estimate.py:311`).

### Task 6 — `conformal_action_value`

`conformal/core.py:46` `conformal_quantile(weights=…)` already accepts likelihood ratios
`dP_test/dP_cal`, which for off-policy evaluation is exactly `π_target/π_behavior`. Add
`conformal_action_value(dataset, policy, …)` computing that ratio internally, and wire it into
`certify_policy` as a lower-confidence-bound gate. Turns 177 orphan lines into safe policy improvement.
Note `certify_conformal_interval(query="counterfactual")` (`core.py:125`) is currently a **string label**
with no counterfactual mathematics behind it — fix or drop the claim.

### Task 7 — `magames`: add the learner or demote it honestly

874 lines of RL vocabulary with zero learners. `cce_regret`'s docstring (`magames/cce.py:186`) says
*"it is what a no-regret population drives to 0"* — the seam is designed for a learner and is empty.
Add `run_no_regret(population, rounds)` producing the empirical joint that `cce_regret` /
`certify_cce_do` already accept. If that is out of scope, relabel the package as game theory and say so.

### Task 8 — docs say what the library is

- `pyproject.toml:4` says *"Causal reinforcement learning…"*; `README.md:10`, rendered directly beneath
  it on PyPI, says *"Causal intervention-selection and causal-RL research tools."* Reconcile.
- `README.md:18-19` actively disclaims RL (*"learning agents are tabular/demo-scale, not production RL"*)
  and `:107-121` positions the library against six causal-inference libraries and zero RL ones.
- `docs/api.md`: Gymnasium wrapper, CGFA and env registration are at slots **18/19/20 of 28**. Move to
  the front.
- `docs/guides.md`: **zero of five** core workflow guides train an agent. Add one.
- `docs/tour.md:8-25` opens with a DoWhy/EconML comparison table, ahead of taxonomy Task 1.
- Vocabulary census: README `confound*` 16 > `agent` 15 > `reward` 4, **`regret` 0**.
- Add an RL classifier to `pyproject.toml:23-35`; `mkdocs.yml:2` drops "reinforcement learning" entirely.

**Worthless before Tasks 1-2 make it true.** Do this last.

### Task 9 — `factored_advantage.py` is not CGFA-PPO

The whole computation is `per_factor = fv - bl[:, np.newaxis]` plus a matvec (`:190-195`) — a
subtraction. The K-head critic that makes CGFA-PPO an algorithm exists nowhere, and
`examples/cgfa_ppo_example.py:100-112` concedes *"In a full CGFA-PPO, you'd have K value heads; here we
use the same value"*. Either implement the K-head critic or rename the module and its claims to what it
is: a factored-advantage decomposition primitive.

---

## Verification

Full gate per commit. Push, PR, watch CI. `v3.0.0` requires the CHANGELOG migration section to be
complete before release.
