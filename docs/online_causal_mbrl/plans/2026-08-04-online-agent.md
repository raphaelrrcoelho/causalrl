# Online causal MBRL — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development.

**Goal:** an agent that learns the SCM while acting — on-policy interventional data plus off-policy
observational logs — and replans as the belief sharpens.

**Architecture:** one class, `src/causalrl/agents/online_causal_mbrl.py`, built entirely on existing
primitives (`discover_interventional` → I-MEC, `fit_scm_mec` → belief, `scm.do().see()` → rollout
values). See `docs/online_causal_mbrl/DESIGN.md` for rationale and prior-art citations.

## Global Constraints

- **Branch:** `online-causal-mbrl`. Design doc already committed (`d58877c`).
- **No `Co-Authored-By` / `Generated with` trailer** in any commit message.
- **This is an implementation of published methods, not a contribution.** Every docstring that
  describes an algorithm cites its source (see DESIGN.md §"This is not novel"). Do not write "novel",
  "new approach", or "we propose". Use the field's vocabulary: **I-MEC**, active intervention design,
  Thompson sampling over structure.
- **Route every causal operation through the library.** Do not reimplement discovery, orientation,
  MEC enumeration, or fitting. If a library function is close but not sufficient, extend it in place
  rather than inlining a private copy.
- **Existing public APIs must not change.** New parameters are keyword-only with defaults preserving
  current behaviour.
- **Run the full CI gate before every commit**, in CI's order:
  ```
  uv run --no-sync ruff check . --exclude examples/causal_corr2cause_prompted.py
  uv run --no-sync ruff format --check . --exclude examples/causal_corr2cause_prompted.py
  uv run --no-sync pyright src
  uv run --no-sync pytest --cov-fail-under=90
  ```
  Baseline to match or beat: ruff clean, pyright 0/0/0, exit 0, coverage 96.75%.
- **`--no-sync` on every `uv run`.** Never `uv sync`, `uv venv`, or `uv run --python X` — each rebuilds
  the shared virtualenv and one of them destroyed this machine's environment on 2026-08-03.
- **Never delete or overwrite anything you did not create.** Untracked files from unrelated projects
  live in this tree (`.b1_*.pt`, `.b3_mech_smoke.log`, `docs/causal_llm/`, `examples/presentation/`,
  `examples/results/`, `skills/`, `examples/causal_corr2cause_prompted.py`). No `rm`, no `git clean`,
  no `find -delete`, no truncating redirection, no chaining destructive commands.
- **`git add` by explicit path.** Never `git add -A` / `git commit -a`.
- **Every test must discriminate.** Before claiming a test covers a behaviour, mutate that behaviour
  and confirm the test fails. Five tests on the predecessor branch had to be rebuilt because they
  asserted an outcome correct code produces without ever failing against broken code.
- Line length 100; pyright strict over `src`; data format `Mapping[str, np.ndarray]`, columnar.

---

### Task 1: buffers, `refit`, and the belief

**Files:** create `src/causalrl/agents/online_causal_mbrl.py`, `tests/test_online_causal_mbrl.py`.

**Produces:**
`OnlineCausalMBRL(variables, *, treatment, outcome, actions, policy="thompson", max_members=32,
refit_every=8, n_rollout=512, seed=0)` with `ingest`, `observe`, `refit`, `belief`, `belief_size`,
`structure_uncertain`, `history`.

- `ingest(data: Mapping[str, np.ndarray], *, source: Literal["observational","interventional"] =
  "observational", target: str | None = None)` — bulk load. `source="interventional"` requires
  `target`; `source="observational"` forbids it. Raise `ValueError` naming the offending argument.
- `observe(row: Mapping[str, float], *, intervention: Mapping[str, float] | None = None)` — one
  transition. With `intervention`, the row joins that target's buffer; exactly one target is supported
  per row (raise on multi-node interventions, naming the limitation — `discover_interventional`
  orients per single target).
- `refit()` — `discover_interventional(obs, int_buffers, variables)` then `fit_scm_mec(all_rows,
  cpdag=..., max_members=...)`. Let `fit_scm_mec`'s over-size refusal propagate; do not catch it.
  Append `(step, belief_size)` to history.
- `belief()` returns the fitted members; `belief_size()` its length; `structure_uncertain()` is
  `belief_size() > 1`.

**Tests that must discriminate** (mutate and confirm failure before committing):
1. Observational data alone leaves `belief_size() > 1` on a chain `X -> Y -> Z` whose CPDAG is
   unoriented; after ingesting `do(X)` data the belief shrinks. *Mutation: drop the interventional
   buffer from the `refit` call — the belief must then stay large.*
2. `ingest(..., source="interventional")` without `target` raises, and with `source="observational"`
   plus `target` raises. Assert on the message naming the argument.
3. A multi-node `intervention` raises naming the single-target limitation.
4. History records one entry per `refit()` call, in order.

### Task 2: `probe` and the three policies

**Files:** modify `src/causalrl/agents/online_causal_mbrl.py`, `tests/test_online_causal_mbrl.py`.

- `act()` — returns an action from `actions`. `policy="thompson"`: seed-derived RNG draws one member,
  return `argmax_a member.do({treatment: a}).see(n_rollout)[outcome].mean()`. `"average"`: mean value
  across members. `"robust"`: maximin across members. Deterministic given `seed`.
- `probe()` — returns the intervention **target** whose implied `do()` distribution the belief members
  disagree about most: for each candidate target, mean pairwise total variation of the outcome
  distribution under `do(target)` across members; return the argmax, ties broken by name. Raise if
  the belief has one member (nothing to disambiguate) — `structure_uncertain()` is the guard callers
  are told to use.

**Tests that must discriminate:**
1. `policy="thompson"` is reproducible under a fixed seed and *varies* across seeds while the belief
   has >1 member. *Mutation: ignore the sampled member and always use member 0 — the across-seed
   variation must vanish.*
2. `"robust"` picks the maximin action on a hand-built two-member belief where the members disagree,
   and `"average"` picks the higher mean. Construct the two members directly so the expected answers
   are exact, not statistical.
3. `probe()` returns the target the members disagree on, not one they agree on. *Mutation: return the
   first candidate by name — must fail.*
4. `probe()` on a singleton belief raises, naming `structure_uncertain`.

### Task 3: the demonstration, exports, docs

**Files:** create `examples/online_causal_mbrl.py`, `tests/test_online_causal_mbrl_example_smoke.py`;
modify `src/causalrl/__init__.py`, `src/causalrl/agents/__init__.py`, `tests/test_public_api.py`,
`docs/online_causal_mbrl/DESIGN.md` (results section), `CHANGELOG.md`.

- A synthetic world with exact ground truth where observation alone leaves ≥2 I-MEC members that
  **disagree about the optimal action**. Three arms: observational-only (plateaus wrong),
  interventional-only (correct, sample-hungry), both (best). Print belief size over time beside regret.
- The example must state in its own printed output that it is a synthetic world with known ground
  truth and not a benchmark result — match the tone of `examples/learned_scm_policy.py`.
- Export `OnlineCausalMBRL` in **both** `_EXPORTS`/`__all__` (top level) and the agents package, in
  sorted position, and extend `tests/test_public_api.py`. *(The predecessor branch shipped a class
  that was reachable only by full module path because this step was skipped — do not repeat it.)*
- Smoke test runs the example on a tiny budget and asserts on stable quantities (belief sizes, the
  ordering of the three arms' final values), never on which arm a truncated run happens to pick.

---

## Verification

- `uv run --no-sync ruff check . --exclude examples/causal_corr2cause_prompted.py`
- `uv run --no-sync ruff format --check . --exclude examples/causal_corr2cause_prompted.py`
- `uv run --no-sync pyright src`
- `uv run --no-sync pytest --cov-fail-under=90`
- Run `examples/online_causal_mbrl.py` end to end and paste its output.
- Push, open a PR, and watch CI. Do not claim green before the run passes.
