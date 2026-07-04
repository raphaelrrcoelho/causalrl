# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.2.0] - 2026-07-03

The confounded-offline-RL scale path: d3rlpy trains, causalrl certifies. Learn a policy with any
d3rlpy offline algorithm, then bound whether its improvement over the behaviour policy survives
hidden confounding — causalrl supplies the causal layer, not a new trainer. Backward compatible.

### Added
- `causalrl.certify_policy` — certify a learned policy's off-policy value improvement
  `V(pi) - V(behaviour)` against hidden confounding under Tan's marginal sensitivity model. The
  learned policy is supplied as its chosen action per logged transition (e.g. a d3rlpy policy's
  greedy prediction), so the certificate needs no trainer dependency; the nominal logging
  propensities come from the dataset's empirical `behavior_propensity`. Returns a
  `DecisionCertificate`.
- `causalrl.scale.d3rlpy.to_mdp_dataset` — bridge a `ConfoundedTrajectoryDataset` into a d3rlpy
  `MDPDataset` (tabular states one-hot encoded). New optional `scale` extra
  (`pip install causalrl[scale]`); the d3rlpy import is lazy so causalrl core never depends on it.
  Runnable example under `examples/scale_d3rlpy_certify.py` and a Scale guide in the docs.

### Scope
- One-step / terminal-return contrast; the per-step cumulative-reward extension uses
  `msm_per_step_bounds`. The behaviour value is the logged empirical return (on-policy, unconfounded);
  the MSM sensitivity is on the logging propensities. d3rlpy owns training.

## [1.1.0] - 2026-07-03

The decision certificate becomes the ecosystem's confounding-robustness front door: any off-policy
value contrast — yours, DoWhy's, or EconML's — can now be certified against hidden confounding
through one typed seam. Backward compatible: `certify_decision`'s existing signature and outputs are
unchanged (its raw-logs path is byte-for-byte identical, pinned by a regression test).

### Added
- `causalrl.certify_estimate` + `causalrl.PolicyValueContrast` — the general decision front door.
  `PolicyValueContrast` is the typed seam (per-unit outcomes, logging propensities, and two target
  policies' action probabilities); `certify_estimate` runs the marginal-sensitivity-model tipping
  layer over any `V(pi_on) - V(pi_off)` contrast, plus the structural pivotality layer when the
  contrast carries a binary-arm reduction. `certify_decision` is now a thin wrapper over it and also
  accepts a pre-built `estimate=` (mutually exclusive with raw `outcomes` / `treated`).
- `causalrl.interop.dowhy.from_dowhy_estimate` and `causalrl.interop.econml.from_econml_cate` —
  adapters that pack a fitted DoWhy propensity-based estimate / an EconML CATE-induced policy into a
  `PolicyValueContrast`. New optional `interop` extra (`pip install causalrl[interop]`); the adapters
  are duck-typed and never import the third-party library. Runnable examples under `examples/`.
- `DecisionCertificate.recommendation` — a computed `"act"` / `"abstain"` verdict, so callers read
  the confounding-robustness verdict rather than the (possibly confounded) naive `decision` field.
- An interop guide in the docs and a certificate-headline section in the Tour.

### Scope (unchanged honesty rule)
- The MSM sensitivity is on the logging propensities: sharp when the two target supports are disjoint,
  valid-but-conservative otherwise. It does not certify an arbitrary outcome-model / doubly-robust
  point estimate. The EconML adapter is MSM-only — the CATE-induced policy is per-unit, not a fixed
  binary arm, so the structural pivotality layer does not apply.

## [1.0.0] - 2026-06-08

First stable release. The exported API converged over the 0.x line is now committed to under
semantic versioning: breaking changes to public names move the major version. No behavioural
change to existing kernels — this release adds one ergonomic front door over the decision stack
and promotes the package to Production/Stable.

### Added
- `causalrl.certify_decision` + `DecisionCertificate` — a one-call decision-certificate front
  door. Given confounded / off-policy logs of a binary decision ("is the treated arm better than
  the control arm?"), it composes the documented decision stack — the cheap sign-robustness
  certificate (`pivotality_certificate`) and, when logging propensities are supplied, the
  marginal-sensitivity-model tipping point (`tipping_gamma` over `msm_contribution_bounds`) — into
  a single call with a human-readable verdict. No new theory: an orchestrator over
  `causalrl.identification.bounds`, one-sided by construction (failure to certify is not evidence
  of a flip).

### Changed
- Development status promoted from Beta to Production/Stable.

## [0.99.7] - 2026-06-05

The decision-pivotality layer: certify sign-robustness of a naive contrast under hidden
confounding from the logging agent's information channel — the zeroth (cheapest) layer of the
decision stack, ahead of the MSM bands and `tipping_gamma` abstention. Additive,
backward-compatible public API.

### Added
- `causalrl.confounding_bias_bound` — upper bound on the omitted-variable bias
  `|naive − Z-adjusted|` from logged rows: total-variation form (sharp — attained with equality
  by an explicit two-point family at every parameter value) and the sharp mutual-information
  form `sqrt(MI/2 · (M₁²/p + M₀²/(1−p)))` (optimal KL split across arms; numerically attained
  to 0.999× in its small-MI working range). Strict positivity validation: strata missing an arm
  raise — a logger that conditions hard on the hidden variable destroys overlap, which is
  surfaced, not averaged away.
- `causalrl.mi_flip_threshold` — the decision-pivotality threshold `MI_flip`: the channel
  capacity (nats) below which NO hidden confounder can flip the naive sign. Combined with the
  data-processing inequality `MI(F;Z) ≤ MI(I;Z)`, the *information structure of the
  environment* (who can see what) caps the reachable confounding budget.
- `causalrl.pivotality_certificate` + `PivotalityCertificate` — one-sided sign-robustness
  certificate, two modes: measured-`Z` (post-hoc oracle/showdown data) and structural `mi_cap`
  (certify from the information rules alone, no outcome model, no analyst-chosen sensitivity
  parameter). One-sided by design: failure to certify is not evidence of a flip.
- `causalrl.confounding_bias_per_step_bounds` — sequential per-step credits, each bounded by
  its own per-step information channel; the average-budget (information) sibling of
  `msm_per_step_bounds`' uniform-odds budgets. Carries the growing-channel property: later
  decisions in imperfect-information play are systematically less protected.
- Promoted on the portfolio's promote-on-pass rule after verification against measured ground
  truth in three logged-game regimes (cheating logger / fair self-play / human play): the bound
  held in all three, and the certificate called every regime correctly — including *certifying*
  the fair self-play cell from its measured 7e-4-nat channel and correctly abstaining on the
  human-play cell. Proofs, tightness/converse, and the sequential theorem are documented with
  the research portfolio; the library ships the kernels and their tests.

## [0.99.6] - 2026-06-03

Transportability made operational for policies. Additive, backward-compatible public API.

### Added
- `causalrl.transport_regret_certificate` + `TransportRegretCertificate` — the transport-regret
  certificate: turns the transport engine's decision (`is_transportable_effect` /
  `transport_formula` on a `SelectionDiagram`) into an operational guarantee for a specific
  policy across the domain shift — a computable `Interval(0, span * mu)` upper bound on the
  policy's aggregate transfer regret, the selection-marked witness the policy must not trust
  off-distribution, and a vacuity check. Pure composition of existing engine outputs (no new
  identification theory; inherits `transport.py`'s conservatism).
- `causalrl.decision_flip_rate` — `mu`, the rate at which a policy's *executed trajectory*
  diverges under the do()-sweep on the selection-marked mechanism (the bound's valid scale: a
  single early flip cascades, so the single-decision rate under-counts and must not scale the
  bound).
- `causalrl.decision_abstain_mask` — per-unit abstention rule: fire where the policy's immediate
  greedy decision is not invariant to intervening on the witness.
- Promoted on its pre-registered promote-on-pass rule from the G1 "Spurious CoinRun"
  world-model-transfer arena (bound covered the realized transfer regret 10/10 seeds and stayed
  non-vacuous; abstention recovered the confused set; negative control silent). All exported at
  top level.

## [0.99.5] - 2026-06-03

Causal-sensitivity reporting. Additive, backward-compatible public API.

### Added
- `causalrl.tipping_gamma` — the sensitivity tipping point: the smallest marginal-sensitivity-model
  `gamma >= 1` at which a partial-ID bound first contains a reference value (default 0) — i.e. how
  strong unobserved confounding must be to overturn a sign / decision. The odds-ratio-scale analog
  of the E-value (VanderWeele & Ding, 2017) / Rosenbaum's `Gamma`; takes any `gamma -> Interval`
  callable (e.g. `msm_contribution_bounds`). Exported at top level.
- `examples/sensitivity_bounds.py` — worked example: MSM contribution bounds + `tipping_gamma`.

## [0.99.4] - 2026-06-02

Gymnasium conformance + CGFA-PPO primitive. Additive, backward-compatible public API.

### Added
- `CausalEnvWrapper` (Gymnasium causal wrapper): relaxed to accept envs with `scm=None`
  or without a `reward_node`; construction now succeeds in pass-through mode.  New
  `has_causal_interface` property reports availability.  `reward_parents`, `do`,
  `intervene`, and `set_intervention` raise the new
  `CausalInterfaceUnavailableError` (with an informative message) when the interface is
  disabled, rather than failing at construction.
- **Persistent interventional rollouts** on `CausalEnvWrapper`: `set_intervention(...)`,
  `clear_intervention()`, and the `active_interventions` property.  When an intervention
  is active, `reset` and `step` temporarily swap the wrapped env's live SCM to the
  pre-computed mutilated SCM (via `try/finally` that always restores the original).
- `factored_advantage` + `FactoredAdvantageConfig` (CGFA-PPO causal primitive,
  arXiv:2605.06066): decompose the advantage along the SCM parents of the return node.
  Framework-agnostic pure-NumPy primitive.
- Gymnasium env registration (`register_envs`): calling `import causalrl` now registers
  `causalrl/StructuralCausalBandit-v0` and `causalrl/FrontdoorBandit-v0` in the Gymnasium
  registry, enabling `gymnasium.make("causalrl/StructuralCausalBandit-v0")` and
  `gymnasium.make_vec(...)` vectorisation out of the box.  `register_envs` is exported at
  top level and is idempotent.
- `CausalInterfaceUnavailableError` exported at top level.
- `[examples]` optional extra (`pip install "causalrl[examples]"`): installs
  `stable-baselines3` and `torch` for the CGFA-PPO wiring example.

## [0.99.3] - 2026-05-31

Causal-core API + performance pass (pre-1.0). The partial-identification bounds now return a
tuple-compatible `Interval`; everything else is additive or backward-compatible.

### Added
- `causalrl.scm.ExogenousPosterior` and `StructuralCausalModel.abduct(evidence=None, *, known=None, ...)`:
  Pearl Layer-3 abduction as an explicit step. `abduct(known=...)` pins supplied (continuous /
  known) exogenous **exactly** — no rejection — enabling exact continuous counterfactuals, and
  lets you abduct once then `predict(do=...)` under many interventions.
- `causalrl.identification.bounds.Interval` — a tuple-compatible `NamedTuple(lower, upper)` for
  partial-identification bounds (`.lower` / `.upper`, still unpacks/indexes as a tuple).
- `causalrl.identification.bounds.msm_per_step_bounds` and `msm_stratified_bounds` — reusable
  cumulative-reward marginal-sensitivity-model kernels (additive per-step; weighted per-stratum,
  never wider than pooled — THEORY Prop 1). Exported at top level.
- `causalrl.identification.bounds.msm_policy_value_bounds` — off-policy marginal-sensitivity-model
  bounds on a target policy's value `V(π_t) = E[(π_t/e0) Y]` (self-normalised IPS under Tan's MSM
  on the logging propensity; Kallus–Zhou 2020). The off-policy generalisation of
  `ipw_sensitivity_bounds`, to which it reduces exactly for a constant target. Exported at top level.
- `causalrl.scm.build_unrolled_scm` — build a time-unrolled (sequential) SCM over a fixed horizon
  from a caller-supplied transition, enabling sequential counterfactuals (abduct → do → re-roll the
  trajectory under the same actions). Exported at top level.
- `causalrl.scm.LinearMechanism` — a linear (coefficient + bias) structural mechanism.
- Top-level re-exports from `causalrl`: `Interval`, the MSM kernels (`ipw_sensitivity_bounds`,
  `msm_policy_value_bounds`, `msm_per_step_bounds`, `msm_stratified_bounds`), `ExogenousPosterior`,
  and `build_unrolled_scm` are now importable directly from the package root; `causalrl.scm`
  resolves its torch-backed names lazily.
- `benchmarks/bench_causal_core.py` — guards the exact-counterfactual and closed-form-MSM
  fast paths (correctness to 1e-6 vs a scipy-LP reference; ≥5× speedup floor; observed 874×).

### Changed
- **Breaking:** `causal_q_bounds`, `manski_bounds`, `ipw_sensitivity_bounds` now return `Interval`
  instead of a bare `tuple[float, float]`. Tuple-compatible, so `lo, hi = ...` and `[0]/[1]`
  callers are unaffected; only `isinstance(x, tuple)`-strict or type-annotation-strict code differs.
- `StructuralCausalModel.counterfactual()` is now thin sugar over `abduct()` + `predict()`
  (behavior, signature, and rejection semantics unchanged).
- `do()`, `abduct(known=...)`, `counterfactual()`, and `predict(do=...)` accept **per-sample vector
  values** (leading dimension = unit count) in addition to scalars (scalars still broadcast).
- Intervention / known-noise mappings are typed `Mapping[str, Value]` (covariant) rather than
  `dict[str, Value]`, so `dict[str, float]` callers type-check; runtime behavior is unchanged.

## [0.99.2] - 2026-05-29

Packaging-only release; the library is identical to 0.99.1. The published source distribution
(sdist) now contains **only the library** — the local docs and experiment scripts are
excluded. Supersedes 0.99.0 and 0.99.1, whose sdists bundled those files.

### Changed
- `sdist` scoped to the library (non-library files excluded from
  the package; the wheel was already clean).

## [0.99.1] - 2026-05-29

First release published to PyPI (`pip install causalrl`). No library/API changes from 0.99.0 —
this is a documentation and packaging release.

### Changed
- Documentation overhaul: mkdocs-material theme with a structured navigation
  (Getting Started · Tour by Task · Tutorials · topic pages · API · Citing). The README is now a
  concise front page with a capability table and a "How it compares" section; the per-task tour
  moved into the docs (`docs/tour.md`).
- Release workflow switched to the canonical `pypa/gh-action-pypi-publish` trusted-publishing
  action, with inline setup notes.
- `CITATION.cff` version corrected to match the package.

## [0.99.0] - 2026-05-27

### Stable
- **API stabilized, humbly labelled v0.99** — a deliberate step short of a 1.0 tag while it settles
  in real use. The public API (the names exported from `causalrl`) follows semantic
  versioning. The full Bareinboim 9-task causal-RL taxonomy is implemented, with the depth
  extensions of v0.13–v0.20: the complete Shpitser–Pearl ID algorithm, gID, sID / mz / meta
  transportability, FCI (latent-confounder discovery), mixed Nash for any number of players,
  validated Manski and marginal-sensitivity-model OPE bounds, and a "reproducing the literature"
  gallery.

### Removed
- Deprecated shims: `POMISThompsonSampling` now requires `manipulable=` explicitly (arm-inference
  removed); the deprecated `causalrl.eval.ope.confounding_sensitivity_bounds` bridge is gone (use
  `causalrl.identification.bounds.ipw_sensitivity_bounds`).
- The partial hand-maintained `causalrl/__init__.pyi` stub — `py.typed` plus complete inline
  annotations (checked by pyright in strict mode) are now the authoritative types.

## [0.20.0] - 2026-05-27

### Added
- **"Reproducing the literature" gallery** (`tests/test_literature_classics.py` + a
  [docs gallery](docs/classics.md)): classic causal cases reproduced end-to-end with the library —
  Simpson's paradox (kidney stones), the front-door criterion (smoking → tar → cancer), Pearl's
  napkin, the instrumental variable (non-identified but Manski-bounded), the bow arc, and
  cross-domain transport (LA → NYC).
- **Difficult RL problems where causal beats associational RL**: MABUC (a confounding-aware bandit
  beats the naive one), the counterfactual "Greedy Casino" (acting on the counterfactual ≈ 0.80
  doubles the best fixed interventional arm ≈ 0.37), and curriculum-driven hard exploration (a causal
  prerequisite curriculum reaches a sparse goal flat Q-learning misses on the same budget).

## [0.19.0] - 2026-05-27

### Added
- **Validated partial-identification / OPE bounds** (closing the sensitivity-bounds gap):
  - `manski_bounds` — sharp no-assumptions (Manski 1990) bounds on `E[outcome | do(treatment)]` from
    observational data (the observational counterpart of `causal_q_bounds`).
  - `ipw_sensitivity_bounds` — the marginal sensitivity model (Tan 2006; Zhao–Small–Bhattacharya
    2019): an odds-ratio-Γ interval on the treated counterfactual mean that collapses to the IPW
    point at Γ=1, widens monotonically with Γ, and contains the truth once Γ exceeds the true
    confounding odds ratio. Validated against a confounded SCM with a known effect.

## [0.18.0] - 2026-05-27

### Added
- **Mixed-strategy Nash equilibria for three or more players** (taxonomy Task 9): `mixed_nash_equilibria`
  now handles any number of agents. Two-player games stay exact (rational support enumeration); games
  with ≥3 agents use support enumeration with a numerical Newton solve of the multilinear indifference
  system, and **every returned profile is verified to be an ε-Nash equilibrium**. Validated on a
  three-player cyclic matching game (recovers the uniform `(1/2, 1/2)` equilibrium).

### Changed
- `mixed_nash_equilibria` no longer raises `NotImplementedError` for more than two agents; it raises
  `CausalGraphError` only for fewer than two agents.

## [0.17.0] - 2026-05-27

### Added
- **FCI: causal discovery with latent confounders** (taxonomy Task 5): `discover_latent` learns a
  `PAG` (partial ancestral graph) without assuming causal sufficiency — PC skeleton + Possible-D-SEP
  refinement + the complete orientation rules R1-R10 (Zhang 2008, sound and complete for latent
  confounders and selection bias). `a <-> b` marks a latent confounder; a circle endpoint is
  undetermined by the equivalence class. Validated against the true MAG of the data-generating
  DAG-with-latents (latent-confounder detection and the M-bias collider), with per-rule unit
  fixtures for R1-R10.
- `PAG` (endpoint marks `o` / `>` / `-`, with `is_directed` / `is_bidirected` / `render`) and a
  Causal Discovery guide (`docs/discovery.md`).

### Changed
- Internal: the PC skeleton phase is factored into `_pc_skeleton`, shared by `discover` and
  `discover_latent` (no behaviour change to `discover`).

## [0.16.0] - 2026-05-27

### Added
- **Multi-domain and experimental transportability (mz / meta)** (taxonomy Task 4): a general
  engine resolves each c-factor of the target effect by searching the domains that can supply it.
  - `Domain` describes a source domain — its selection-marked variables and the surrogate
    experiments it offers.
  - `identify_transport_general` / `is_transportable_general` / `estimate_transport_general` decide
    and compute `P*(y | do(x))` across one or more `Domain`s plus the target. With a single
    observational source they coincide with `identify_transport`; with no selection and no
    experiments they reduce to the ID algorithm.
  - **mz**: a surrogate experiment in a source domain supplies a c-factor that no observational
    distribution can (validated: a source `do(X)` breaks a bow-arc hedge, matching simulation).
  - **meta**: invariant c-factors are contributed by different source domains (validated: an effect
    assembled from two sources marked on different covariates matches the target's true `do()`).
- A [Transportability guide](docs/transportability.md) with runnable covariate-shift, mz, and meta
  examples.

### Changed
- The single-source `identify_transport` now delegates to the general engine — behaviour-preserving;
  its signature and results are unchanged.
- Internal: S-node separation helpers moved to `causalrl.identification._separation` (shared by the
  transport code and the ID engine; no public API change).

### Notes
- At c-factor granularity, invariance is exactly "touches no selection-marked variable", so
  single-source observational transport was already complete; this release adds surrogate
  experiments and multiple source domains. The one remaining edge — a single c-factor identifiable
  only by *combining several experiments* — is reported non-transportable rather than guessed.

## [0.15.0] - 2026-05-27

### Added
- **Cross-domain transportability (sID)** (taxonomy Task 4): `identify_transport` /
  `is_transportable_effect` / `estimate_transported_effect` (and the `transport_estimand`
  `SelectionDiagram` adapter) decide and compute the target effect `P*(y | do(x))` across a
  selection diagram by routing each c-factor — invariant factors transfer from the source,
  selection-marked factors are identified from the target. With no selection it reduces to ID.
  Validated by simulation: under a covariate shift the transported estimate matches the target's
  true `do()` distribution (and differs from naively reusing the source). It subsumes the
  direct / S-admissible-adjustment cases; it is sound but not the *complete* sID.

### Fixed
- gID: Tian's `Identify` assumes its domain is a single c-component. Both gID and sID now route
  through `_c_factor_from`, which decomposes the domain into c-components before extracting — fixing
  a latent error on multi-component experiment/source domains (previous gID tests only hit the
  single-component case).

### Notes
- The complete sID (transport c-factors identifiable only by combining source and target) is still
  out of scope and reported as non-transportable rather than guessed.

## [0.14.0] - 2026-05-27

### Added
- **General identification from surrogate experiments (gID)** (taxonomy Task 4):
  `identify_effect_with_experiments` extends the ID recursion so that a c-factor observation cannot
  identify (a hedge) is instead obtained from an available experiment (Tian's `Identify`
  subroutine). `is_gid_identifiable` gives the decision and `estimate_effect_with_experiments`
  evaluates the estimand on observational plus randomized-experimental data. With no experiments it
  coincides exactly with the ID algorithm. Validated by simulation: the bow-arc and a
  confounded-mediator graph (neither observationally identifiable) are recovered from a surrogate
  experiment, matching the true `do()` distribution.

### Notes
- Full cross-domain transportability (the complete sID algorithm) remains out of scope: it reduces
  to a conditional-gID over an augmented selection diagram and is documented as the next frontier.
  Transportability stays at the direct / S-admissible-adjustment slice for now.

## [0.13.0] - 2026-05-27

Depth pass closing the taxonomy gaps surfaced by re-checking the library against the Bareinboim
causal-RL program page.

### Added
- **General causal-effect identification** (taxonomy Task 4): `identify_effect` runs the sound and
  complete Shpitser-Pearl ID algorithm, returning a do-free `Estimand` for `P(y | do(x))` in any
  ADMG or raising `NotIdentifiableError` with the witnessing hedge. `estimate_effect` evaluates the
  estimand on data; `is_identifiable_effect` gives the decision. Validated by simulation:
  back-door/front-door estimands match the true `do()` distribution, and the bow-arc and
  instrumental-variable graphs are correctly non-identifiable.
- **Interventional causal discovery** (Task 5): `discover_interventional` combines observational
  (L1) and experimental (L2) data, orienting edges incident to each intervention target by the
  invariance principle to recover the interventional essential graph.
- **Mixed-strategy Nash equilibria** (Task 9): `mixed_nash_equilibria` finds all equilibria of a
  two-player game exactly by support enumeration over rational arithmetic.
- **Curriculum-driven RL** (Task 7): `curriculum_q_learning` trains Q-learning through a sequence of
  subtasks with warm-start transfer, reaching a sparse target that flat learning misses.
- **"When to intervene"** (Task 2): `requires_experiment` reports when an experiment is necessary —
  exactly when the effect is not observationally identifiable.

### Changed
- `is_identifiable` now delegates to the complete ID algorithm, returning a definite boolean for any
  ADMG (no longer `None` for front-door-style cases).
- Documentation: the API reference and `guarantees.md` now cover the full taxonomy (tasks 3-9).
- Packaging: added `authors`, `keywords`, per-version classifiers, `Typing :: Typed`, and a
  Changelog URL to the project metadata.
- CI: the test matrix now spans Linux/macOS/Windows; added a 90% coverage gate, example-notebook
  execution (`nbmake`), and a `twine check` before publishing.
- Repository hygiene: added `SECURITY.md` and issue/PR templates; stopped tracking a reference PDF
  (`papers/*.pdf` is gitignored).

### Fixed
- `examples/offline_to_online.ipynb`: multi-stage `DOVI` now passes
  `transition_assumption="unconfounded"` (made mandatory by the correctness-hardening pass); the
  notebook previously raised `UnverifiedAssumptionError`.

## [0.12.0] - 2026-05-27

### Added
- **Causal game theory** (taxonomy Task 9, completing the 9-task taxonomy). `CausalGame` represents a
  finite game as a multi-agent causal influence diagram (a decision and a utility node per agent);
  `best_response`, `is_nash_equilibrium`, and `pure_nash_equilibria` reason about equilibria by
  enumeration. The canonical games recover their textbook pure equilibria: Prisoner's Dilemma (mutual
  defection), a coordination game (two equilibria), matching pennies (none). Faithful to Koller &
  Milch (MAIDs, 2003) and Hammond et al., *Reasoning about Causality in Games* (2023).
- `prisoners_dilemma`, `coordination_game`, `matching_pennies` demo games.

## [0.11.0] - 2026-05-27

### Added
- **Causal reward shaping** (taxonomy Task 8). `apply_potential_shaping` adds `gamma*Phi(s') - Phi(s)`
  to an MDP's rewards — policy-invariant for any potential (Ng, Harada & Russell, ICML 1999) — and
  `causal_potential` supplies the ideal potential `V*` from the causal model. With `TabularMDP`,
  `value_iteration`, and tabular `q_learning`, causal-potential shaping makes a sparse reward dense:
  the shaped learner reaches the optimal policy within a few episodes while unshaped Q-learning lags,
  and the optimal policy is provably unchanged.
- `make_sparse_chain_mdp` — the sparse-reward chain demo.

## [0.10.0] - 2026-05-27

### Added
- **Causal curriculum learning** (taxonomy Task 7). `causal_curriculum` orders skills by a
  topological sort of the causal/prerequisite graph (learn causes before effects),
  `is_valid_curriculum` checks an order respects prerequisites, and `PrerequisiteLearner` models
  causally-gated mastery (a skill is learned only once its parents are). On a skill chain/diamond the
  causal curriculum masters the goal while a prerequisite-violating order does not. Faithful to
  Bengio, Louradour, Collobert & Weston, *Curriculum Learning* (ICML 2009).
- `make_skill_chain` / `make_skill_diamond` prerequisite-graph demos.

## [0.9.0] - 2026-05-27

### Added
- **Causal imitation learning** (taxonomy Task 6). `is_imitable` / `imitation_backdoor_set` decide
  whether an expert can be imitated from observed demonstrations and return the observed back-door
  set to clone on; `CausalImitator` clones `P(A | Z)` and reproduces the expert's reward, while the
  `BehavioralCloning` baseline (cloning the marginal `P(A)`) is biased by the confounding.
  Conservative: returns `None` / `False` when no observed admissible set exists rather than a biased
  policy. Faithful to Zhang, Kumor & Bareinboim, *Causal Imitation Learning with Unobserved
  Confounders* (NeurIPS 2020).
- `ImitationEnv` (`make_imitation_diagram`, `generate_demonstrations`, `expert_policy`): a confounded
  one-step demo where the causal imitator matches the expert (~0.9) and naive BC is stuck near ~0.5.
- `is_backdoor_admissible` (the back-door criterion check) is now public, shared by transportability
  and imitation.

## [0.8.0] - 2026-05-27

### Added
- **Causal discovery** (taxonomy Task 5, learning causal models). `discover` runs the PC algorithm
  over discrete data — conditional-independence tests by thresholded conditional mutual information
  (`conditional_mutual_information`), then collider orientation and Meek rules R1–R3 — returning a
  `CPDAG`. `CPDAG.to_causal_graph()` bridges a fully oriented result into the rest of the library, so
  a discovered structure feeds straight into POMIS planning. Faithful to Spirtes, Glymour & Scheines
  (PC) and Meek (UAI 1995). Conservative: assumes causal sufficiency, and the bridge raises rather
  than guessing an orientation.
- `build_discovery_scm` / `sample_discovery_data` — a collider demo (`X→Z←Y`, `Z→W`) whose CPDAG is
  recovered from data and then handed to `pomis`.

## [0.7.0] - 2026-05-27

### Added
- **Transportability** (taxonomy Task 4, generalizability & robustness). `SelectionDiagram`
  represents a source/target pair differing in some mechanisms; `transport_formula` /
  `is_transportable` decide whether `P*(y | do(x))` transfers and return the transport formula
  (direct or S-admissible adjustment); `transported_effect` computes the transported estimate by
  reweighting source conditionals with the target covariate marginal. Conservative — returns `None`
  outside the supported class (no hedge-based sID completeness check). Faithful to Bareinboim &
  Pearl (AAAI 2012; J. Causal Inference 2013) and Pearl & Bareinboim (Statistical Science 2014).
- `make_transport_domains` — the canonical covariate-shift demo (`Z→X, Z→Y, X→Y`, selection on
  `Z`): the transport formula recovers the true target effect (~0.82) while naively reusing the
  source effect is biased (~0.58).
- `CausalGraph.directed_edges` and `CausalGraph.bidirected_edges` accessors.

## [0.6.0] - 2026-05-27

### Added
- **Counterfactual decision-making** (taxonomy Task 3, Layer 3). `counterfactual_expectation`
  returns `E[Y_{do(x)} | evidence]` and `effect_of_treatment_on_treated` returns the ETT
  `E[Y_{treated} − Y_{control} | X = treated]`, both computed on an executable
  `StructuralCausalModel` via abduction-action-prediction (`causalrl.identification.counterfactual`).
  Faithful to Bareinboim, Forney & Pearl, *Bandits with Unobserved Confounders: A Causal Approach*
  (NeurIPS 2015) and Pearl, *Causality* (2nd ed.) §8.2.1.
- `CounterfactualOptimalPolicy` — a model-based Regret Decision Criterion policy that decides by
  `argmax_a E[Y_{do(a)} | intent]` — and `CounterfactualBanditEnv` /
  `make_counterfactual_bandit_env`, a 3-arm confounded bandit. The counterfactual-optimal policy and
  a trained `CausalThompsonSampling` reach the per-intent optimum (~0.8); a confounding-naive agent
  is stuck at the best fixed intervention (the `do`-optimum, ~0.367).

### Changed
- Public environments now satisfy Gymnasium's checker, rollout helpers handle truncation, SCM
  sampling isolates Torch RNG state, executable SCM definitions validate their graph contract,
  and structural-bandit Thompson sampling supports bounded fractional rewards correctly.
- Multi-stage `DOVI` exposes whether transition-value propagation is causally certified, and
  `POMISThompsonSampling` accepts an explicit validated `manipulable` contract.
- Deterministic multi-seed benchmark reporting, API documentation, citation/contribution
  metadata, documentation CI, and release-only trusted PyPI publishing scaffolding were added.

## [0.5.0] - 2026-05-26

### Added
- **Non-manipulable variables** (extends taxonomy Task 2): `pomis` and
  `minimal_intervention_sets` accept an optional `manipulable` subset. With non-manipulable set
  `N`, POMIS equals the unconstrained POMIS of the latent projection onto `V\N` (Lee &
  Bareinboim, *Structural Causal Bandits with Non-Manipulable Variables*, AAAI 2019, R-40,
  Theorem 4); MIS simply filters to sets disjoint from `N`.
- `CausalGraph.latent_projection(keep)` — the Tian-Pearl / Verma latent projection.
- `make_frontdoor_env` (the R-40 front-door / cholesterol demo, with `Z` non-manipulable) and
  `NaivePOMISThompsonSampling`; the manipulability-aware `POMISThompsonSampling` reaches the
  `do(X)` optimum (~0.56) that the naive filter baseline (stuck at ~0.50 observation) cannot see.

### Changed
- `POMISThompsonSampling` infers its manipulable set from the environment's arms, so it
  respects non-manipulable variables automatically (identical behavior when all variables are
  manipulable).
- Stable causal-method contracts are now conservative: `StructuralCausalModel` executes
  explicit-latent DAGs only, `backdoor_adjustment_set` refuses latent-confounded treatments,
  and `is_identifiable` returns unknown for unsupported ADMG cases rather than an optimistic
  positive result.
- The qualitative `confounding_sensitivity_bounds` helper moved to
  `causalrl.experimental.ope`; its previous module path remains as a deprecated bridge and it
  is no longer part of the stable top-level exports.
- Stable public exports are loaded lazily. Core graph, POMIS, tabular-agent, and tabular-env
  use no longer requires PyTorch; install the `torch` extra for SCM/neural/Torch-backed
  components. Supported Python now begins at 3.11 and CI covers 3.11 and 3.14.

## [0.4.0] - 2026-05-26

### Added
- **POMIS engine** (taxonomy Task 2, "where to intervene"): `pomis` and
  `minimal_intervention_sets` compute the possibly-optimal / minimal intervention sets of a
  single-reward ADMG via MUCT (minimal unobserved-confounder territory) and the
  interventional border. Adapted from the MIT-licensed reference implementation of
  Lee & Bareinboim, *Structural Causal Bandits: Where to Intervene?* (NeurIPS 2018),
  github.com/sanghack81/SCMMAB-NIPS2018 (Copyright (c) 2018 Sanghack Lee).
- `StructuralCausalBanditEnv` — an SCM-backed bandit whose arms are interventions, plus the
  `make_confounded_chain_env` demo where observing beats every fixed intervention.
- `POMISThompsonSampling`, `BruteForceInterventionTS`, and `FixedSetThompsonSampling` agents;
  the POMIS agent converges to the optimal arm far faster than brute force and beats a naive
  fixed-set agent.
- `CausalGraph` gains `ancestors`, `descendants`, `induced_subgraph`, and `do_mutilate`.

## [0.3.0] - 2026-05-26

### Added
- Horizon-indexed **DOVI** (Deconfounded Optimistic Value Iteration): finite-horizon backward
  induction with Manski-bound-capped rewards; reduces exactly to the v0.2 backup at `H=1`.
- `SequentialDTREnv` — a genuinely confounded, multi-stage dynamic-treatment-regime environment
  with a foresight gap (the immediate-greedy and lookahead-optimal first actions diverge).
- Curated top-level public API with an explicit `__all__`:
  `from causalrl import DOVI, StructuralCausalModel, DTREnv, generate_logs, ...`.
- `py.typed` marker (PEP 561) so downstream consumers receive our type information.
- `LICENSE` file (MIT) and this changelog.

### Changed
- `SequentialMABUCEnv` rewritten to be genuinely confounded: a hidden confounder drives both
  logging and reward, so the naive per-context mean is biased above the true interventional value.
- Sharpened the Manski-bounds property tests (higher sample size, tighter propensity, smaller slack).
- Version is single-sourced from `pyproject.toml` and read at runtime via `importlib.metadata`.

### Fixed
- The offline-to-online example re-imported names already imported in an earlier cell, tripping
  ruff's `F811` and failing the CI lint step; the redundant imports are removed (the notebook
  runs identically). `ruff check .` is now clean across the whole repository, including notebooks.

## [0.2.0] - 2026-05-23

### Added
- Causal **offline-to-online** learning (taxonomy Task 1): `UCDTR`, `DOVI`, and
  `DeepDeconfoundedQ` agents that read confounded logs through causal bounds.
- Confounded environments: `DTREnv`, `ConfoundedGridworld`, `SequentialMABUCEnv`.
- Manski natural bounds (`causal_q_bounds`) and the offline-to-online evaluation harness.

## [0.1.0] - 2026-05-23

### Added
- Structural causal model core: `CausalGraph`, mechanisms, and `StructuralCausalModel` with
  `see` (L1), `do` (L2), and `counterfactual` (L3) queries.
- Scoped identification: back-door parent set and bow-arc detection.
- MABUC bandit slice with causal Thompson sampling that beats a confounding-naive baseline.
