# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
