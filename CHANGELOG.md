# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
