# API Reference

## Graphs

::: causalrl.scm.graph.CausalGraph

## Intervention Sets

::: causalrl.identification.intervention_sets.pomis

::: causalrl.identification.intervention_sets.minimal_intervention_sets

::: causalrl.identification.intervention_sets.requires_experiment

::: causalrl.identification.intervention_sets.AdmissibleInterventions

## Structural Causal Models

::: causalrl.scm.scm.StructuralCausalModel

## Assumption-Aware Agents

::: causalrl.agents.dovi.DOVI

::: causalrl.agents.scbandit.POMISThompsonSampling

## Benchmark Reports

::: causalrl.eval.benchmark.BenchmarkEstimate

::: causalrl.eval.benchmark.run_confounded_chain_benchmark

::: causalrl.eval.benchmark.run_frontdoor_benchmark

## Counterfactual Decision-Making (L3 / ETT)

::: causalrl.identification.counterfactual.counterfactual_expectation

::: causalrl.identification.counterfactual.effect_of_treatment_on_treated

::: causalrl.agents.counterfactual.CounterfactualOptimalPolicy

## Transportability

::: causalrl.identification.transport.SelectionDiagram

::: causalrl.identification.transport.transport_formula

::: causalrl.identification.transport.is_transportable

::: causalrl.identification.transport.transported_effect

::: causalrl.identification.transport.is_backdoor_admissible

### General Transportability (sID)

::: causalrl.identification.transport.transport_estimand

::: causalrl.identification.id_algorithm.identify_transport

::: causalrl.identification.id_algorithm.is_transportable_effect

::: causalrl.identification.id_algorithm.estimate_transported_effect

### Multiple Domains And Experiments (mz / meta)

::: causalrl.identification.id_algorithm.Domain

::: causalrl.identification.id_algorithm.identify_transport_general

::: causalrl.identification.id_algorithm.is_transportable_general

::: causalrl.identification.id_algorithm.estimate_transport_general

## General Identification (ID Algorithm)

::: causalrl.identification.id_algorithm.identify_effect

::: causalrl.identification.id_algorithm.is_identifiable_effect

::: causalrl.identification.id_algorithm.estimate_effect

::: causalrl.identification.id_algorithm.Estimand

### From Surrogate Experiments (gID)

::: causalrl.identification.id_algorithm.identify_effect_with_experiments

::: causalrl.identification.id_algorithm.is_gid_identifiable

::: causalrl.identification.id_algorithm.estimate_effect_with_experiments

## Causal Discovery

::: causalrl.discovery.discover

::: causalrl.discovery.discover_interventional

::: causalrl.discovery.discover_latent

::: causalrl.discovery.CPDAG

::: causalrl.discovery.PAG

::: causalrl.discovery.conditional_mutual_information

## Causal Imitation Learning

::: causalrl.imitation.is_imitable

::: causalrl.imitation.imitation_backdoor_set

::: causalrl.imitation.CausalImitator

::: causalrl.imitation.BehavioralCloning

## Causal Curriculum Learning

::: causalrl.curriculum.causal_curriculum

::: causalrl.curriculum.curriculum_q_learning

::: causalrl.curriculum.is_valid_curriculum

::: causalrl.curriculum.PrerequisiteLearner

## Causal Reward Shaping

::: causalrl.shaping.apply_potential_shaping

::: causalrl.shaping.causal_potential

::: causalrl.shaping.value_iteration

::: causalrl.shaping.q_learning

::: causalrl.shaping.TabularMDP

## Causal Game Theory

::: causalrl.games.CausalGame

::: causalrl.games.pure_nash_equilibria

::: causalrl.games.mixed_nash_equilibria

::: causalrl.games.best_response

::: causalrl.games.is_nash_equilibrium

## Causal Gymnasium Wrapper

::: causalrl.envs.wrapper.CausalEnvWrapper

## Causal Graph-Factored Advantage (CGFA)

::: causalrl.agents.factored_advantage.factored_advantage

::: causalrl.agents.factored_advantage.FactoredAdvantageConfig

## Gymnasium Env Registration

::: causalrl.envs.registration.register_envs

## Exceptions

::: causalrl.exceptions.CausalRLError

::: causalrl.exceptions.CausalInterfaceUnavailableError

::: causalrl.exceptions.NotIdentifiableError

::: causalrl.exceptions.CausalGraphError

::: causalrl.exceptions.RealizabilityError

::: causalrl.exceptions.UnverifiedAssumptionError

## Partial-Identification And OPE Bounds

::: causalrl.identification.bounds.manski_bounds

::: causalrl.identification.bounds.ipw_sensitivity_bounds

::: causalrl.identification.bounds.causal_q_bounds

::: causalrl.identification.bounds.msm_policy_value_bounds

::: causalrl.identification.bounds.msm_contribution_bounds

::: causalrl.identification.bounds.msm_per_step_bounds

::: causalrl.identification.bounds.msm_stratified_bounds

## Decision Certificates

The decision stack — certify whether a confounded / off-policy decision ("is the treated arm
better than the control arm?") is robust to hidden confounding, cheapest layer first.
`certify_decision` is the one-call front door over the layers below.

::: causalrl.identification.decision.certify_decision

::: causalrl.identification.decision.DecisionCertificate

::: causalrl.identification.bounds.pivotality_certificate

::: causalrl.identification.bounds.confounding_bias_bound

::: causalrl.identification.bounds.mi_flip_threshold

::: causalrl.identification.bounds.tipping_gamma

## Unified Certificates (§5.2)

::: causalrl.certify.certificate.Certificate

::: causalrl.certify.adapters.as_certificate

## Identification-Aware Estimation (Phase 1)

::: causalrl.estimate.compiler.certify_effect

::: causalrl.bounds.continuous.msm_sensitivity_bounds

::: causalrl.bounds.continuous.certify_mean

::: causalrl.transport.estimate.certify_transported_effect

## Multi-Agent Core (Phase 2)

::: causalrl.magames.population.Population

::: causalrl.magames.equilibrium.certify_equilibrium

::: causalrl.magames.views.agent_causal_env_view

## Scale & Streaming (Phase 3)

::: causalrl.estimate.streaming.stream_policy_value

::: causalrl.estimate.streaming.stream_quantile_certificate

::: causalrl.bounds.streaming.stream_msm_bounds

::: causalrl.backends.streaming.StreamingMoments

::: causalrl.backends.streaming.WeightedStreamingRatio

::: causalrl.backends.quantile_sketch.GKQuantileSketch

## Interop & Scale (Phase 4)

::: causalrl.interop.sbi_numpyro.regimes_from_posterior

::: causalrl.interop.sbi_numpyro.across_regimes

::: causalrl.interop.columnar_sim.ColumnarSimulator

::: causalrl.interop.columnar_sim.simulator_from_callables

::: causalrl.scale.certify_policy

::: causalrl.scale.d3rlpy.certify_fqe

::: causalrl.scale.d3rlpy.policy_actions

## Set-Valued Interventions

::: causalrl.intervention.InterventionSpace

::: causalrl.intervention.canonical

::: causalrl.agents.interventional.InterventionalAgent

::: causalrl.agents.interventional.ScalarAgentAdapter

::: causalrl.deadline.Deadline

## Interference (Spillovers)

::: causalrl.interference.ExposureMapping

::: causalrl.interference.ExposureContrast

::: causalrl.interference.neighbourhood_count

::: causalrl.interference.neighbourhood_fraction

::: causalrl.interference.any_neighbour_treated

::: causalrl.interference.population_share

::: causalrl.interference.adjacency_from_matrix

::: causalrl.interference.direct_effect

::: causalrl.interference.spillover_effect

::: causalrl.interference.total_effect

## Known Mechanisms

::: causalrl.scm.fitters.PinnedMechanism

## Continuous States

::: causalrl.state.StateEncoder

::: causalrl.state.OneHotEncoder

::: causalrl.state.IdentityEncoder

::: causalrl.state.RBFEncoder

::: causalrl.state.FeatureTransition

::: causalrl.state.encode_batch

::: causalrl.agents.fitted.FittedQIteration

## Function-Valued Causal Bounds

::: causalrl.bounds.functional.FunctionalManskiBounds

::: causalrl.bounds.functional.OverlapDiagnostic

::: causalrl.agents.bounded_fitted.BoundedFittedQIteration
