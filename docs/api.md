# API Reference

## Graphs

::: causalrl.scm.graph.CausalGraph

## Intervention Sets

::: causalrl.identification.intervention_sets.pomis

::: causalrl.identification.intervention_sets.minimal_intervention_sets

::: causalrl.identification.intervention_sets.requires_experiment

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
