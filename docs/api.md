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

## Causal Graph-Factored Advantage (CGFA-PPO)

The pure-NumPy rollout arithmetic (no framework dependency):

::: causalrl.agents.factored_advantage.factor_rewards

::: causalrl.agents.factored_advantage.factor_gae

::: causalrl.agents.factored_advantage.blend_advantages

::: causalrl.agents.factored_advantage.factored_advantage

::: causalrl.agents.factored_advantage.FactoredAdvantageConfig

The `K`-head critic that makes it an algorithm (needs the `causalrl[torch]` extra):

::: causalrl.agents.cgfa_critic.FactoredCritic

::: causalrl.agents.cgfa_critic.CGFACriticConfig

::: causalrl.agents.cgfa_critic.CGFAAdvantages

::: causalrl.agents.cgfa_critic.CGFALosses

::: causalrl.agents.cgfa_critic.CGFAUpdateStats

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

::: causalrl.conformal.core.conformal_action_value

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

::: causalrl.magames.learning.run_no_regret

::: causalrl.agents.no_regret.NoRegretLearner

::: causalrl.magames.equilibrium.certify_equilibrium

::: causalrl.magames.views.linear_gaussian_population_env

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


<!-- The sections below complete the reference: every name in `causalrl.__all__` has an
     entry, grouped by `causalrl.API_TIERS`. `test_every_export_appears_in_the_api_reference`
     fails if a new export is added without one. -->


## Core — Complete Reference

::: causalrl.agents.causal_mbrl.CausalMBRLAgent

::: causalrl.agents.bandits.CausalThompsonSampling

::: causalrl.scm.fit.fit_scm


## Identification, Transport & Discovery — Complete Reference

::: causalrl.identification.counterfactual_bounds.CounterfactualBound

::: causalrl.identification.bounds.Interval

::: causalrl.identification.bounds.PivotalityCertificate

::: causalrl.identification.estimate.PolicyValueContrast

::: causalrl.identification.transport.TransportFormula

::: causalrl.identification.transport_regret.TransportRegretCertificate

::: causalrl.identification.criteria.backdoor_adjustment_set

::: causalrl.identification.decision.certify_estimate

::: causalrl.transport.estimate.certify_sequential_transport

::: causalrl.identification.bounds.confounding_bias_per_step_bounds

::: causalrl.identification.counterfactual_bounds.counterfactual_interval

::: causalrl.identification.transport_regret.decision_abstain_mask

::: causalrl.identification.transport_regret.decision_flip_rate

::: causalrl.identification.criteria.is_identifiable

::: causalrl.discovery.orient

::: causalrl.identification.transport_regret.transport_regret_certificate


## Structural Models & Data — Complete Reference

::: causalrl.scm.fitters.ANMFit

::: causalrl.scm.continuous.abduction.AmortizedGaussianAbduction

::: causalrl.scm.continuous.bayesian_fit.BayesianLinearFit

::: causalrl.protocols.CausalEnvProtocol

::: causalrl.scm.continuous.mechanisms.ConditionalFlowMechanism

::: causalrl.data.dataset.ConfoundedTrajectoryDataset

::: causalrl.protocols.DictNoiseLedger

::: causalrl.scm.scm.ExogenousPosterior

::: causalrl.scm.fit.FitReport

::: causalrl.scm.mechanisms.FunctionalMechanism

::: causalrl.scm.fitters.LinearGaussianFit

::: causalrl.scm.mechanisms.LinearGaussianMechanism

::: causalrl.scm.continuous.mechanisms.LocationScaleMechanism

::: causalrl.scm.continuous.mechanisms.MLPMechanism

::: causalrl.scm.mechanisms.Mechanism

::: causalrl.scm.continuous.nuts.NUTSNoisePosterior

::: causalrl.scm.fitters.NeuralFit

::: causalrl.scm.mechanisms.NeuralMechanism

::: causalrl.scm.fit.NodeFit

::: causalrl.protocols.NoiseLedger

::: causalrl.protocols.NoisePosterior

::: causalrl.scm.fitters.PoissonGLMFit

::: causalrl.regime.Regime

::: causalrl.protocols.SCMCausalEnv

::: causalrl.scm.fitters.TabularCPT

::: causalrl.data.trajectory.TrajectoryLog

::: causalrl.data.dataset.Transition

::: causalrl.scm.continuous.abduction.abduct_invertible

::: causalrl.scm.continuous.abduction.abduct_location_scale

::: causalrl.scm.continuous.nuts.abduct_nuts

::: causalrl.scm.unrolled.build_unrolled_scm

::: causalrl.conformal.core.certify_conformal_interval

::: causalrl.scm.continuous.abduction.certify_counterfactual

::: causalrl.scm.continuous.nuts.certify_nuts_counterfactual

::: causalrl.conformal.core.conformal_quantile

::: causalrl.conformal.core.cqr_interval

::: causalrl.scm.fit.fit_scm_mec

::: causalrl.data.dataset.generate_logs

::: causalrl.conformal.core.split_conformal_interval


## Agents, Environments & Interventions — Complete Reference

::: causalrl.agents.base.Agent

::: causalrl.magames.AgentType

::: causalrl.agents.mbrl.BackdoorAdjustedAgent

::: causalrl.agents.scbandit.BruteForceInterventionTS

::: causalrl.magames.CCEPolytope

::: causalrl.envs.base.CausalEnv

::: causalrl.agents.mbrl.CertifiedPolicyAgent

::: causalrl.envs.suite.confounded_context.ConfoundedContextualBandit

::: causalrl.envs.suite.gridworld.ConfoundedGridworld

::: causalrl.envs.base.ConfoundedMDP

::: causalrl.envs.suite.continuous_confounded.ContinuousConfoundedBandit

::: causalrl.envs.suite.counterfactual_bandit.CounterfactualBanditEnv

::: causalrl.envs.suite.dtr.DTREnv

::: causalrl.agents.deep_deconfounded.DeepDeconfoundedQ

::: causalrl.agents.mbrl.DiscoveryBackdoorAgent

::: causalrl.agents.scbandit.FixedSetThompsonSampling

::: causalrl.agents.mbrl.FunctionApproxBackdoorAgent

::: causalrl.agents.mbrl.GFormulaBackdoorAgent

::: causalrl.intervention.Intervention

::: causalrl.magames.KindNotLicensedError

::: causalrl.magames.LearnerTopology

::: causalrl.envs.suite.mabuc.MABUCEnv

::: causalrl.agents.baselines.NaiveOffline

::: causalrl.agents.scbandit.NaivePOMISThompsonSampling

::: causalrl.agents.bandits.NaiveThompsonSampling

::: causalrl.agents.online_causal_mbrl.OnlineCausalMBRL

::: causalrl.agents.baselines.OnlineOnlyUCB

::: causalrl.magames.PopulationAgentView

::: causalrl.envs.suite.seq_dtr.SequentialDTREnv

::: causalrl.envs.suite.seq_mabuc.SequentialMABUCEnv

::: causalrl.envs.suite.simpson_bandit.SimpsonBandit

::: causalrl.envs.suite.scbandit.StructuralCausalBanditEnv

::: causalrl.agents.mbrl.TransportBackdoorAgent

::: causalrl.envs.suite.transport_bandit.TransportableConfoundedBandit

::: causalrl.agents.offline_online.UCDTR

::: causalrl.magames.cce_bounds

::: causalrl.magames.cce_polytope

::: causalrl.magames.cce_regret

::: causalrl.magames.certify_cce_do

::: causalrl.magames.topology_max_kind


## Estimation, Bounds & Certificates — Complete Reference

::: causalrl.certify.Assumption

::: causalrl.estimate.estimators.EffectEstimate

::: causalrl.certify.EstimandSpec

::: causalrl.certify.Hedge

::: causalrl.certify.Kind

::: causalrl.certify.Provenance

::: causalrl.estimate.sequential.SequentialValueEstimate

::: causalrl.certify.Witness

::: causalrl.bounds.continuous.certify_quantile

::: causalrl.bounds.continuous.certify_sensitivity_bounds

::: causalrl.estimate.sequential.certify_sequential_value

::: causalrl.eval.metrics.cumulative_regret

::: causalrl.estimate.estimators.estimate_ate

::: causalrl.estimate.sequential.estimate_sequential_value

::: causalrl.eval.metrics.finite_horizon_regret

::: causalrl.certify.identify_effect_certified

::: causalrl.certify.ipw_sensitivity_bounds_certified

::: causalrl.eval.ope.ipw_value

::: causalrl.bounds.continuous.moment_diagnostic

::: causalrl.certify.msm_policy_value_bounds_certified

::: causalrl.eval.benchmark.report_to_dict

::: causalrl.eval.harness.run_episodes

::: causalrl.eval.mbrl_probe.run_m0_kill_gate

::: causalrl.eval.mbrl_probe.run_m1_discovery_gate

::: causalrl.eval.mbrl_probe.run_m1b_dtr_gate

::: causalrl.eval.mbrl_probe.run_m2_phase_diagram

::: causalrl.eval.mbrl_probe.run_m3_function_approx_gate

::: causalrl.estimate.sequential.sequential_ice_values


## Interop, Scale & Errors — Complete Reference

::: causalrl.interop.sbi_numpyro.PosteriorRegimeSampler

::: causalrl.interop.columnar_sim.check_conformance

::: causalrl.interop.causal_gym.from_causal_gym

::: causalrl.interop.pettingzoo.pettingzoo_to_trajectory_log

