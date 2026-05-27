# pyright: reportUnsupportedDunderAll=false
"""causalrl: causal intervention-selection and causal-RL research tools.

The stable public API is loaded lazily so graph algorithms and tabular components can be used
without installing the optional PyTorch-backed SCM and neural functionality.
"""

from importlib import import_module as _import_module
from importlib.metadata import version as _pkg_version
from typing import cast as _cast

__version__ = _pkg_version("causalrl")

_EXPORTS: dict[str, tuple[str, str]] = {
    "DOVI": ("causalrl.agents.dovi", "DOVI"),
    "UCDTR": ("causalrl.agents.offline_online", "UCDTR"),
    "Agent": ("causalrl.agents.base", "Agent"),
    "BruteForceInterventionTS": ("causalrl.agents.scbandit", "BruteForceInterventionTS"),
    "BehavioralCloning": ("causalrl.imitation", "BehavioralCloning"),
    "BenchmarkEstimate": ("causalrl.eval.benchmark", "BenchmarkEstimate"),
    "CPDAG": ("causalrl.discovery", "CPDAG"),
    "CausalEnv": ("causalrl.envs.base", "CausalEnv"),
    "CausalGame": ("causalrl.games", "CausalGame"),
    "CausalGraph": ("causalrl.scm.graph", "CausalGraph"),
    "CausalGraphError": ("causalrl.exceptions", "CausalGraphError"),
    "CausalImitator": ("causalrl.imitation", "CausalImitator"),
    "CausalRLError": ("causalrl.exceptions", "CausalRLError"),
    "CausalThompsonSampling": ("causalrl.agents.bandits", "CausalThompsonSampling"),
    "ConfoundedGridworld": ("causalrl.envs.suite.gridworld", "ConfoundedGridworld"),
    "ConfoundedMDP": ("causalrl.envs.base", "ConfoundedMDP"),
    "ConfoundedTrajectoryDataset": ("causalrl.data.dataset", "ConfoundedTrajectoryDataset"),
    "CounterfactualBanditEnv": (
        "causalrl.envs.suite.counterfactual_bandit",
        "CounterfactualBanditEnv",
    ),
    "CounterfactualOptimalPolicy": (
        "causalrl.agents.counterfactual",
        "CounterfactualOptimalPolicy",
    ),
    "DTREnv": ("causalrl.envs.suite.dtr", "DTREnv"),
    "DeepDeconfoundedQ": ("causalrl.agents.deep_deconfounded", "DeepDeconfoundedQ"),
    "FixedSetThompsonSampling": ("causalrl.agents.scbandit", "FixedSetThompsonSampling"),
    "FunctionalMechanism": ("causalrl.scm.mechanisms", "FunctionalMechanism"),
    "LinearGaussianMechanism": ("causalrl.scm.mechanisms", "LinearGaussianMechanism"),
    "MABUCEnv": ("causalrl.envs.suite.mabuc", "MABUCEnv"),
    "Mechanism": ("causalrl.scm.mechanisms", "Mechanism"),
    "NaiveOffline": ("causalrl.agents.baselines", "NaiveOffline"),
    "NaivePOMISThompsonSampling": ("causalrl.agents.scbandit", "NaivePOMISThompsonSampling"),
    "NaiveThompsonSampling": ("causalrl.agents.bandits", "NaiveThompsonSampling"),
    "NeuralMechanism": ("causalrl.scm.mechanisms", "NeuralMechanism"),
    "NotIdentifiableError": ("causalrl.exceptions", "NotIdentifiableError"),
    "OnlineOnlyUCB": ("causalrl.agents.baselines", "OnlineOnlyUCB"),
    "POMISThompsonSampling": ("causalrl.agents.scbandit", "POMISThompsonSampling"),
    "PrerequisiteLearner": ("causalrl.curriculum", "PrerequisiteLearner"),
    "RealizabilityError": ("causalrl.exceptions", "RealizabilityError"),
    "UnverifiedAssumptionError": ("causalrl.exceptions", "UnverifiedAssumptionError"),
    "SelectionDiagram": ("causalrl.identification.transport", "SelectionDiagram"),
    "SequentialDTREnv": ("causalrl.envs.suite.seq_dtr", "SequentialDTREnv"),
    "SequentialMABUCEnv": ("causalrl.envs.suite.seq_mabuc", "SequentialMABUCEnv"),
    "StructuralCausalBanditEnv": ("causalrl.envs.suite.scbandit", "StructuralCausalBanditEnv"),
    "StructuralCausalModel": ("causalrl.scm.scm", "StructuralCausalModel"),
    "TabularMDP": ("causalrl.shaping", "TabularMDP"),
    "Transition": ("causalrl.data.dataset", "Transition"),
    "TransportFormula": ("causalrl.identification.transport", "TransportFormula"),
    "apply_potential_shaping": ("causalrl.shaping", "apply_potential_shaping"),
    "backdoor_adjustment_set": ("causalrl.identification.criteria", "backdoor_adjustment_set"),
    "best_response": ("causalrl.games", "best_response"),
    "causal_curriculum": ("causalrl.curriculum", "causal_curriculum"),
    "causal_potential": ("causalrl.shaping", "causal_potential"),
    "causal_q_bounds": ("causalrl.identification.bounds", "causal_q_bounds"),
    "conditional_mutual_information": ("causalrl.discovery", "conditional_mutual_information"),
    "counterfactual_expectation": (
        "causalrl.identification.counterfactual",
        "counterfactual_expectation",
    ),
    "cumulative_regret": ("causalrl.eval.metrics", "cumulative_regret"),
    "discover": ("causalrl.discovery", "discover"),
    "discover_interventional": ("causalrl.discovery", "discover_interventional"),
    "effect_of_treatment_on_treated": (
        "causalrl.identification.counterfactual",
        "effect_of_treatment_on_treated",
    ),
    "finite_horizon_regret": ("causalrl.eval.metrics", "finite_horizon_regret"),
    "generate_logs": ("causalrl.data.dataset", "generate_logs"),
    "imitation_backdoor_set": ("causalrl.imitation", "imitation_backdoor_set"),
    "ipw_value": ("causalrl.eval.ope", "ipw_value"),
    "is_backdoor_admissible": ("causalrl.identification.transport", "is_backdoor_admissible"),
    "is_identifiable": ("causalrl.identification.criteria", "is_identifiable"),
    "is_imitable": ("causalrl.imitation", "is_imitable"),
    "is_nash_equilibrium": ("causalrl.games", "is_nash_equilibrium"),
    "is_transportable": ("causalrl.identification.transport", "is_transportable"),
    "is_valid_curriculum": ("causalrl.curriculum", "is_valid_curriculum"),
    "minimal_intervention_sets": (
        "causalrl.identification.intervention_sets",
        "minimal_intervention_sets",
    ),
    "mixed_nash_equilibria": ("causalrl.games", "mixed_nash_equilibria"),
    "pomis": ("causalrl.identification.intervention_sets", "pomis"),
    "pure_nash_equilibria": ("causalrl.games", "pure_nash_equilibria"),
    "q_learning": ("causalrl.shaping", "q_learning"),
    "report_to_dict": ("causalrl.eval.benchmark", "report_to_dict"),
    "run_episodes": ("causalrl.eval.harness", "run_episodes"),
    "run_confounded_chain_benchmark": ("causalrl.eval.benchmark", "run_confounded_chain_benchmark"),
    "run_frontdoor_benchmark": ("causalrl.eval.benchmark", "run_frontdoor_benchmark"),
    "transport_formula": ("causalrl.identification.transport", "transport_formula"),
    "transported_effect": ("causalrl.identification.transport", "transported_effect"),
    "value_iteration": ("causalrl.shaping", "value_iteration"),
}

__all__ = [
    "CPDAG",
    "DOVI",
    "UCDTR",
    "Agent",
    "BehavioralCloning",
    "BenchmarkEstimate",
    "BruteForceInterventionTS",
    "CausalEnv",
    "CausalGame",
    "CausalGraph",
    "CausalGraphError",
    "CausalImitator",
    "CausalRLError",
    "CausalThompsonSampling",
    "ConfoundedGridworld",
    "ConfoundedMDP",
    "ConfoundedTrajectoryDataset",
    "CounterfactualBanditEnv",
    "CounterfactualOptimalPolicy",
    "DTREnv",
    "DeepDeconfoundedQ",
    "FixedSetThompsonSampling",
    "FunctionalMechanism",
    "LinearGaussianMechanism",
    "MABUCEnv",
    "Mechanism",
    "NaiveOffline",
    "NaivePOMISThompsonSampling",
    "NaiveThompsonSampling",
    "NeuralMechanism",
    "NotIdentifiableError",
    "OnlineOnlyUCB",
    "POMISThompsonSampling",
    "PrerequisiteLearner",
    "RealizabilityError",
    "SelectionDiagram",
    "SequentialDTREnv",
    "SequentialMABUCEnv",
    "StructuralCausalBanditEnv",
    "StructuralCausalModel",
    "TabularMDP",
    "Transition",
    "TransportFormula",
    "UnverifiedAssumptionError",
    "__version__",
    "apply_potential_shaping",
    "backdoor_adjustment_set",
    "best_response",
    "causal_curriculum",
    "causal_potential",
    "causal_q_bounds",
    "conditional_mutual_information",
    "counterfactual_expectation",
    "cumulative_regret",
    "discover",
    "discover_interventional",
    "effect_of_treatment_on_treated",
    "finite_horizon_regret",
    "generate_logs",
    "imitation_backdoor_set",
    "ipw_value",
    "is_backdoor_admissible",
    "is_identifiable",
    "is_imitable",
    "is_nash_equilibrium",
    "is_transportable",
    "is_valid_curriculum",
    "minimal_intervention_sets",
    "mixed_nash_equilibria",
    "pomis",
    "pure_nash_equilibria",
    "q_learning",
    "report_to_dict",
    "run_confounded_chain_benchmark",
    "run_episodes",
    "run_frontdoor_benchmark",
    "transport_formula",
    "transported_effect",
    "value_iteration",
]


def __getattr__(name: str) -> object:
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = target
    try:
        value = _cast(object, getattr(_import_module(module_name), attribute))
    except ModuleNotFoundError as exc:
        if exc.name == "torch":
            raise ImportError(
                f"{name} requires PyTorch support; install the 'causalrl[torch]' extra"
            ) from exc
        raise
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
