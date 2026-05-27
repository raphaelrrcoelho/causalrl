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
    "BenchmarkEstimate": ("causalrl.eval.benchmark", "BenchmarkEstimate"),
    "CausalEnv": ("causalrl.envs.base", "CausalEnv"),
    "CausalGraph": ("causalrl.scm.graph", "CausalGraph"),
    "CausalGraphError": ("causalrl.exceptions", "CausalGraphError"),
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
    "RealizabilityError": ("causalrl.exceptions", "RealizabilityError"),
    "UnverifiedAssumptionError": ("causalrl.exceptions", "UnverifiedAssumptionError"),
    "SequentialDTREnv": ("causalrl.envs.suite.seq_dtr", "SequentialDTREnv"),
    "SequentialMABUCEnv": ("causalrl.envs.suite.seq_mabuc", "SequentialMABUCEnv"),
    "StructuralCausalBanditEnv": ("causalrl.envs.suite.scbandit", "StructuralCausalBanditEnv"),
    "StructuralCausalModel": ("causalrl.scm.scm", "StructuralCausalModel"),
    "Transition": ("causalrl.data.dataset", "Transition"),
    "backdoor_adjustment_set": ("causalrl.identification.criteria", "backdoor_adjustment_set"),
    "causal_q_bounds": ("causalrl.identification.bounds", "causal_q_bounds"),
    "counterfactual_expectation": (
        "causalrl.identification.counterfactual",
        "counterfactual_expectation",
    ),
    "cumulative_regret": ("causalrl.eval.metrics", "cumulative_regret"),
    "effect_of_treatment_on_treated": (
        "causalrl.identification.counterfactual",
        "effect_of_treatment_on_treated",
    ),
    "finite_horizon_regret": ("causalrl.eval.metrics", "finite_horizon_regret"),
    "generate_logs": ("causalrl.data.dataset", "generate_logs"),
    "ipw_value": ("causalrl.eval.ope", "ipw_value"),
    "is_identifiable": ("causalrl.identification.criteria", "is_identifiable"),
    "minimal_intervention_sets": (
        "causalrl.identification.intervention_sets",
        "minimal_intervention_sets",
    ),
    "pomis": ("causalrl.identification.intervention_sets", "pomis"),
    "report_to_dict": ("causalrl.eval.benchmark", "report_to_dict"),
    "run_episodes": ("causalrl.eval.harness", "run_episodes"),
    "run_confounded_chain_benchmark": ("causalrl.eval.benchmark", "run_confounded_chain_benchmark"),
    "run_frontdoor_benchmark": ("causalrl.eval.benchmark", "run_frontdoor_benchmark"),
}

__all__ = [
    "DOVI",
    "UCDTR",
    "Agent",
    "BenchmarkEstimate",
    "BruteForceInterventionTS",
    "CausalEnv",
    "CausalGraph",
    "CausalGraphError",
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
    "RealizabilityError",
    "SequentialDTREnv",
    "SequentialMABUCEnv",
    "StructuralCausalBanditEnv",
    "StructuralCausalModel",
    "Transition",
    "UnverifiedAssumptionError",
    "__version__",
    "backdoor_adjustment_set",
    "causal_q_bounds",
    "counterfactual_expectation",
    "cumulative_regret",
    "effect_of_treatment_on_treated",
    "finite_horizon_regret",
    "generate_logs",
    "ipw_value",
    "is_identifiable",
    "minimal_intervention_sets",
    "pomis",
    "report_to_dict",
    "run_confounded_chain_benchmark",
    "run_episodes",
    "run_frontdoor_benchmark",
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
