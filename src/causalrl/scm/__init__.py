"""Causal core: graphs, mechanisms, structural causal models.

Re-exports the package's public surface so callers can ``from causalrl.scm import
StructuralCausalModel`` (the per-submodule paths, e.g. ``causalrl.scm.scm``, keep working).

The torch-backed names (mechanisms, the SCM) are resolved lazily via module ``__getattr__`` so
that importing the torch-free graph surface (``CausalGraph``) — which the top-level package's
lazy loader does when torch is absent — never forces a PyTorch import.
"""

from __future__ import annotations

from importlib import import_module as _import_module
from typing import TYPE_CHECKING

from causalrl.scm.graph import CausalGraph  # torch-free; safe to import eagerly

if TYPE_CHECKING:  # for type checkers / IDEs only; not executed at runtime
    from causalrl.scm.fit import FitReport, NodeFit, fit_scm, fit_scm_mec
    from causalrl.scm.fitters import ANMFit, LinearGaussianFit, NeuralFit, PoissonGLMFit, TabularCPT
    from causalrl.scm.mechanisms import (
        FunctionalMechanism,
        LinearGaussianMechanism,
        LinearMechanism,
        Mechanism,
        NeuralMechanism,
    )
    from causalrl.scm.scm import ExogenousPosterior, StructuralCausalModel
    from causalrl.scm.unrolled import build_unrolled_scm

# name -> (submodule, attribute); resolved on first attribute access.
_LAZY: dict[str, tuple[str, str]] = {
    "FitReport": ("causalrl.scm.fit", "FitReport"),
    "NodeFit": ("causalrl.scm.fit", "NodeFit"),
    "fit_scm": ("causalrl.scm.fit", "fit_scm"),
    "fit_scm_mec": ("causalrl.scm.fit", "fit_scm_mec"),
    "ANMFit": ("causalrl.scm.fitters", "ANMFit"),
    "LinearGaussianFit": ("causalrl.scm.fitters", "LinearGaussianFit"),
    "NeuralFit": ("causalrl.scm.fitters", "NeuralFit"),
    "PoissonGLMFit": ("causalrl.scm.fitters", "PoissonGLMFit"),
    "TabularCPT": ("causalrl.scm.fitters", "TabularCPT"),
    "FunctionalMechanism": ("causalrl.scm.mechanisms", "FunctionalMechanism"),
    "LinearGaussianMechanism": ("causalrl.scm.mechanisms", "LinearGaussianMechanism"),
    "LinearMechanism": ("causalrl.scm.mechanisms", "LinearMechanism"),
    "Mechanism": ("causalrl.scm.mechanisms", "Mechanism"),
    "NeuralMechanism": ("causalrl.scm.mechanisms", "NeuralMechanism"),
    "ExogenousPosterior": ("causalrl.scm.scm", "ExogenousPosterior"),
    "StructuralCausalModel": ("causalrl.scm.scm", "StructuralCausalModel"),
    "build_unrolled_scm": ("causalrl.scm.unrolled", "build_unrolled_scm"),
}

__all__ = [
    "ANMFit",
    "CausalGraph",
    "ExogenousPosterior",
    "FitReport",
    "FunctionalMechanism",
    "LinearGaussianFit",
    "LinearGaussianMechanism",
    "LinearMechanism",
    "Mechanism",
    "NeuralFit",
    "NeuralMechanism",
    "NodeFit",
    "PoissonGLMFit",
    "StructuralCausalModel",
    "TabularCPT",
    "build_unrolled_scm",
    "fit_scm",
    "fit_scm_mec",
]


def __getattr__(name: str) -> object:
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = target
    value = getattr(_import_module(module_name), attribute)
    globals()[name] = value  # cache so subsequent lookups skip __getattr__
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
