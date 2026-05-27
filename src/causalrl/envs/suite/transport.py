"""Two-domain transportability demo (taxonomy Task 4).

A source and a target domain that share every mechanism except the covariate distribution
``P(Z)`` — the canonical selection-diagram example (Bareinboim & Pearl). The graph is
``Z -> X, Z -> Y, X -> Y`` with selection variable ``Z``. Because ``do(X)`` does not sever
``Z -> Y``, the source interventional effect is biased for the target; the S-admissible adjustment
``sum_z P(Y|X,z) P*(z)`` recovers the true target effect.
"""

from __future__ import annotations

import torch
from torch.distributions import Bernoulli, Distribution, Uniform

from causalrl.identification.transport import SelectionDiagram
from causalrl.scm.graph import CausalGraph
from causalrl.scm.mechanisms import FunctionalMechanism, Mechanism
from causalrl.scm.scm import StructuralCausalModel

_EDGES = [("Z", "X"), ("Z", "Y"), ("X", "Y")]


def _build_domain(p_z: float) -> StructuralCausalModel:
    graph = CausalGraph(directed_edges=_EDGES)

    def x_mech(pa: dict[str, torch.Tensor], u: torch.Tensor) -> torch.Tensor:
        # P(X=1) = 0.8 if Z==1 else 0.3
        p = 0.3 + 0.5 * (pa["Z"] == 1).float()
        return (u < p).float()

    def y_mech(pa: dict[str, torch.Tensor], u: torch.Tensor) -> torch.Tensor:
        # P(Y=1 | X, Z): (1,1)=0.9, (1,0)=0.5, (0,1)=0.4, (0,0)=0.1
        x, z = pa["X"], pa["Z"]
        p = 0.1 + 0.4 * x + 0.3 * z + 0.1 * x * z
        return (u < p).float()

    mechanisms: dict[str, Mechanism] = {
        "Z": FunctionalMechanism([], lambda pa, u: u),
        "X": FunctionalMechanism(["Z"], x_mech),
        "Y": FunctionalMechanism(["X", "Z"], y_mech),
    }
    exogenous: dict[str, Distribution] = {
        "Z": Bernoulli(p_z),
        "X": Uniform(0.0, 1.0),
        "Y": Uniform(0.0, 1.0),
    }
    return StructuralCausalModel(graph, mechanisms, exogenous)


def make_transport_domains(
    p_z_source: float = 0.2, p_z_target: float = 0.8
) -> tuple[StructuralCausalModel, StructuralCausalModel, SelectionDiagram]:
    """Return ``(source, target, diagram)``. Domains differ only in ``P(Z)``; the selection
    diagram marks ``Z``. The true target ``E*[Y|do(X=1)] = 0.9*P*(Z=1) + 0.5*P*(Z=0)``."""
    source = _build_domain(p_z_source)
    target = _build_domain(p_z_target)
    diagram = SelectionDiagram(CausalGraph(directed_edges=_EDGES), frozenset({"Z"}))
    return source, target, diagram
