"""Causal-discovery demo dataset (taxonomy Task 5).

A latent-free SCM ``X -> Z, Y -> Z, Z -> W`` with a collider at ``Z`` and a downstream child ``W``.
``X, Y ~ Bernoulli(0.5)``; ``Z`` is a *noisy* OR of ``X, Y``, and ``W`` a *noisy* copy of ``Z``
(each flipped with probability 0.1). The true CPDAG is fully oriented: ``X -> Z <- Y`` (a
v-structure) plus ``Z -> W`` (forced by Meek R1).

The noise matters: ``XOR`` would make a parent marginally independent of the collider, and a
deterministic ``Z`` or ``W`` would let conditioning on it numerically determine its argument. Both
break the faithfulness that constraint-based discovery assumes.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.distributions import Bernoulli, Distribution, Uniform

from causalrl.scm.graph import CausalGraph
from causalrl.scm.mechanisms import FunctionalMechanism, Mechanism
from causalrl.scm.scm import StructuralCausalModel

_EDGES = [("X", "Z"), ("Y", "Z"), ("Z", "W")]


def build_discovery_scm() -> StructuralCausalModel:
    """The collider SCM ``X -> Z <- Y``, ``Z -> W``: ``Z`` is a noisy OR of ``X, Y`` and ``W`` a
    noisy copy of ``Z`` (each flipped with probability 0.1)."""
    graph = CausalGraph(directed_edges=_EDGES)

    def noisy_or(pa: dict[str, torch.Tensor], u: torch.Tensor) -> torch.Tensor:
        base = ((pa["X"] + pa["Y"]) > 0).float()
        return (base + (u < 0.1).float()) % 2

    def noisy_copy(pa: dict[str, torch.Tensor], u: torch.Tensor) -> torch.Tensor:
        return (pa["Z"] + (u < 0.1).float()) % 2

    mechanisms: dict[str, Mechanism] = {
        "X": FunctionalMechanism([], lambda pa, u: u),
        "Y": FunctionalMechanism([], lambda pa, u: u),
        "Z": FunctionalMechanism(["X", "Y"], noisy_or),
        "W": FunctionalMechanism(["Z"], noisy_copy),
    }
    exogenous: dict[str, Distribution] = {
        "X": Bernoulli(0.5),
        "Y": Bernoulli(0.5),
        "Z": Uniform(0.0, 1.0),
        "W": Uniform(0.0, 1.0),
    }
    return StructuralCausalModel(graph, mechanisms, exogenous)


def sample_discovery_data(n: int = 10_000, seed: int | None = 0) -> dict[str, np.ndarray]:
    """Sample ``n`` rows from :func:`build_discovery_scm` as integer columns for discovery."""
    samples = build_discovery_scm().see(n, seed=seed)
    out: dict[str, np.ndarray] = {}
    for name, column in samples.items():
        out[name] = column.long().numpy()
    return out
