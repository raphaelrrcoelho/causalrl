"""Build a randomly-parameterized StructuralCausalModel realizing a given ADMG.

Unrolls each BIDIRECTED EDGE into its own explicit shared Bernoulli latent feeding exactly its
two endpoints (NOT one shared latent per whole c-component -- a c-component that is a chain
rather than a bidirected clique, e.g. A<->B<->C without A<->C, is not faithfully realized by a
single latent touching all three; that over-confounds A and C and silently tests a different,
more-confounded ADMG than the one passed in). Every node gets a random-parity (XOR) mechanism of
its full parent set plus its own independent flip noise. Used to dogfood
identify_effect/estimate_effect against ground-truth simulation across many random small graphs
(issue #16).
"""

from __future__ import annotations

import numpy as np
import torch
from torch.distributions import Uniform

from causalrl.scm.graph import CausalGraph
from causalrl.scm.mechanisms import FunctionalMechanism
from causalrl.scm.scm import StructuralCausalModel


def _flip(u: torch.Tensor, p: float) -> torch.Tensor:
    return (u < p).float()


def build_random_scm(graph: CausalGraph, *, seed: int) -> StructuralCausalModel:
    rng = np.random.default_rng(seed)
    nodes = graph.nodes
    latent_nodes = [f"__U{i}" for i in range(len(graph.bidirected_edges))]

    directed = list(graph.directed_edges)
    for lname, (u, v) in zip(latent_nodes, graph.bidirected_edges, strict=True):
        directed.append((lname, u))
        directed.append((lname, v))
    unrolled = CausalGraph(directed_edges=directed, nodes=nodes + latent_nodes)

    mechanisms: dict = {}
    exogenous: dict = {}
    for lname in latent_nodes:
        p = float(rng.uniform(0.3, 0.7))
        mechanisms[lname] = FunctionalMechanism([], lambda _pa, u, p=p: _flip(u, p))
        exogenous[lname] = Uniform(0.0, 1.0)

    for n in nodes:
        full_pa = list(unrolled.parents(n))
        flip_p = float(rng.uniform(0.02, 0.15))
        if full_pa:
            signs = rng.integers(0, 2, size=len(full_pa))

            def fn(pa, u, full_pa=full_pa, signs=signs, flip_p=flip_p):
                acc = torch.zeros_like(u)  # type: ignore[reportPrivateImportUsage]
                for name, sign in zip(full_pa, signs, strict=True):
                    acc = acc + (pa[name] if sign else (1.0 - pa[name]))
                parity = acc % 2.0
                return torch.where(u < flip_p, 1.0 - parity, parity)  # type: ignore[reportPrivateImportUsage]

            mechanisms[n] = FunctionalMechanism(full_pa, fn)
        else:
            p0 = float(rng.uniform(0.3, 0.7))
            mechanisms[n] = FunctionalMechanism([], lambda _pa, u, p0=p0: _flip(u, p0))
        exogenous[n] = Uniform(0.0, 1.0)

    return StructuralCausalModel(unrolled, mechanisms, exogenous)
