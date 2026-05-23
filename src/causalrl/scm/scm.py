from __future__ import annotations

import torch
from torch.distributions import Distribution

from causalrl.exceptions import CausalGraphError
from causalrl.scm.graph import CausalGraph
from causalrl.scm.mechanisms import FunctionalMechanism, Mechanism

Tensor = torch.Tensor


class StructuralCausalModel:
    """M = <V, U, F, P(U)>. Supports L1 (see), L2 (do), L3 (counterfactual) queries."""

    def __init__(
        self,
        graph: CausalGraph,
        mechanisms: dict[str, Mechanism],
        exogenous: dict[str, Distribution],
    ) -> None:
        self.graph = graph
        self.mechanisms = mechanisms
        self.exogenous = exogenous
        missing = set(graph.nodes) - set(mechanisms)
        if missing:
            raise ValueError(f"missing mechanisms for nodes: {sorted(missing)}")

    def _sample_exogenous(self, n: int, _generator: torch.Generator | None) -> dict[str, Tensor]:  # type: ignore[reportPrivateImportUsage]
        # torch Distributions don't accept a generator; seed globally for reproducibility.
        out: dict[str, Tensor] = {}
        for name, dist in self.exogenous.items():
            out[name] = dist.sample((n,)).reshape(n).float()
        return out

    def _evaluate(self, noise: dict[str, Tensor]) -> dict[str, Tensor]:
        values: dict[str, Tensor] = {}
        for node in self.graph.topological_order():
            mech = self.mechanisms[node]
            parent_values = {p: values[p] for p in self.graph.parents(node)}
            values[node] = mech(parent_values, noise[node])
        return values

    def do(self, interventions: dict[str, float]) -> StructuralCausalModel:
        """Layer 2: return the mutilated SCM under do(interventions). Original is unchanged."""
        graph = self.graph
        mechanisms = dict(self.mechanisms)
        for node, value in interventions.items():
            if node not in self.mechanisms:
                raise CausalGraphError(f"cannot intervene on unknown node: {node!r}")
            graph = graph.remove_incoming_edges(node)
            const = float(value)
            mechanisms[node] = FunctionalMechanism(
                [],
                lambda pa, u, _c=const: torch.full_like(u, _c),  # type: ignore[reportPrivateImportUsage]
            )
        return StructuralCausalModel(graph, mechanisms, self.exogenous)

    def see(self, n: int, *, seed: int | None = None) -> dict[str, Tensor]:
        """Layer 1: draw n observational samples P(V)."""
        if seed is not None:
            torch.manual_seed(seed)  # type: ignore[reportUnknownMemberType]
        noise = self._sample_exogenous(n, None)
        return self._evaluate(noise)
