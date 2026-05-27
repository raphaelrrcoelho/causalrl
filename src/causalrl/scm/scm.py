from __future__ import annotations

import torch
from torch.distributions import Distribution

from causalrl.exceptions import CausalGraphError, RealizabilityError
from causalrl.scm.graph import CausalGraph
from causalrl.scm.mechanisms import FunctionalMechanism, Mechanism

Tensor = torch.Tensor


class StructuralCausalModel:
    """Executable explicit-latent DAG SCM supporting L1/L2/L3 queries.

    Bidirected-edge ADMGs are accepted by :class:`CausalGraph` for analytical graph
    algorithms, but they are not executable SCMs: shared latent causes must be represented
    as explicit parent nodes with their own mechanism and exogenous distribution.
    """

    def __init__(
        self,
        graph: CausalGraph,
        mechanisms: dict[str, Mechanism],
        exogenous: dict[str, Distribution],
    ) -> None:
        if graph.has_bidirected_edges():
            raise CausalGraphError(
                "StructuralCausalModel requires an explicit latent-variable DAG; "
                "bidirected ADMG edges are analytical only. Represent shared latent causes "
                "as explicit latent nodes."
            )
        nodes = set(graph.nodes)
        for name, entries in (("mechanisms", mechanisms), ("exogenous distributions", exogenous)):
            missing = nodes - set(entries)
            extra = set(entries) - nodes
            if missing or extra:
                raise CausalGraphError(
                    f"{name} must exactly cover graph nodes; "
                    f"missing={sorted(missing)}, extra={sorted(extra)}"
                )
        for node in graph.nodes:
            expected = set(graph.parents(node))
            declared = set(mechanisms[node].parents)
            if declared != expected:
                raise CausalGraphError(
                    f"mechanism parents for {node!r} do not match graph parents; "
                    f"expected={sorted(expected)}, declared={sorted(declared)}"
                )
        self.graph = graph
        self.mechanisms = mechanisms
        self.exogenous = exogenous
        self._generator = torch.Generator()  # type: ignore[reportPrivateImportUsage]
        self._generator.seed()

    def _sample_exogenous(self, n: int, seed: int | None) -> dict[str, Tensor]:
        """Sample using a private CPU RNG stream without mutating process-global Torch state."""
        generator = torch.Generator()  # type: ignore[reportPrivateImportUsage]
        if seed is None:
            generator.set_state(self._generator.get_state())
        else:
            generator.manual_seed(seed)
        with torch.random.fork_rng():  # type: ignore[reportUnknownMemberType]
            torch.random.set_rng_state(generator.get_state())  # type: ignore[reportUnknownMemberType]
            out = {
                name: dist.sample((n,)).reshape(n).float() for name, dist in self.exogenous.items()
            }
            if seed is None:
                self._generator.set_state(torch.random.get_rng_state())  # type: ignore[reportUnknownMemberType]
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

    def counterfactual(
        self,
        evidence: dict[str, float],
        interventions: dict[str, float],
        n: int,
        *,
        seed: int | None = None,
        atol: float = 1e-6,
    ) -> dict[str, Tensor]:
        """Layer 3: abduction-action-prediction via rejection sampling.

        Draw n exogenous samples, keep those whose factual evaluation matches `evidence`,
        then re-evaluate the mutilated model under `interventions` with the retained noise.
        """
        noise = self._sample_exogenous(n, seed)
        factual = self._evaluate(noise)

        mask = torch.ones(n, dtype=torch.bool)  # type: ignore[reportPrivateImportUsage]
        for node, val in evidence.items():
            if node not in self.mechanisms:
                raise CausalGraphError(f"unknown evidence node: {node!r}")
            mask &= (factual[node] - float(val)).abs() <= atol
        kept = int(mask.sum())
        if kept == 0:
            raise RealizabilityError(
                f"no exogenous draws match evidence {evidence!r}; "
                "increase n or check that the evidence has nonzero probability"
            )

        retained = {name: u[mask] for name, u in noise.items()}
        cf_model = self.do(interventions) if interventions else self
        return cf_model._evaluate(retained)

    def see(self, n: int, *, seed: int | None = None) -> dict[str, Tensor]:
        """Layer 1: draw n observational samples P(V)."""
        noise = self._sample_exogenous(n, seed)
        return self._evaluate(noise)
