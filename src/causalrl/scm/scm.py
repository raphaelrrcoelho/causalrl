from __future__ import annotations

from collections.abc import Sequence

import torch
from torch.distributions import Distribution

from causalrl.exceptions import CausalGraphError, RealizabilityError
from causalrl.scm.graph import CausalGraph
from causalrl.scm.mechanisms import FunctionalMechanism, Mechanism

Tensor = torch.Tensor

# An intervention / known-noise value: a scalar (broadcast to all n units) OR a length-n
# vector applied elementwise (the per-sample / per-trajectory path).
Value = float | Sequence[float] | Tensor


def _broadcast_value(value: object, n: int) -> Tensor:
    """Coerce an intervention / known-noise value to a length-``n`` float tensor.

    A python/0-d scalar is broadcast to all ``n`` units (the original behaviour). A
    sequence/array/tensor of length ``n`` is applied elementwise (one value per sample) —
    the per-trajectory path. Any other length is a programming error and raises.
    """
    if isinstance(value, Tensor):
        t = value
    elif isinstance(value, (int, float, bool)):
        return torch.full((n,), float(value))  # type: ignore[reportPrivateImportUsage]
    else:
        t = torch.as_tensor(value)
    t = t.float().reshape(-1)
    if t.numel() == 1:
        return t.expand(n).clone()
    if t.numel() != n:
        raise ValueError(
            f"per-sample value has length {t.numel()} but n={n}; "
            "pass a scalar (broadcast to all units) or a length-n vector"
        )
    return t


class ExogenousPosterior:
    """Retained exogenous values from abduction, ready for prediction under interventions.

    Holds the source SCM and a dict of exogenous tensors (the abducted units). Call
    :meth:`predict` to evaluate the model — optionally mutilated by ``do`` — on them. Abduct
    once, predict under many interventions (the efficient counterfactual pattern).
    """

    def __init__(self, scm: StructuralCausalModel, noise: dict[str, Tensor]) -> None:
        self._scm = scm
        self.noise = noise

    def __len__(self) -> int:
        return int(next(iter(self.noise.values())).shape[0]) if self.noise else 0

    def predict(self, *, do: dict[str, Value] | None = None) -> dict[str, Tensor]:
        """Evaluate the (optionally do-mutilated) model on the retained exogenous units."""
        model = self._scm.do(do) if do else self._scm
        return model._evaluate(self.noise)


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

    def do(self, interventions: dict[str, Value]) -> StructuralCausalModel:
        """Layer 2: return the mutilated SCM under do(interventions). Original is unchanged."""
        graph = self.graph
        mechanisms = dict(self.mechanisms)
        for node, value in interventions.items():
            if node not in self.mechanisms:
                raise CausalGraphError(f"cannot intervene on unknown node: {node!r}")
            graph = graph.remove_incoming_edges(node)
            if isinstance(value, (int, float, bool)):
                const = float(value)
                mechanisms[node] = FunctionalMechanism(
                    [],
                    lambda pa, u, _c=const: torch.full_like(u, _c),  # type: ignore[reportPrivateImportUsage]
                )
            else:
                # Per-sample vector: pin elementwise, broadcast-checked against the unit count.
                mechanisms[node] = FunctionalMechanism(
                    [],
                    lambda pa, u, _v=value: _broadcast_value(_v, u.shape[0]).to(u.dtype),
                )
        return StructuralCausalModel(graph, mechanisms, self.exogenous)

    def abduct(
        self,
        evidence: dict[str, float] | None = None,
        *,
        known: dict[str, Value] | None = None,
        n: int = 20_000,
        seed: int | None = None,
        atol: float = 1e-6,
    ) -> ExogenousPosterior:
        """Layer 3, step 1 — infer the exogenous posterior given evidence/known noise.

        ``known`` pins supplied exogenous values *exactly* (the exact, continuous path: no
        rejection). Remaining exogenous are sampled; if ``evidence`` is given they are
        rejection-filtered so the factual evaluation matches ``evidence`` within ``atol``.
        Returns an :class:`ExogenousPosterior`; call its ``predict(do=...)``.
        """
        known = known or {}
        bad = set(known) - set(self.exogenous)
        if bad:
            raise CausalGraphError(f"unknown exogenous node(s): {sorted(bad)}")
        # Exact path: all-known exogenous, no evidence to reject against.
        if known and not evidence:
            noise = {
                name: (
                    _broadcast_value(known[name], n)
                    if name in known
                    else dist.sample((n,)).reshape(n).float()
                )
                for name, dist in self.exogenous.items()
            }
            return ExogenousPosterior(self, noise)
        # Evidence path: sample, pin any known, reject to match evidence.
        sampled = self._sample_exogenous(n, seed)
        for name, val in known.items():
            sampled[name] = _broadcast_value(val, n)
        factual = self._evaluate(sampled)
        mask = torch.ones(n, dtype=torch.bool)  # type: ignore[reportPrivateImportUsage]
        for node, val in (evidence or {}).items():
            if node not in self.mechanisms:
                raise CausalGraphError(f"unknown evidence node: {node!r}")
            mask &= (factual[node] - float(val)).abs() <= atol
        if int(mask.sum()) == 0:
            raise RealizabilityError(
                f"no exogenous draws match evidence {evidence!r}; "
                "increase n or check that the evidence has nonzero probability"
            )
        return ExogenousPosterior(self, {name: u[mask] for name, u in sampled.items()})

    def counterfactual(
        self,
        evidence: dict[str, float],
        interventions: dict[str, Value],
        n: int,
        *,
        seed: int | None = None,
        atol: float = 1e-6,
    ) -> dict[str, Tensor]:
        """Layer 3: abduction-action-prediction. Sugar over :meth:`abduct` + predict."""
        post = self.abduct(evidence, n=n, seed=seed, atol=atol)
        return post.predict(do=interventions or None)

    def see(self, n: int, *, seed: int | None = None) -> dict[str, Tensor]:
        """Layer 1: draw n observational samples P(V)."""
        noise = self._sample_exogenous(n, seed)
        return self._evaluate(noise)
