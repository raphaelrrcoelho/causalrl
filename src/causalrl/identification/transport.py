"""Selection-diagram transportability (taxonomy Task 4).

Decide whether a causal effect identified in a source domain transfers to a target domain that
differs in some mechanisms (a *selection diagram*), and produce the transport formula plus a numeric
transported estimate. Conservative by design, mirroring :mod:`causalrl.identification.criteria`: it
proves transportability for the two workhorse cases — direct transportability and S-admissible
adjustment — and returns ``None`` otherwise rather than claiming a result it cannot justify (it does
not implement the full hedge-based sID completeness check).

Faithful to:

- E. Bareinboim, J. Pearl, *Transportability of Causal Effects: Completeness Results*, AAAI 2012.
- E. Bareinboim, J. Pearl, *A General Algorithm for Deciding Transportability of Experimental
  Results*, Journal of Causal Inference 2013 (selection diagrams, S-admissibility, sID).
- J. Pearl, E. Bareinboim, *External Validity: From Do-Calculus to Transportability Across
  Populations*, Statistical Science 2014.

No external code is ported; implemented on our own :class:`~causalrl.scm.graph.CausalGraph`.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product
from typing import TYPE_CHECKING, Literal

from causalrl.exceptions import CausalGraphError
from causalrl.identification._separation import LAT, SEL, d_separated, selection_nodes
from causalrl.identification.counterfactual import counterfactual_expectation
from causalrl.identification.id_algorithm import Estimand, identify_transport
from causalrl.scm.graph import CausalGraph

if TYPE_CHECKING:
    import torch

    from causalrl.scm.scm import StructuralCausalModel

__all__ = [
    "SelectionDiagram",
    "TransportFormula",
    "is_backdoor_admissible",
    "is_transportable",
    "transport_estimand",
    "transport_formula",
    "transported_effect",
]


@dataclass(frozen=True)
class SelectionDiagram:
    """A causal graph plus the variables whose mechanism differs between source and target.

    Each selection variable carries an implicit selection node ``S -> variable`` (Bareinboim &
    Pearl). ``selection_variables`` must be a subset of ``graph.nodes``.
    """

    graph: CausalGraph
    selection_variables: frozenset[str]

    def __post_init__(self) -> None:
        unknown = set(self.selection_variables) - set(self.graph.nodes)
        if unknown:
            raise CausalGraphError(f"selection variables not in graph: {sorted(unknown)}")
        if any(n.startswith(SEL) or n.startswith(LAT) for n in self.graph.nodes):
            raise CausalGraphError(f"node names must not start with {SEL!r} or {LAT!r}")


@dataclass(frozen=True)
class TransportFormula:
    """How to compute ``P*(y | do(x))`` from source and target data."""

    kind: Literal["direct", "adjustment"]
    adjustment_set: frozenset[str]
    expression: str


def is_backdoor_admissible(graph: CausalGraph, treatment: str, outcome: str, z: set[str]) -> bool:
    """Back-door criterion: `z` has no descendant of `treatment` and blocks every back-door path
    (``treatment ⊥ outcome | z`` in the graph with `treatment`'s outgoing edges removed)."""
    if z & graph.descendants(treatment):
        return False
    underline = CausalGraph(
        [(u, v) for u, v in graph.directed_edges if u != treatment],
        graph.bidirected_edges,
        nodes=graph.nodes,
    )
    return d_separated(underline, {treatment}, {outcome}, z)


def transport_formula(
    diagram: SelectionDiagram,
    *,
    treatment: str,
    outcome: str,
    max_adjustment_size: int = 3,
) -> TransportFormula | None:
    """Return the transport formula for ``P*(outcome | do(treatment))``, or ``None`` if it is not
    provably transportable within the supported class (direct / S-admissible adjustment)."""
    graph = diagram.graph
    for name in (treatment, outcome):
        if name not in graph.nodes:
            raise CausalGraphError(f"unknown node: {name!r}")
    selection = diagram.selection_variables
    g_bar_x = graph.do_mutilate(treatment)  # the interventional graph G_{\bar X}
    s_nodes = selection_nodes(selection)

    # Case 1: direct transportability — the interventional law is invariant (S ⊥ Y | X).
    if not selection or d_separated(g_bar_x, s_nodes, {outcome}, {treatment}, selection):
        return TransportFormula(
            "direct",
            frozenset(),
            f"P*({outcome}|do({treatment})) = P({outcome}|do({treatment}))",
        )

    # Case 2: an S-admissible adjustment set Z (back-door admissible and S ⊥ Y | Z, X).
    descendants = graph.descendants(treatment)
    candidates = sorted(
        v for v in graph.nodes if v not in descendants and v not in {treatment, outcome}
    )
    for size in range(min(max_adjustment_size, len(candidates)) + 1):
        for combo in combinations(candidates, size):
            z = set(combo)
            if not is_backdoor_admissible(graph, treatment, outcome, z):
                continue
            if not d_separated(g_bar_x, s_nodes, {outcome}, z | {treatment}, selection):
                continue
            expr = f"P*({outcome}|do {treatment}) = sum_z P({outcome}|{treatment},z) P*(z)"
            return TransportFormula("adjustment", frozenset(z), expr)
    return None


def is_transportable(diagram: SelectionDiagram, *, treatment: str, outcome: str) -> bool:
    """Whether the target effect is provably transportable (see :func:`transport_formula`)."""
    return transport_formula(diagram, treatment=treatment, outcome=outcome) is not None


def transport_estimand(diagram: SelectionDiagram, *, treatment: str, outcome: str) -> Estimand:
    """The general (sID) transport estimand for ``P*(outcome | do(treatment))`` over ``diagram``.

    A :class:`SelectionDiagram` adapter over
    :func:`causalrl.identification.id_algorithm.identify_transport`: each target c-factor is taken
    from the source if its mechanism is invariant, else identified from the target. Raises
    :class:`~causalrl.exceptions.NotIdentifiableError` when not transportable. Generalizes the
    direct / S-admissible-adjustment :func:`transport_formula` (which returns a readable closed form
    for those two cases).
    """
    return identify_transport(diagram.graph, [treatment], [outcome], diagram.selection_variables)


def transported_effect(
    formula: TransportFormula,
    *,
    treatment: str,
    treated_value: float,
    outcome: str,
    source: StructuralCausalModel,
    target: StructuralCausalModel,
    n: int = 40_000,
    seed: int | None = None,
) -> float:
    """Compute ``E*[outcome | do(treatment=treated_value)]`` via the transport `formula`.

    ``direct``: the source interventional mean transfers unchanged. ``adjustment``: reweight the
    source conditionals ``E[outcome | treatment, z]`` by the *target* covariate marginal ``P*(z)``
    (discrete `z` assumed; the demo uses binary covariates). Strata absent from the source sample
    are skipped.
    """
    if formula.kind == "direct":
        return counterfactual_expectation(
            source,
            outcome=outcome,
            intervention={treatment: treated_value},
            evidence={},
            n=n,
            seed=seed,
        )

    z_vars = sorted(formula.adjustment_set)
    src = source.see(n, seed=seed)
    tgt = target.see(n, seed=None if seed is None else seed + 1)
    x_mask = src[treatment] == float(treated_value)
    if not z_vars:
        kept = int(x_mask.sum().item())
        return float(src[outcome][x_mask].float().mean().item()) if kept else 0.0

    value_lists: list[list[float]] = []
    for z in z_vars:
        column: list[float] = src[z].tolist()  # type: ignore[reportUnknownMemberType]
        value_lists.append(sorted(set(column)))
    total = 0.0
    for combo in product(*value_lists):
        src_sel = x_mask.clone()
        tgt_sel: torch.Tensor | None = None
        for z, value in zip(z_vars, combo, strict=True):
            src_sel = src_sel & (src[z] == value)
            cond = tgt[z] == value
            tgt_sel = cond if tgt_sel is None else tgt_sel & cond
        assert tgt_sel is not None
        if int(src_sel.sum().item()) == 0:
            continue
        e_y = float(src[outcome][src_sel].float().mean().item())
        p_z = float(tgt_sel.float().mean().item())
        total += e_y * p_z
    return total
