"""Transport-regret certificate: a decision-regret bound + abstention rule on a selection diagram.

Turns the transport engine's *decision* — is ``P*(value | do(action))`` transportable across the
selection diagram, and through what adjustment — into an **operational guarantee for a specific
policy/planner** deployed across the domain shift:

* a computable upper bound on the policy's aggregate **transfer regret** (an
  :class:`~causalrl.identification.bounds.Interval`, Manski/MSM style), and
* a per-unit **abstention mask** that fires exactly where the policy's decision depends on a
  selection-marked (non-transportable) mechanism.

Composes existing engine outputs (:func:`is_transportable_effect`,
:func:`~causalrl.identification.transport.transport_formula`) — it introduces **no new
identification theory** and inherits :mod:`causalrl.identification.transport`'s conservatism (it is
a sound decision-regret bound, not the complete hedge-based sID partial-ID bound).

VALIDITY (pinned by the unit tests). For a deterministic policy ``pi`` whose only source/target
difference is the selection-marked mechanism (``S -> W`` for a witness variable ``W``), and a
paired unit distribution, the true aggregate transfer regret satisfies

    R_shift = E[ V^pi(u; source W-law) - V^pi(u; target W-law) ]
            <= (V_max - V_min) * mu,

where ``mu`` is the **divergence rate of the executed behaviour** under the ``do(W)``-sweep
(:func:`decision_flip_rate`). Where the behaviour is invariant to ``W`` the executed trajectory is
identical under both laws (``W`` is causally inert for the dynamics — encoded in the diagram), so
the unit contributes 0; where it diverges the value drop is at most the span. The *trajectory*
divergence rate is required — a single early ``W``-driven action flip cascades, so the
single-decision flip rate (:func:`decision_abstain_mask`, the abstention statistic) under-counts
and must NOT scale the bound. Validated end-to-end (coverage, non-vacuity, calibrated abstention,
silent negative control) as the G1 "Spurious CoinRun" world-model-transfer arena certificate and
promoted on its GO (the experiments' promote-on-pass rule).

Faithful to:

- E. Bareinboim, J. Pearl, *Transportability of Causal Effects: Completeness Results*, AAAI 2012
  (selection diagrams; what the witness detection rides on).
- J. Pearl, E. Bareinboim, *External Validity: From Do-Calculus to Transportability Across
  Populations*, Statistical Science 2014.
- P. de Haan, D. Jayaraman, S. Levine, *Causal Confusion in Imitation Learning*, NeurIPS 2019
  (the failure mode the abstention rule targets).
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass
from typing import TypeVar

import numpy as np
from numpy.typing import NDArray

from causalrl.identification.bounds import Interval
from causalrl.identification.id_algorithm import is_transportable_effect
from causalrl.identification.transport import (
    SelectionDiagram,
    TransportFormula,
    transport_formula,
)

__all__ = [
    "TransportRegretCertificate",
    "decision_abstain_mask",
    "decision_flip_rate",
    "transport_regret_certificate",
]

U = TypeVar("U")


@dataclass(frozen=True)
class TransportRegretCertificate:
    """The transport-regret certificate for ``P*(value | do(action))`` and a specific policy.

    Attributes
    ----------
    transportable:
        :func:`is_transportable_effect` on the selection diagram.
    formula:
        The readable :class:`TransportFormula` for the transportable part, or ``None``.
    non_transportable_witness:
        The selection-marked variable(s) the policy must not trust off-distribution: the
        selection-marked variables appearing in the transport formula's adjustment set (the
        mechanism one would have to re-weight by the target marginal to transport), or the full
        selection set if the effect is outright non-identifiable. Empty for a ``direct`` formula
        with empty adjustment — the certified zero-regret case.
    reweight_required:
        Whether transporting the queried effect requires re-weighting by a selection-marked
        variable (``non_transportable_witness`` non-empty). Can be ``True`` even when
        ``transportable`` is ``True``: a policy that merely *reads* the witness without
        re-weighting still incurs the bounded regret.
    decision_dependence:
        ``mu`` — the fraction of decision units whose executed behaviour depends on the
        selection-marked mechanism, measured by the ``do()``-sweep (:func:`decision_flip_rate`).
        A property of the *specific policy being certified*, supplied by the caller.
    value_range:
        ``(V_min, V_max)`` of the policy value, used to scale the bound.
    regret_bound:
        ``Interval(0, (V_max - V_min) * mu)`` — the computable transfer-regret upper bound.
        Collapses to ``Interval(0, 0)`` for a behaviour-invariant policy (``mu = 0``).
    """

    transportable: bool
    formula: TransportFormula | None
    non_transportable_witness: frozenset[str]
    reweight_required: bool
    decision_dependence: float
    value_range: tuple[float, float]
    regret_bound: Interval

    def is_vacuous(self, *, frac: float = 0.9) -> bool:
        """Whether the bound is vacuous: its width >= ``frac`` of the trivial value span."""
        v_min, v_max = self.value_range
        span = v_max - v_min
        if span <= 0:
            return True
        width = self.regret_bound.upper - self.regret_bound.lower
        return width >= frac * span


def decision_flip_rate(
    trace: Callable[[U, int], Hashable],
    units: Sequence[U],
    *,
    values: tuple[int, int] = (0, 1),
) -> float:
    """``mu`` — fraction of units whose *executed trajectory* diverges under the ``do()``-sweep.

    ``trace(unit, w)`` returns the policy's executed rollout trace (action or visited-state
    sequence — any hashable summary of *what the policy does*) from ``unit`` under
    ``do(W = w)``, with everything else held fixed. Two traces differing is exactly the event on
    which ``V^pi(u; w0) != V^pi(u; w1)`` can be nonzero, so ``span * mu`` is a *valid* regret
    bound (module docstring). Use this — not the single-decision flip rate — to scale the bound.
    """
    if not units:
        return 0.0
    w0, w1 = values
    flips = sum(1 for u in units if trace(u, w0) != trace(u, w1))
    return flips / len(units)


def decision_abstain_mask(
    greedy: Callable[[U, int], object],
    units: Sequence[U],
    *,
    values: tuple[int, int] = (0, 1),
) -> NDArray[np.bool_]:
    """Abstention mask: fire where the policy's *immediate* greedy decision is not invariant to
    intervening on the selection-marked witness.

    Fires on ``unit`` iff ``greedy(unit, w0) != greedy(unit, w1)``. This is the per-unit
    decision-flip rule — by construction it recovers the truly-confused set when "confused" is
    defined as the same single-decision flip (the mechanistic guarantee the G1 gate's M4 pins).
    A behaviour-invariant policy has an (almost) empty mask. Distinct from
    :func:`decision_flip_rate`'s trajectory-divergence ``mu`` that scales the regret bound.
    """
    w0, w1 = values
    mask = np.zeros(len(units), dtype=bool)
    for i, u in enumerate(units):
        mask[i] = greedy(u, w0) != greedy(u, w1)
    return mask


def transport_regret_certificate(
    diagram: SelectionDiagram,
    *,
    action: str,
    value: str,
    value_range: tuple[float, float],
    decision_dependence: float,
    max_adjustment_size: int = 3,
) -> TransportRegretCertificate:
    """Build the transport-regret certificate for ``P*(value | do(action))`` on ``diagram``.

    Composes the library transport decision with the decision-regret wrapper:

    * ``transportable`` / ``formula`` from the engine (:func:`is_transportable_effect` /
      :func:`transport_formula`).
    * ``non_transportable_witness`` = selection-marked variables the transport formula must
      re-weight (they appear in its adjustment set); the full selection set if the effect is
      non-identifiable; empty for a direct formula with empty adjustment (certified zero).
    * ``regret_bound`` = ``Interval(0, (V_max - V_min) * mu)`` with ``mu`` the supplied
      ``decision_dependence`` (the ``do()``-sweep result, :func:`decision_flip_rate` — the
      trajectory-divergence rate, NOT the abstention statistic).

    ``decision_dependence`` is supplied by the caller because it is a property of the *specific
    policy* being certified; the certificate primitive itself stays a pure composition of
    causal-engine outputs.
    """
    if not 0.0 <= decision_dependence <= 1.0:
        raise ValueError(f"decision_dependence must be in [0, 1], got {decision_dependence}")
    v_min, v_max = value_range
    if v_max < v_min:
        raise ValueError(
            f"value_range must be (V_min, V_max) with V_max >= V_min, got {value_range}"
        )

    selection = diagram.selection_variables
    transportable = is_transportable_effect(diagram.graph, [action], [value], selection)
    formula = transport_formula(
        diagram, treatment=action, outcome=value, max_adjustment_size=max_adjustment_size
    )

    if formula is None:
        witness: frozenset[str] = frozenset(selection)
    else:
        witness = frozenset(selection) & formula.adjustment_set
    reweight_required = len(witness) > 0

    span = v_max - v_min
    regret_bound = Interval(0.0, float(span * decision_dependence))

    return TransportRegretCertificate(
        transportable=transportable,
        formula=formula,
        non_transportable_witness=witness,
        reweight_required=reweight_required,
        decision_dependence=float(decision_dependence),
        value_range=(float(v_min), float(v_max)),
        regret_bound=regret_bound,
    )
