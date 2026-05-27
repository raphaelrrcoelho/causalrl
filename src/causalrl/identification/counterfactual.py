"""Layer-3 counterfactual decision quantities: ETT and the Regret Decision Criterion.

These compute counterfactual decision estimands on an executable
:class:`~causalrl.scm.scm.StructuralCausalModel` via its abduction-action-prediction
``counterfactual`` query. They power counterfactual-optimal policies that condition the action
choice on the agent's own intent (a proxy for the unobserved confounder), beating the best fixed
interventional action under confounding.

Faithful to:

- E. Bareinboim, A. Forney, J. Pearl, *Bandits with Unobserved Confounders: A Causal Approach*,
  NeurIPS 2015 — the Regret Decision Criterion (condition the choice on the agent's intuition).
- J. Pearl, *Causality* (2nd ed.), §8.2.1 — the Effect of Treatment on the Treated.
- A. Forney, J. Pearl, E. Bareinboim, *Counterfactual Data-Fusion for Online Reinforcement
  Learners*, ICML 2017.

The estimands are implemented on our own SCM; no external code is ported.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from causalrl.exceptions import CausalGraphError

if TYPE_CHECKING:
    from causalrl.scm.scm import StructuralCausalModel

__all__ = [
    "counterfactual_expectation",
    "effect_of_treatment_on_treated",
    "regret_decision_table",
]


def _validate_nodes(scm: StructuralCausalModel, names: set[str]) -> None:
    unknown = names - set(scm.graph.nodes)
    if unknown:
        raise CausalGraphError(f"unknown node(s) in counterfactual query: {sorted(unknown)}")


def counterfactual_expectation(
    scm: StructuralCausalModel,
    *,
    outcome: str,
    intervention: dict[str, float],
    evidence: dict[str, float],
    n: int = 20_000,
    seed: int | None = None,
) -> float:
    """Return ``E[ outcome_{do(intervention)} | evidence ]`` (a Layer-3 counterfactual mean).

    Wraps :meth:`StructuralCausalModel.counterfactual` (abduction-action-prediction) and averages
    the outcome over the retained, evidence-consistent units. With empty ``evidence`` this reduces
    to the interventional mean ``E[outcome_{do(intervention)}]``.
    """
    _validate_nodes(scm, {outcome} | set(intervention) | set(evidence))
    result = scm.counterfactual(evidence, intervention, n, seed=seed)
    return float(result[outcome].float().mean().item())


def effect_of_treatment_on_treated(
    scm: StructuralCausalModel,
    *,
    treatment: str,
    outcome: str,
    treated: float,
    control: float,
    n: int = 20_000,
    seed: int | None = None,
) -> float:
    """Effect of Treatment on the Treated: ``E[Y_{treated} - Y_{control} | treatment = treated]``.

    The treatment effect among the subpopulation that actually received ``treated`` (Pearl,
    *Causality* §8.2.1). Under confounding this differs from the average treatment effect. Both
    potential outcomes use the same ``seed`` (common random numbers), so they are evaluated on the
    same abducted units and the difference is matched and low-variance.
    """
    factual = counterfactual_expectation(
        scm,
        outcome=outcome,
        intervention={treatment: treated},
        evidence={treatment: treated},
        n=n,
        seed=seed,
    )
    counterfactual = counterfactual_expectation(
        scm,
        outcome=outcome,
        intervention={treatment: control},
        evidence={treatment: treated},
        n=n,
        seed=seed,
    )
    return factual - counterfactual


def regret_decision_table(
    scm: StructuralCausalModel,
    *,
    outcome: str,
    action_node: str,
    intent_node: str,
    arms: Sequence[int],
    intents: Sequence[int],
    n: int = 20_000,
    seed: int | None = None,
) -> dict[int, dict[int, float]]:
    """The Regret Decision Criterion table ``intent -> {arm -> E[Y_{do(arm)} | intent]}``.

    Each cell is a counterfactual reward: the expected ``outcome`` of playing ``arm`` for the
    subpopulation whose natural intent is ``intent``. The optimal counterfactual policy plays the
    ``argmax`` arm of each row. A per-cell ``seed`` offset keeps the table reproducible.
    """
    _validate_nodes(scm, {outcome, action_node, intent_node})
    table: dict[int, dict[int, float]] = {}
    for i, intent in enumerate(intents):
        row: dict[int, float] = {}
        for j, arm in enumerate(arms):
            cell_seed = None if seed is None else seed + i * len(arms) + j
            row[int(arm)] = counterfactual_expectation(
                scm,
                outcome=outcome,
                intervention={action_node: float(arm)},
                evidence={intent_node: float(intent)},
                n=n,
                seed=cell_seed,
            )
        table[int(intent)] = row
    return table
