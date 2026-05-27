"""Counterfactual decision-making agents (Layer 3).

A model-based policy that decides by querying the counterfactual reward ``E[Y_{do(a)} | intent]``
on a known SCM — the Regret Decision Criterion of Bareinboim, Forney & Pearl (NeurIPS 2015).
Conditioning the choice on the agent's own intent (a proxy for the unobserved confounder) recovers
the per-intent optimum that the best fixed intervention cannot see.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from causalrl.agents.base import Agent
from causalrl.identification.counterfactual import regret_decision_table

if TYPE_CHECKING:
    from causalrl.scm.scm import StructuralCausalModel

__all__ = ["CounterfactualOptimalPolicy"]


class CounterfactualOptimalPolicy(Agent):
    """Plays ``argmax_a E[Y_{do(action_node=a)} | intent]`` from a known SCM.

    The Layer-3 oracle: it precomputes the Regret Decision Criterion table once at construction and
    then acts greedily on the observed intent. ``update`` is a no-op — the model is known, so there
    is nothing to learn online. The computed table is exposed as ``decision_table`` for inspection.
    """

    def __init__(
        self,
        scm: StructuralCausalModel,
        *,
        outcome: str,
        action_node: str,
        intent_node: str,
        arms: Sequence[int],
        intents: Sequence[int],
        intent_key: str = "intuition",
        n: int = 20_000,
        seed: int | None = None,
    ) -> None:
        self._intent_key = intent_key
        self.decision_table = regret_decision_table(
            scm,
            outcome=outcome,
            action_node=action_node,
            intent_node=intent_node,
            arms=arms,
            intents=intents,
            n=n,
            seed=seed,
        )
        self._best_arm: dict[int, int] = {
            intent: max(row, key=lambda arm: row[arm])
            for intent, row in self.decision_table.items()
        }

    def act(self, observation: dict[str, Any]) -> int:
        return self._best_arm[int(observation[self._intent_key])]

    def update(self, observation: dict[str, Any], action: int, reward: float) -> None:
        """No-op: the SCM is known, so there is nothing to learn online."""
