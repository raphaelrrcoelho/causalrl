"""CausalMBRLAgent: one front-door over causalrl's causal model-based planners.

Describe the offline, confounded decision problem you have; the agent routes to the right planner
(discover / back-door / transport / function-approx / deconfounded sequential) and gives you a
uniform ``fit`` → ``act`` surface plus the identification provenance it relied on. This is the
DESIGN's discover → plan-under-confounding → transport agent behind a single class; each specialized
planner remains available directly for full control.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from causalrl.agents.mbrl import (
    BackdoorAdjustedAgent,
    DiscoveryBackdoorAgent,
    FunctionApproxBackdoorAgent,
    TransportBackdoorAgent,
)
from causalrl.scm.graph import CausalGraph


class CausalMBRLAgent:
    """Route an offline confounded decision problem to the right causal planner, behind fit/act.

    Routing (first match wins):

    ============================  ============================================================
    You provide                    Strategy (planner)
    ============================  ============================================================
    ``horizon=...``                ``"sequential"`` — deconfounded value iteration (``DOVI``)
    ``transport=(...)``            ``"transport"`` — deconfound + transport by the target ``P``
    ``continuous_confounder``      ``"function_approx"`` — ridge-RBF back-door on a continuous Z
    ``graph=None``                 ``"discovery"`` — skeleton discovery + temporal tiers, adjust
    otherwise (``graph`` given)    ``"backdoor"`` — back-door adjust an observed confounder
    ============================  ============================================================

    Then ``fit`` on offline data and ``act`` on an observation. ``explain()`` reports the chosen
    strategy and the identification basis (adjustment set / transportability / confounder) — the
    trust artifact for the identifiable regime this agent targets. It beats correlational offline
    RL only there (confounded / offline / transfer); it is not a clean-benchmark performer.
    """

    def __init__(
        self,
        n_actions: int,
        *,
        graph: CausalGraph | None = None,
        variables: Sequence[str] | None = None,
        tiers: Sequence[Sequence[str]] | None = None,
        treatment: str = "A",
        outcome: str = "Y",
        transport: Sequence[str] | None = None,
        continuous_confounder: bool = False,
        horizon: int | None = None,
        n_states: int | None = None,
        seed: int = 0,
    ) -> None:
        self.n_actions = n_actions
        self._tiers = tiers
        self._planner: Any
        if horizon is not None:
            if n_states is None:
                raise ValueError("sequential mode (horizon set) requires n_states")
            self.strategy = "sequential"
            from causalrl.agents.dovi import DOVI  # lazy: keep the front door torch-light otherwise

            self._planner = DOVI(
                n_states=n_states,
                n_actions=n_actions,
                horizon=horizon,
                seed=seed,
                transition_assumption="unconfounded",
            )
        elif transport is not None:
            if graph is None:
                raise ValueError("transport mode requires a graph")
            self.strategy = "transport"
            self._planner = TransportBackdoorAgent(
                n_actions, graph=graph, treatment=treatment, outcome=outcome, transport=transport
            )
        elif continuous_confounder:
            if graph is None:
                raise ValueError("continuous_confounder mode requires a graph")
            self.strategy = "function_approx"
            self._planner = FunctionApproxBackdoorAgent(
                n_actions, graph=graph, treatment=treatment, outcome=outcome
            )
        elif graph is None:
            if variables is None or tiers is None:
                raise ValueError("discovery mode (no graph) requires variables and tiers")
            self.strategy = "discovery"
            self._planner = DiscoveryBackdoorAgent(
                n_actions, variables=variables, treatment=treatment, outcome=outcome
            )
        else:
            self.strategy = "backdoor"
            self._planner = BackdoorAdjustedAgent(
                n_actions, graph=graph, treatment=treatment, outcome=outcome
            )

    def fit(
        self,
        data: Any,
        *,
        target_covariates: Mapping[str, np.ndarray] | None = None,
    ) -> CausalMBRLAgent:
        """Fit the routed planner on offline ``data`` and return ``self``.

        ``data`` is columnar ``{treatment, outcome, *covariates}`` for every strategy except
        ``"sequential"``, which takes a ``ConfoundedTrajectoryDataset`` of trajectory logs.
        ``target_covariates`` (target-domain draws of the transport variables) is required for the
        ``"transport"`` strategy.
        """
        if self.strategy == "sequential":
            self._planner.ingest_offline(data)
        elif self.strategy == "transport":
            if target_covariates is None:
                raise ValueError("transport strategy requires target_covariates")
            self._planner.fit(data, target_covariates=target_covariates)
        elif self.strategy == "discovery":
            if self._tiers is None:
                raise ValueError("discovery strategy requires tiers")
            self._planner.discover_and_fit(data, tiers=self._tiers)
        else:
            self._planner.fit(data)
        return self

    def act(self, observation: dict[str, Any]) -> int:
        """Return the planner's chosen action for ``observation`` (e.g. ``{"state": 0}``)."""
        return int(self._planner.act(observation))

    def update(self, observation: dict[str, Any], action: int, reward: float) -> None:
        """Offline agents hold a fixed policy after ``fit``; no online update."""

    @property
    def planner(self) -> Any:
        """The underlying specialized planner, for full control or inspection."""
        return self._planner

    @property
    def adjustment_set(self) -> tuple[str, ...] | None:
        """The back-door adjustment set the planner used, when the strategy exposes one."""
        adjustment = getattr(self._planner, "adjustment", None)
        return tuple(adjustment) if adjustment is not None else None

    def explain(self) -> str:
        """Human-readable provenance: the strategy and the identification basis it relied on."""
        parts = [f"strategy={self.strategy}"]
        adjustment = self.adjustment_set
        if adjustment is not None:
            parts.append("adjustment_set={" + ", ".join(adjustment) + "}")
        if self.strategy == "transport":
            parts.append(f"transportable={getattr(self._planner, 'transportable', None)}")
        if self.strategy == "function_approx":
            parts.append(f"confounder={getattr(self._planner, 'confounder', None)}")
        return "CausalMBRLAgent(" + ", ".join(parts) + ")"
