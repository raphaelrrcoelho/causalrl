"""Model-based agents for the causal-MBRL probe.

:class:`CertifiedPolicyAgent` is the confounding-robust causal agent for the M0 kill-gate. It ships
the highest-contrast deterministic policy whose improvement over the behavior policy is *certified*
robust to hidden confounding by :func:`causalrl.certify_policy` (Tan's marginal sensitivity model),
and abstains to the empirical behavior policy when nothing certifies. The certificate is the
decision rule — the honest robust planner, since a naive Manski-lower-bound greedy does not correct
a backdoor ``A <- U -> Y``.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from causalrl.agents.base import Agent
from causalrl.data.dataset import ConfoundedTrajectoryDataset
from causalrl.discovery import discover
from causalrl.identification.criteria import backdoor_adjustment_set
from causalrl.identification.id_algorithm import is_transportable_effect
from causalrl.scale import certify_policy
from causalrl.scm.graph import CausalGraph


def _backdoor_value(
    action: int,
    data: Mapping[str, np.ndarray],
    treatment: str,
    outcome: str,
    adjustment: tuple[str, ...],
) -> float:
    """Back-door formula: sum over adjustment strata of P(z) * E[Y | treatment=action, z]."""
    a = np.asarray(data[treatment])
    y = np.asarray(data[outcome], dtype=float)
    if not adjustment:
        sel = y[a == action]
        return float(sel.mean()) if sel.size else 0.0
    strata = np.stack([np.asarray(data[z]) for z in adjustment], axis=1)
    total = 0.0
    for stratum in np.unique(strata, axis=0):
        in_z = np.all(strata == stratum, axis=1)
        p_z = float(in_z.mean())
        in_az = in_z & (a == action)
        # On a positivity gap (action unseen in a stratum) fall back to the stratum-marginal mean.
        outcome_mean = float(y[in_az].mean()) if in_az.any() else float(y[in_z].mean())
        total += p_z * outcome_mean
    return total


def _match(
    data: Mapping[str, np.ndarray], variables: tuple[str, ...], combo: tuple[object, ...]
) -> np.ndarray:
    """Boolean mask over ``data`` where every ``variables[i] == combo[i]`` (all-true if empty)."""
    length = len(next(iter(data.values())))
    mask = np.ones(length, dtype=bool)
    for variable, value in zip(variables, combo, strict=True):
        mask &= np.asarray(data[variable]) == value
    return mask


def _transport_value(
    action: int,
    source: Mapping[str, np.ndarray],
    treatment: str,
    outcome: str,
    adjust: tuple[str, ...],
    transport: tuple[str, ...],
    target_covariates: Mapping[str, np.ndarray],
) -> float:
    """Transport formula ``Σ_s Σ_t P_src(s) · P_tgt(t) · E_src[Y | action, s, t]``.

    ``adjust`` are back-door confounders weighted by the SOURCE marginal; ``transport`` are the
    selection variables weighted by the TARGET marginal (from ``target_covariates``). Assumes the
    adjust and transport blocks are independent (the transportable-by-S-admissible-adjustment case).
    """
    a = np.asarray(source[treatment])
    y = np.asarray(source[outcome], dtype=float)
    adj_levels = [np.unique(np.asarray(source[v])) for v in adjust]
    tr_levels = [np.unique(np.asarray(target_covariates[v])) for v in transport]
    total = 0.0
    for adj_combo in itertools.product(*adj_levels):
        src_adj = _match(source, adjust, adj_combo)
        p_adj = float(src_adj.mean())
        if p_adj == 0.0:
            continue
        for tr_combo in itertools.product(*tr_levels):
            p_tr = float(_match(target_covariates, transport, tr_combo).mean())
            if p_tr == 0.0:
                continue
            cell = src_adj & _match(source, transport, tr_combo) & (a == action)
            if cell.any():
                mean_y = float(y[cell].mean())
            else:
                # Positivity gap: fall back to the stratum mean ignoring the action.
                stratum = src_adj & _match(source, transport, tr_combo)
                mean_y = float(y[stratum].mean()) if stratum.any() else 0.0
            total += p_adj * p_tr * mean_y
    return total


class CertifiedPolicyAgent(Agent):
    """Ship the best deterministic policy whose improvement over behavior certifies robust to hidden
    confounding; abstain to the empirical behavior policy otherwise."""

    def __init__(self, n_states: int, n_actions: int, *, gamma_max: float = 5.0) -> None:
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma_max = gamma_max
        self.policy: list[int] = [0] * n_states

    def _behavior_policy(self, dataset: ConfoundedTrajectoryDataset) -> list[int]:
        """Empirical behavior policy: the most-logged action per state (abstention target)."""
        return [
            max(range(self.n_actions), key=lambda a: dataset.behavior_propensity(s, a))
            for s in range(self.n_states)
        ]

    def ingest_offline(self, dataset: ConfoundedTrajectoryDataset) -> None:
        transitions = dataset.transitions
        best_policy = self._behavior_policy(dataset)  # abstention default
        best_contrast = 0.0
        for candidate in itertools.product(range(self.n_actions), repeat=self.n_states):
            # Skip a policy that assigns a never-logged action in some state: an unseen action's
            # value is not identified from the logs, and certify_policy has no support to bound it.
            if any(
                dataset.behavior_propensity(s, candidate[s]) == 0.0 for s in range(self.n_states)
            ):
                continue
            target_actions = [candidate[tr.state] for tr in transitions]
            cert = certify_policy(dataset, target_actions, gamma_max=self.gamma_max)
            if cert.certified and cert.naive_contrast > best_contrast:
                best_contrast = cert.naive_contrast
                best_policy = list(candidate)
        self.policy = best_policy

    def act(self, observation: dict[str, Any]) -> int:
        return int(self.policy[int(observation["state"])])

    def update(self, observation: dict[str, Any], action: int, reward: float) -> None:
        """Fixed policy from the logs; no online update."""


class BackdoorAdjustedAgent(Agent):
    """Active deconfounded optimizer: pick the action with the highest back-door-adjusted value
    ``E[Y | do(A=a)] = Σ_z P(z) · E[Y | A=a, Z=z]``, with the adjustment set read from the graph via
    :func:`~causalrl.backdoor_adjustment_set`.

    Unlike the certify-gated agent (whose ceiling is the behavior policy), this *recovers* the
    interventional optimum from confounded logs, given an observed admissible adjustment set. It is
    fitted on columnar data ``{treatment, outcome, *adjustment}`` (equal-length arrays).
    """

    def __init__(
        self,
        n_actions: int,
        *,
        graph: CausalGraph,
        treatment: str = "A",
        outcome: str = "Y",
    ) -> None:
        self.n_actions = n_actions
        self.treatment = treatment
        self.outcome = outcome
        self.adjustment: tuple[str, ...] = tuple(
            sorted(backdoor_adjustment_set(graph, treatment, outcome))
        )
        self._best_action = 0
        self.values: list[float] = [0.0] * n_actions

    def fit(self, data: Mapping[str, np.ndarray]) -> None:
        """Estimate each action's back-door-adjusted value and select the argmax."""
        self.values = [self._adjusted_value(a, data) for a in range(self.n_actions)]
        self._best_action = int(np.argmax(np.asarray(self.values)))

    def _adjusted_value(self, action: int, data: Mapping[str, np.ndarray]) -> float:
        return _backdoor_value(action, data, self.treatment, self.outcome, self.adjustment)

    def act(self, observation: dict[str, Any]) -> int:
        return int(self._best_action)

    def update(self, observation: dict[str, Any], action: int, reward: float) -> None:
        """Fixed action from the fitted adjustment; no online update."""


class TransportBackdoorAgent(Agent):
    """Deconfound + transport: back-door-adjust for the confounders (source weighting) and reweight
    the selection variables by the target distribution, then ship the argmax of the transported
    interventional value ``E_target[Y | do(A=a)]``. The M2 upgrade of :class:`BackdoorAdjustedAgent`
    that carries a policy across a covariate shift where a correlational agent cannot.

    The back-door set comes from ``graph`` via :func:`~causalrl.backdoor_adjustment_set` (minus the
    declared ``transport`` variables); the estimand's transportability is confirmed up front with
    :func:`~causalrl.is_transportable_effect` (dogfooding the identification layer). Fitted on
    columnar source logs plus unlabeled target draws of the transport variables via :meth:`fit`.
    """

    def __init__(
        self,
        n_actions: int,
        *,
        graph: CausalGraph,
        treatment: str = "A",
        outcome: str = "Y",
        transport: Sequence[str] = ("W",),
    ) -> None:
        self.n_actions = n_actions
        self.treatment = treatment
        self.outcome = outcome
        self.transport = tuple(transport)
        adjust = set(backdoor_adjustment_set(graph, treatment, outcome)) - set(self.transport)
        self.adjustment: tuple[str, ...] = tuple(sorted(adjust))
        self.transportable = is_transportable_effect(
            graph, {treatment}, {outcome}, set(self.transport)
        )
        self._best_action = 0
        self.values: list[float] = [0.0] * n_actions

    def fit(
        self,
        source: Mapping[str, np.ndarray],
        *,
        target_covariates: Mapping[str, np.ndarray],
    ) -> None:
        """Estimate each action's transported interventional value and select the argmax.

        ``source`` is columnar ``{treatment, outcome, *adjustment, *transport}``;
        ``target_covariates`` supplies unlabeled target draws of the ``transport`` variables.
        """
        self.values = [
            _transport_value(
                a,
                source,
                self.treatment,
                self.outcome,
                self.adjustment,
                self.transport,
                target_covariates,
            )
            for a in range(self.n_actions)
        ]
        self._best_action = int(np.argmax(np.asarray(self.values)))

    def act(self, observation: dict[str, Any]) -> int:
        return int(self._best_action)

    def update(self, observation: dict[str, Any], action: int, reward: float) -> None:
        """Fixed action from the fitted transport estimate; no online update."""


class DiscoveryBackdoorAgent(Agent):
    """Learns the structure: discovers the causal skeleton from data, orients it with the known
    temporal tier order (covariates precede treatment precede outcome — standard in DTR/medicine),
    takes the treatment's earlier-tier neighbours as the back-door set, then adjusts. The M1 upgrade
    of the handed-the-graph :class:`BackdoorAdjustedAgent`. Fitted with :meth:`discover_and_fit`."""

    def __init__(
        self,
        n_actions: int,
        *,
        variables: Sequence[str],
        treatment: str = "A",
        outcome: str = "Y",
    ) -> None:
        self.n_actions = n_actions
        self.variables = tuple(variables)
        self.treatment = treatment
        self.outcome = outcome
        self.adjustment: tuple[str, ...] = ()
        self._best_action = 0
        self.values: list[float] = [0.0] * n_actions

    def discover_and_fit(
        self,
        data: Mapping[str, np.ndarray],
        *,
        tiers: Sequence[Sequence[str]],
        threshold: float = 0.01,
    ) -> None:
        """Discover the skeleton, orient it by the temporal ``tiers`` (ordered groups, earliest
        first), take the treatment's earlier-tier neighbours as the back-door set, and adjust.

        Temporal tiering is standard domain knowledge in DTR / medicine and makes adjustment-set
        recovery reliable (pure interventional edge-orientation was not, on this graph).
        """
        cpdag = discover(data, self.variables, threshold=threshold)
        neighbours: set[str] = set()
        for a, b in cpdag.directed_edges:
            if self.treatment in (a, b):
                neighbours.update({a, b} - {self.treatment})
        for edge in cpdag.undirected_edges:
            if self.treatment in edge:
                neighbours.update(v for v in edge if v != self.treatment)
        tier_of = {v: i for i, group in enumerate(tiers) for v in group}
        adj = [v for v in neighbours if tier_of[v] < tier_of[self.treatment]]
        self.adjustment = tuple(sorted(adj))
        self.values = [
            _backdoor_value(a, data, self.treatment, self.outcome, self.adjustment)
            for a in range(self.n_actions)
        ]
        self._best_action = int(np.argmax(np.asarray(self.values)))

    def act(self, observation: dict[str, Any]) -> int:
        return int(self._best_action)

    def update(self, observation: dict[str, Any], action: int, reward: float) -> None:
        """Fixed action from the fitted adjustment; no online update."""


def _rbf_features(z: np.ndarray, centers: np.ndarray, bandwidth: float) -> np.ndarray:
    """Design matrix ``[1, exp(-(z - c_k)^2 / (2 bandwidth^2))]`` for RBF ridge regression."""
    z = np.asarray(z, dtype=float).reshape(-1, 1)
    phi = np.exp(-((z - centers.reshape(1, -1)) ** 2) / (2.0 * bandwidth**2))
    return np.hstack([np.ones((z.shape[0], 1)), phi])


class FunctionApproxBackdoorAgent(Agent):
    """Function-approximation back-door: fit ``qhat(a, z)`` by ridge regression on RBF features of a
    CONTINUOUS confounder, then back-door-adjust by Monte-Carlo integrating ``qhat(a, .)`` over the
    observed confounder sample, and ship the argmax. The M3 upgrade that carries the deconfounding
    recipe past the tabular regime to a learned continuous estimator.

    The single continuous confounder is read from ``graph`` via
    :func:`~causalrl.backdoor_adjustment_set`. Fit on columnar
    ``{confounder, treatment, outcome}`` via :meth:`fit`.
    """

    def __init__(
        self,
        n_actions: int,
        *,
        graph: CausalGraph,
        treatment: str = "A",
        outcome: str = "Y",
        n_centers: int = 12,
        bandwidth: float = 0.08,
        ridge: float = 1e-2,
    ) -> None:
        adjustment = sorted(backdoor_adjustment_set(graph, treatment, outcome))
        if len(adjustment) != 1:
            raise ValueError(
                "FunctionApproxBackdoorAgent supports a single continuous confounder; "
                f"graph gives adjustment set {adjustment}"
            )
        self.n_actions = n_actions
        self.treatment = treatment
        self.outcome = outcome
        self.confounder = adjustment[0]
        self.n_centers = n_centers
        self.bandwidth = bandwidth
        self.ridge = ridge
        self._best_action = 0
        self.values: list[float] = [0.0] * n_actions

    def fit(self, data: Mapping[str, np.ndarray]) -> None:
        """Fit the per-action RBF-ridge outcome model and select the back-door-adjusted argmax."""
        z = np.asarray(data[self.confounder], dtype=float)
        a = np.asarray(data[self.treatment])
        y = np.asarray(data[self.outcome], dtype=float)
        centers = np.linspace(float(z.min()), float(z.max()), self.n_centers)
        phi_all = _rbf_features(z, centers, self.bandwidth)
        self.values = [
            self._adjusted_value(action, z, a, y, centers, phi_all)
            for action in range(self.n_actions)
        ]
        self._best_action = int(np.argmax(np.asarray(self.values)))

    def _adjusted_value(
        self,
        action: int,
        z: np.ndarray,
        a: np.ndarray,
        y: np.ndarray,
        centers: np.ndarray,
        phi_all: np.ndarray,
    ) -> float:
        """Back-door value: mean over Z of ``qhat(action, .)`` fit on that action's rows."""
        mask = a == action
        if not mask.any():
            return 0.0
        phi = _rbf_features(z[mask], centers, self.bandwidth)
        gram = phi.T @ phi + self.ridge * np.eye(phi.shape[1])
        weights = np.linalg.solve(gram, phi.T @ y[mask])
        return float((phi_all @ weights).mean())

    def act(self, observation: dict[str, Any]) -> int:
        return int(self._best_action)

    def update(self, observation: dict[str, Any], action: int, reward: float) -> None:
        """Fixed action from the fitted function approximator; no online update."""


def _standardize(fit_x: np.ndarray, apply_x: np.ndarray) -> np.ndarray:
    """Z-score ``apply_x`` by the column mean/std of ``fit_x`` (ridge conditioning)."""
    mean = fit_x.mean(axis=0)
    std = fit_x.std(axis=0) + 1e-8
    return (apply_x - mean) / std


class GFormulaBackdoorAgent(Agent):
    """Multivariate g-formula (standardization) back-door for many, mixed-type covariates.

    Fits a per-action outcome model ``Ehat_a[Y | X]`` over ALL covariates X (a T-learner), then
    standardizes ``E[Y | do(a)] = mean_i Ehat_a(X_i)`` and ships the argmax. This is the strategy
    for the realistic confounding of real datasets, where per-stratum adjustment degenerates (each
    stratum has ~1 row) and a single-confounder RBF is too narrow. The default outcome model is a
    numpy ridge over ``[1, standardized X]`` (dependency-free); pass ``outcome_model`` — a factory
    returning a fresh sklearn-style estimator (``fit``/``predict``) — for a flexible offline model.
    """

    def __init__(
        self,
        n_actions: int,
        *,
        covariates: Sequence[str],
        treatment: str = "A",
        outcome: str = "Y",
        outcome_model: Callable[[], Any] | None = None,
        ridge: float = 1.0,
    ) -> None:
        self.n_actions = n_actions
        self.covariates = tuple(covariates)
        self.treatment = treatment
        self.outcome = outcome
        self._outcome_model = outcome_model
        self.ridge = ridge
        self._best_action = 0
        self.values: list[float] = [0.0] * n_actions

    def fit(self, data: Mapping[str, np.ndarray]) -> GFormulaBackdoorAgent:
        """Fit the per-action outcome model, standardize to ``E[Y|do(a)]``, select the argmax."""
        covariate_matrix = np.column_stack(
            [np.asarray(data[c], dtype=float) for c in self.covariates]
        )
        treatment = np.asarray(data[self.treatment])
        outcome = np.asarray(data[self.outcome], dtype=float)
        self.values = [
            self._standardized_value(action, covariate_matrix, treatment, outcome)
            for action in range(self.n_actions)
        ]
        self._best_action = int(np.argmax(np.asarray(self.values)))
        return self

    def _action_predictions(
        self, action: int, x: np.ndarray, treatment: np.ndarray, outcome: np.ndarray
    ) -> np.ndarray | None:
        """Fit action's outcome model on its rows; predict on all rows (None if never played)."""
        mask = treatment == action
        if not mask.any():
            return None
        x_a, y_a = x[mask], outcome[mask]
        if self._outcome_model is not None:
            model = self._outcome_model()
            model.fit(x_a, y_a)
            return np.asarray(model.predict(x), dtype=float)
        phi_a = self._design(x_a, x_a)
        gram = phi_a.T @ phi_a + self.ridge * np.eye(phi_a.shape[1])
        weights = np.linalg.solve(gram, phi_a.T @ y_a)
        return self._design(x_a, x) @ weights

    def _standardized_value(
        self, action: int, x: np.ndarray, treatment: np.ndarray, outcome: np.ndarray
    ) -> float:
        predictions = self._action_predictions(action, x, treatment, outcome)
        return float(predictions.mean()) if predictions is not None else 0.0

    def _design(self, fit_x: np.ndarray, apply_x: np.ndarray) -> np.ndarray:
        standardized = _standardize(fit_x, apply_x)
        return np.hstack([np.ones((standardized.shape[0], 1)), standardized])

    def cate(self, data: Mapping[str, np.ndarray]) -> np.ndarray:
        """Per-unit CATE ``Ehat_1(X_i) - Ehat_0(X_i)`` for each row of ``data`` (the T-learner
        individual-effect estimate; binary treatment). ``data`` carries the covariates plus the
        observed ``treatment``/``outcome`` the per-action models are fit on."""
        if self.n_actions != 2:
            raise ValueError("cate is defined only for a binary treatment")
        x = np.column_stack([np.asarray(data[c], dtype=float) for c in self.covariates])
        treatment = np.asarray(data[self.treatment])
        outcome = np.asarray(data[self.outcome], dtype=float)
        mu0 = self._action_predictions(0, x, treatment, outcome)
        mu1 = self._action_predictions(1, x, treatment, outcome)
        if mu0 is None or mu1 is None:
            raise ValueError("both treatment arms must appear in the data to estimate CATE")
        return mu1 - mu0

    @property
    def contrast(self) -> float:
        """``E[Y|do(1)] - E[Y|do(0)]`` (the ATE) for a binary treatment."""
        if self.n_actions != 2:
            raise ValueError("contrast is defined only for a binary treatment")
        return self.values[1] - self.values[0]

    def act(self, observation: dict[str, Any]) -> int:
        return int(self._best_action)

    def update(self, observation: dict[str, Any], action: int, reward: float) -> None:
        """Fixed action from the fitted g-formula; no online update."""
