"""Model-based agents for the causal-MBRL probe.

:class:`CertifiedPolicyAgent` is the confounding-robust causal agent for the M0 kill-gate. It ships
the highest-contrast deterministic policy whose improvement over the behavior policy is *certified*
robust to hidden confounding by :func:`causalrl.certify_policy` (Tan's marginal sensitivity model),
and abstains to the empirical behavior policy when nothing certifies. The certificate is the
decision rule — the honest robust planner, since a naive Manski-lower-bound greedy does not correct
a backdoor ``A <- U -> Y``. With ``alpha`` it also gates on the finite-sample conformal lower bound
(:func:`causalrl.conformal_action_value`), the agent-side entry point into the conformal layer.

Every planner here fits an interventional outcome model and every ``act`` **reads that model at the
observation**: the tabular agents look the action values up in the observed back-door stratum, the
function-approximation agent evaluates ``qhat(a, z)`` at the observed confounder, and the g-formula
agent evaluates its per-action T-learner at the observed covariate row. The marginal
``argmax_a E[Y | do(a)]`` is the answer only when the observation supplies no context — the arm that
wins on average need not win in any particular stratum, which is the whole point of a policy.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from causalrl.agents.base import BatchAgent
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


def _scalar(value: Any) -> Any:
    """Unwrap a 0-d numpy value to a plain Python scalar, so it hashes with a stratum key."""
    item = getattr(value, "item", None)
    return item() if callable(item) else value


def _context_key(
    observation: Mapping[str, Any], variables: tuple[str, ...]
) -> tuple[Any, ...] | None:
    """``observation``'s values for ``variables``, or ``None`` when it carries none of them.

    ``None`` means "no context observed", for which the marginal ``argmax_a E[Y | do(a)]`` is the
    right decision. A PARTIAL context raises instead: quietly marginalizing away a covariate the
    outcome model conditions on would answer a different query than the caller asked.
    """
    if not variables:
        return None
    present = tuple(v for v in variables if v in observation)
    if not present:
        return None
    if len(present) != len(variables):
        missing = sorted(set(variables) - set(present))
        raise KeyError(
            f"observation is missing {missing}: supply every conditioning variable "
            f"{list(variables)} for the contextual decision, or none of them for the marginal one"
        )
    return tuple(_scalar(observation[v]) for v in variables)


def _stratum_action_values(
    data: Mapping[str, np.ndarray],
    treatment: str,
    outcome: str,
    variables: tuple[str, ...],
    n_actions: int,
) -> dict[tuple[Any, ...], tuple[float, ...]]:
    """Cache ``Ehat[Y | A=a, Z=z]`` per action, for every stratum ``z`` of ``variables`` in
    ``data``.

    These are the same conditional means the back-door formula averages over — kept per stratum
    instead of summed away, which is what lets ``act`` condition on an observation. On a positivity
    gap (an action never played in a stratum) the stratum's marginal mean stands in, matching
    :func:`_backdoor_value`.
    """
    if not variables:
        return {}
    actions = np.asarray(data[treatment])
    y = np.asarray(data[outcome], dtype=float)
    strata = np.stack([np.asarray(data[v]) for v in variables], axis=1)
    table: dict[tuple[Any, ...], tuple[float, ...]] = {}
    for stratum in np.unique(strata, axis=0):
        in_z = np.all(strata == stratum, axis=1)
        marginal = float(y[in_z].mean())
        cells = [in_z & (actions == a) for a in range(n_actions)]
        table[tuple(_scalar(v) for v in stratum)] = tuple(
            float(y[cell].mean()) if cell.any() else marginal for cell in cells
        )
    return table


def _stratum_action(
    table: Mapping[tuple[Any, ...], tuple[float, ...]],
    observation: Mapping[str, Any],
    variables: tuple[str, ...],
    marginal_action: int,
) -> int:
    """``argmax_a Ehat[Y | A=a, Z=z]`` at the observed stratum.

    Falls back to ``marginal_action`` when there is no context to condition on: no ``variables``
    at all, none of them supplied, or a stratum that never appears in the logs.
    """
    key = _context_key(observation, variables)
    row = table.get(key) if key is not None else None
    if row is None:
        return int(marginal_action)
    return int(np.argmax(np.asarray(row)))


class CertifiedPolicyAgent(BatchAgent):
    """Ship the best deterministic policy whose improvement over behavior certifies robust to hidden
    confounding; abstain to the empirical behavior policy otherwise.

    Setting ``alpha`` adds the finite-sample layer: a candidate must also clear
    :func:`causalrl.certify_policy`'s conformal lower-confidence-bound gate, i.e. its calibrated
    worst-case return (:func:`causalrl.conformal_action_value`, weights = the propensity ratio
    ``pi/pi_behavior``) must be at least the behavior policy's. This is the safe-policy-improvement
    reading of the agent: a candidate that improves the *mean* but degrades the downside, or that
    has too little effective support to calibrate a bound at all, is refused. ``None`` (the
    default) runs the confounding layer alone.
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        *,
        gamma_max: float = 5.0,
        alpha: float | None = None,
    ) -> None:
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma_max = gamma_max
        self.alpha = alpha
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
            cert = certify_policy(
                dataset, target_actions, gamma_max=self.gamma_max, alpha=self.alpha
            )
            if cert.certified and cert.naive_contrast > best_contrast:
                best_contrast = cert.naive_contrast
                best_policy = list(candidate)
        self.policy = best_policy

    def act(self, observation: dict[str, Any]) -> int:
        return int(self.policy[int(observation["state"])])


class BackdoorAdjustedAgent(BatchAgent):
    """Active deconfounded optimizer: pick the action with the highest back-door-adjusted value
    ``E[Y | do(A=a)] = Σ_z P(z) · E[Y | A=a, Z=z]``, with the adjustment set read from the graph via
    :func:`~causalrl.backdoor_adjustment_set`.

    Unlike the certify-gated agent (whose ceiling is the behavior policy), this *recovers* the
    interventional optimum from confounded logs, given an observed admissible adjustment set. It is
    fitted on columnar data ``{treatment, outcome, *adjustment}`` (equal-length arrays).

    :meth:`fit` keeps the per-stratum conditional means, so :meth:`act` is a *policy* over the
    adjustment set — ``argmax_a Ehat[Y | do(a), Z=z]`` — not the one action that wins on average.
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
        self._stratum_values: dict[tuple[Any, ...], tuple[float, ...]] = {}

    def fit(self, data: Mapping[str, np.ndarray]) -> None:
        """Estimate each action's back-door-adjusted value, cache the per-stratum action values
        :meth:`act` conditions on, and record the marginal argmax."""
        self.values = [self._adjusted_value(a, data) for a in range(self.n_actions)]
        self._best_action = int(np.argmax(np.asarray(self.values)))
        self._stratum_values = _stratum_action_values(
            data, self.treatment, self.outcome, self.adjustment, self.n_actions
        )

    def _adjusted_value(self, action: int, data: Mapping[str, np.ndarray]) -> float:
        return _backdoor_value(action, data, self.treatment, self.outcome, self.adjustment)

    def act(self, observation: dict[str, Any]) -> int:
        """Contextual policy ``argmax_a Ehat[Y | do(A=a), Z=z]`` at the adjustment values in
        ``observation`` (supplied by name, e.g. ``{"Z": 1}``).

        Conditioning on the back-door set is what makes this a policy rather than a single action:
        the arm that wins *after* averaging over Z need not win *inside* a stratum. With none of
        the adjustment variables supplied — or an empty adjustment set, or a stratum never logged —
        the marginal decision ``argmax_a E[Y | do(a)]`` is returned, which is the right call when
        there is no context to condition on. A partial adjustment set raises.
        """
        return _stratum_action(
            self._stratum_values, observation, self.adjustment, self._best_action
        )


class TransportBackdoorAgent(BatchAgent):
    """Deconfound + transport: back-door-adjust for the confounders (source weighting) and reweight
    the selection variables by the target distribution, then ship the argmax of the transported
    interventional value ``E_target[Y | do(A=a)]``. The M2 upgrade of :class:`BackdoorAdjustedAgent`
    that carries a policy across a covariate shift where a correlational agent cannot.

    The back-door set comes from ``graph`` via :func:`~causalrl.backdoor_adjustment_set` (minus the
    declared ``transport`` variables); the estimand's transportability is confirmed up front with
    :func:`~causalrl.is_transportable_effect` (dogfooding the identification layer). Fitted on
    columnar source logs plus unlabeled target draws of the transport variables via :meth:`fit`.

    :meth:`act` is a *policy* over ``adjustment + transport``: the transported marginal can say
    "never play arm 1" while arm 1 still wins in a particular cell.
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
        self._context: tuple[str, ...] = self.adjustment + self.transport
        self._best_action = 0
        self.values: list[float] = [0.0] * n_actions
        self._stratum_values: dict[tuple[Any, ...], tuple[float, ...]] = {}

    def fit(
        self,
        source: Mapping[str, np.ndarray],
        *,
        target_covariates: Mapping[str, np.ndarray],
    ) -> None:
        """Estimate each action's transported interventional value, cache the per-cell action
        values :meth:`act` conditions on, and record the marginal argmax.

        ``source`` is columnar ``{treatment, outcome, *adjustment, *transport}``;
        ``target_covariates`` supplies unlabeled target draws of the ``transport`` variables.
        """
        self._stratum_values = _stratum_action_values(
            source, self.treatment, self.outcome, self._context, self.n_actions
        )
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
        """Contextual policy ``argmax_a E_target[Y | do(A=a), Z=z, W=w]`` at the adjustment and
        transport values in ``observation`` (supplied by name, e.g. ``{"Z": 1, "W": 0}``).

        The target reweighting that :meth:`fit` applies is a statement about *P(W)*, not about the
        outcome mechanism: under the S-admissibility this agent checks up front,
        ``E_target[Y | do(a), z, w] = E_source[Y | a, z, w]``, so the per-unit decision carries
        across the shift with no reweighting at all — only the marginal needs one. With none of the
        conditioning variables supplied, or a cell never logged in the source, the transported
        marginal decision is returned; a partial context raises.
        """
        return _stratum_action(self._stratum_values, observation, self._context, self._best_action)


class DiscoveryBackdoorAgent(BatchAgent):
    """Learns the structure: discovers the causal skeleton from data, orients it with the known
    temporal tier order (covariates precede treatment precede outcome — standard in DTR/medicine),
    takes the treatment's earlier-tier neighbours as the back-door set, then adjusts. The M1 upgrade
    of the handed-the-graph :class:`BackdoorAdjustedAgent`. Fitted with :meth:`discover_and_fit`.

    The *discovered* adjustment set is also the policy's context: :meth:`act` returns
    ``argmax_a Ehat[Y | do(a), Z=z]`` over the variables discovery selected."""

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
        self._stratum_values: dict[tuple[Any, ...], tuple[float, ...]] = {}

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
        self._stratum_values = _stratum_action_values(
            data, self.treatment, self.outcome, self.adjustment, self.n_actions
        )

    def act(self, observation: dict[str, Any]) -> int:
        """Contextual policy ``argmax_a Ehat[Y | do(A=a), Z=z]`` over the *discovered* adjustment
        set, at the values in ``observation`` (supplied by name, e.g. ``{"Z": 1}``).

        With none of the discovered variables supplied — or an empty discovered set, or a stratum
        never logged — the marginal decision ``argmax_a E[Y | do(a)]`` is returned; a partial
        context raises.
        """
        return _stratum_action(
            self._stratum_values, observation, self.adjustment, self._best_action
        )


def _rbf_features(z: np.ndarray, centers: np.ndarray, bandwidth: float) -> np.ndarray:
    """Design matrix ``[1, exp(-(z - c_k)^2 / (2 bandwidth^2))]`` for RBF ridge regression."""
    z = np.asarray(z, dtype=float).reshape(-1, 1)
    phi = np.exp(-((z - centers.reshape(1, -1)) ** 2) / (2.0 * bandwidth**2))
    return np.hstack([np.ones((z.shape[0], 1)), phi])


class FunctionApproxBackdoorAgent(BatchAgent):
    """Function-approximation back-door: fit ``qhat(a, z)`` by ridge regression on RBF features of a
    CONTINUOUS confounder, then back-door-adjust by Monte-Carlo integrating ``qhat(a, .)`` over the
    observed confounder sample, and ship the argmax. The M3 upgrade that carries the deconfounding
    recipe past the tabular regime to a learned continuous estimator.

    The single continuous confounder is read from ``graph`` via
    :func:`~causalrl.backdoor_adjustment_set`. Fit on columnar
    ``{confounder, treatment, outcome}`` via :meth:`fit`.

    The fitted ``qhat(a, .)`` is kept, so :meth:`act` evaluates it at the observed ``z`` and returns
    ``argmax_a qhat(a, z)`` — the same model the back-door integral averages, read per unit instead
    of summed away. A thin reward bump can make an arm interventionally *worse* on average and still
    optimal inside the bump; only a contextual policy can play it there.
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
        self._centers: np.ndarray = np.zeros(0)
        self._weights: list[np.ndarray | None] = [None] * n_actions

    def fit(self, data: Mapping[str, np.ndarray]) -> None:
        """Fit the per-action RBF-ridge outcome model, cache it for :meth:`act`, and select the
        back-door-adjusted (marginal) argmax."""
        z = np.asarray(data[self.confounder], dtype=float)
        a = np.asarray(data[self.treatment])
        y = np.asarray(data[self.outcome], dtype=float)
        self._centers = np.linspace(float(z.min()), float(z.max()), self.n_centers)
        self._weights = [
            self._fit_action_weights(action, z, a, y) for action in range(self.n_actions)
        ]
        # The back-door value is qhat(a, .) Monte-Carlo integrated over the observed confounder.
        self.values = list(self._action_values(_rbf_features(z, self._centers, self.bandwidth)))
        self._best_action = int(np.argmax(np.asarray(self.values)))

    def _fit_action_weights(
        self, action: int, z: np.ndarray, a: np.ndarray, y: np.ndarray
    ) -> np.ndarray | None:
        """Ridge weights of ``qhat(action, .)`` over the RBF design, fit on that action's rows
        (``None`` when the action is never played)."""
        mask = a == action
        if not mask.any():
            return None
        phi = _rbf_features(z[mask], self._centers, self.bandwidth)
        gram = phi.T @ phi + self.ridge * np.eye(phi.shape[1])
        return np.linalg.solve(gram, phi.T @ y[mask])

    def _action_values(self, phi: np.ndarray) -> list[float]:
        """Each action's mean ``qhat(a, .)`` over the design rows ``phi``.

        Over the whole confounder sample this is the back-door-adjusted value ``E[Y | do(a)]``;
        over a single row it is the per-unit ``qhat(a, z)``. A never-played action scores 0.0, as
        in the back-door formula.
        """
        return [0.0 if w is None else float((phi @ w).mean()) for w in self._weights]

    def act(self, observation: dict[str, Any]) -> int:
        """Contextual policy ``argmax_a qhat(a, z)`` at the confounder value in ``observation``
        (supplied by name, e.g. ``{"Z": 0.85}``).

        This is the fitted outcome model read at one point rather than integrated over the sample,
        so the arm an interventional *average* rejects can still be played where it wins. Without
        the confounder in ``observation`` (or before :meth:`fit`) the marginal decision
        ``argmax_a E[Y | do(a)]`` is returned.
        """
        if self._centers.size == 0 or self.confounder not in observation:
            return int(self._best_action)
        z = np.asarray([float(observation[self.confounder])], dtype=float)
        at_z = self._action_values(_rbf_features(z, self._centers, self.bandwidth))
        return int(np.argmax(np.asarray(at_z)))


def _standardization(fit_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Column mean and (floored) std of ``fit_x`` — the ridge conditioning, returned as statistics
    rather than applied in place, so a fitted model can rescale a NEW row exactly as it was fit."""
    return np.asarray(fit_x.mean(axis=0)), np.asarray(fit_x.std(axis=0)) + 1e-8


class _OutcomeModel(Protocol):
    """Duck type of a fitted per-action outcome model: sklearn-style ``predict(X) -> yhat``."""

    def predict(self, x: np.ndarray, /) -> Any: ...


@dataclass(frozen=True)
class _RidgeOutcomeModel:
    """One action's ridge fit over ``[1, standardized X]``, with the standardization it was fit
    under. Carrying the statistics is what makes the fit reusable at prediction time — the reason
    the g-formula agent can score an unseen covariate row instead of only its training sample."""

    weights: np.ndarray
    mean: np.ndarray
    std: np.ndarray

    def predict(self, x: np.ndarray, /) -> np.ndarray:
        return _ridge_design(x, self.mean, self.std) @ self.weights


def _ridge_design(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Design matrix ``[1, (x - mean) / std]`` for the standardized ridge outcome model."""
    standardized = (np.asarray(x, dtype=float) - mean) / std
    return np.hstack([np.ones((standardized.shape[0], 1)), standardized])


def _predict(model: _OutcomeModel | None, x: np.ndarray) -> np.ndarray | None:
    """``Ehat_a(x)`` for a fitted action model; ``None`` propagates (action never played)."""
    return None if model is None else np.asarray(model.predict(x), dtype=float)


def _mean_prediction(model: _OutcomeModel | None, x: np.ndarray) -> float:
    """``mean_i Ehat_a(x_i)`` — the g-formula standardization over a covariate sample, or the
    single-unit prediction when ``x`` is one row. 0.0 for a never-played action."""
    predictions = _predict(model, x)
    return float(predictions.mean()) if predictions is not None else 0.0


class GFormulaBackdoorAgent(BatchAgent):
    """Multivariate g-formula (standardization) back-door for many, mixed-type covariates.

    Fits a per-action outcome model ``Ehat_a[Y | X]`` over ALL covariates X (a T-learner), then
    standardizes ``E[Y | do(a)] = mean_i Ehat_a(X_i)`` and ships the argmax. This is the strategy
    for the realistic confounding of real datasets, where per-stratum adjustment degenerates (each
    stratum has ~1 row) and a single-confounder RBF is too narrow. The default outcome model is a
    numpy ridge over ``[1, standardized X]`` (dependency-free); pass ``outcome_model`` — a factory
    returning a fresh sklearn-style estimator (``fit``/``predict``) — for a flexible offline model.

    :meth:`fit` keeps the per-action models, so :meth:`act` is the *policy the T-learner already
    implies*: ``argmax_a Ehat_a(x)``, whose sign for a binary treatment is exactly the sign of
    :meth:`cate`. That is the same CATE-to-policy conversion
    :func:`~causalrl.interop.econml.policy_from_econml_cate` performs for a third-party estimator.
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
        self._models: list[_OutcomeModel | None] = [None] * n_actions

    def fit(self, data: Mapping[str, np.ndarray]) -> GFormulaBackdoorAgent:
        """Fit the per-action outcome models, keep them for the contextual :meth:`act`, standardize
        to ``E[Y|do(a)]``, and select the marginal argmax."""
        covariate_matrix = self._covariate_matrix(data)
        treatment = np.asarray(data[self.treatment])
        outcome = np.asarray(data[self.outcome], dtype=float)
        self._models = self._fit_action_models(covariate_matrix, treatment, outcome)
        self.values = [_mean_prediction(model, covariate_matrix) for model in self._models]
        self._best_action = int(np.argmax(np.asarray(self.values)))
        return self

    def _covariate_matrix(self, data: Mapping[str, np.ndarray]) -> np.ndarray:
        return np.column_stack([np.asarray(data[c], dtype=float) for c in self.covariates])

    def _fit_action_models(
        self, x: np.ndarray, treatment: np.ndarray, outcome: np.ndarray
    ) -> list[_OutcomeModel | None]:
        """Fit one outcome model per action on that action's rows (``None`` if never played)."""
        return [self._fit_action_model(a, x, treatment, outcome) for a in range(self.n_actions)]

    def _fit_action_model(
        self, action: int, x: np.ndarray, treatment: np.ndarray, outcome: np.ndarray
    ) -> _OutcomeModel | None:
        mask = treatment == action
        if not mask.any():
            return None
        x_a, y_a = x[mask], outcome[mask]
        if self._outcome_model is not None:
            model = self._outcome_model()
            model.fit(x_a, y_a)
            return model
        mean, std = _standardization(x_a)
        phi_a = _ridge_design(x_a, mean, std)
        gram = phi_a.T @ phi_a + self.ridge * np.eye(phi_a.shape[1])
        weights = np.linalg.solve(gram, phi_a.T @ y_a)
        return _RidgeOutcomeModel(weights=weights, mean=mean, std=std)

    def cate(self, data: Mapping[str, np.ndarray]) -> np.ndarray:
        """Per-unit CATE ``Ehat_1(X_i) - Ehat_0(X_i)`` for each row of ``data`` (the T-learner
        individual-effect estimate; binary treatment). ``data`` carries the covariates plus the
        observed ``treatment``/``outcome`` the per-action models are fit on."""
        if self.n_actions != 2:
            raise ValueError("cate is defined only for a binary treatment")
        x = self._covariate_matrix(data)
        treatment = np.asarray(data[self.treatment])
        outcome = np.asarray(data[self.outcome], dtype=float)
        models = self._fit_action_models(x, treatment, outcome)
        mu0, mu1 = _predict(models[0], x), _predict(models[1], x)
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
        """Contextual policy ``argmax_a Ehat_a(x)`` at the covariates in ``observation`` (supplied
        by name, e.g. ``{"age": 42, "smoke": 1}``).

        For a binary treatment this is exactly the sign of :meth:`cate` at that row — the per-unit
        decision the T-learner already estimates, shipped instead of discarded. With none of
        :attr:`covariates` in ``observation`` (or before :meth:`fit`) the marginal decision
        ``argmax_a E[Y | do(a)]`` is returned; a partial covariate vector raises rather than
        silently answering a different query.
        """
        key = _context_key(observation, self.covariates)
        if key is None or all(model is None for model in self._models):
            return int(self._best_action)
        x = np.asarray([[float(value) for value in key]], dtype=float)
        return int(np.argmax(np.asarray([_mean_prediction(m, x) for m in self._models])))
