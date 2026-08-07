"""Envelope propagation over state features — the continuous-state counterpart of DOVI's bound.

:class:`causalrl.agents.fitted.FittedQIteration` gave up the causal bound to go continuous: it caps
optimism globally by horizon and reward range and reports ``EMPIRICAL``. This class gets the bound
back. The two ingredients that were missing are now available separately:

* :class:`causalrl.bounds.functional.FunctionalManskiBounds` bounds the *immediate* reward at any
  feature point, generalising the per-cell Manski bound; and
* the conditional expectation of the *continuation* value is itself a regression, which is what
  fitted iteration already estimates.

Composing them propagates an interval rather than a point. With ``L`` and ``U`` the lower and upper
envelopes and ``H`` the horizon::

    L_{H+1} = U_{H+1} = 0
    U_h(x, a) = upper_reward(x, a) + E[ max_a' U_{h+1}(X', a') | X = x, A = a ]
    L_h(x, a) = lower_reward(x, a) + E[ max_a' L_{h+1}(X', a') | X = x, A = a ]

The recursion preserves the envelope because both ``max`` and conditional expectation are monotone:
if ``L_{h+1} <= Q_{h+1} <= U_{h+1}`` pointwise then the same holds at ``h``. No explicit transition
tensor or generative model appears — the successor distribution enters only through an expectation,
which is estimated by regressing the realised continuation value on the current features.

**The transition caveat is the same one DOVI carries, and it is load-bearing.** That expectation is
taken over the *logged* successor distribution. If the behaviour policy's hidden confounder also
drives the dynamics, the logged successors are not the interventional ones and the propagated
interval is not a bound on anything. :class:`causalrl.DOVI` gates this with
``transition_assumption``; so does this class, with the same two values and the same refusal, and
its certificate degrades to ``EMPIRICAL`` whenever the gate is opened with ``allow_heuristic``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from causalrl.agents.base import BatchAgent
from causalrl.agents.dovi import TransitionAssumption
from causalrl.bounds.functional import FunctionalManskiBounds
from causalrl.certify.certificate import (
    Assumption,
    Certificate,
    EstimandSpec,
    Hedge,
    Kind,
    Provenance,
)
from causalrl.estimate.nuisance import Regressor, RidgeRegressor
from causalrl.exceptions import UnverifiedAssumptionError
from causalrl.identification.bounds import Interval
from causalrl.state import FeatureTransition, FloatArray, StateEncoder, unpack_transitions

__all__ = ["BoundedFittedQIteration"]


class BoundedFittedQIteration(BatchAgent):
    """Finite-horizon envelope propagation over encoded features, under Manski reward bounds.

    ``encoder`` decides what a state is; ``bounds_model`` supplies the immediate-reward interval at
    each feature point (default :class:`~causalrl.bounds.functional.FunctionalManskiBounds` with the
    dependency-free nuisances); ``regressor`` estimates the continuation expectation.

    As with :class:`~causalrl.agents.fitted.FittedQIteration`, the tabular case is contained rather
    than replaced: with :class:`~causalrl.state.OneHotEncoder` the reward bounds reduce to
    :func:`causalrl.causal_q_bounds` per cell, so the envelope reduces to the interval DOVI's
    ceiling caps.

    ``transition_assumption`` mirrors :class:`causalrl.DOVI`. At ``"unknown"`` with a horizon above
    1, :meth:`fit` raises :class:`~causalrl.exceptions.UnverifiedAssumptionError` unless
    ``allow_heuristic=True`` — propagating an interval through a confounded successor distribution
    produces a number that is not a bound, and running it silently would be the one failure mode
    this class exists to prevent.
    """

    def __init__(
        self,
        n_actions: int,
        horizon: int,
        encoder: StateEncoder,
        *,
        bounds_model: FunctionalManskiBounds | None = None,
        regressor: Callable[[], Regressor] | None = None,
        reward_range: tuple[float, float] = (0.0, 1.0),
        transition_assumption: TransitionAssumption = "unknown",
        allow_heuristic: bool = False,
        seed: int | None = None,
    ) -> None:
        if n_actions < 1:
            raise ValueError(f"n_actions={n_actions} must be at least 1")
        if horizon < 1:
            raise ValueError(f"horizon={horizon} must be at least 1")
        if transition_assumption not in ("unknown", "unconfounded"):
            raise ValueError(
                "transition_assumption must be 'unknown' or 'unconfounded', "
                f"got {transition_assumption!r}"
            )
        self.n_actions = n_actions
        self.horizon = horizon
        self.encoder = encoder
        self.reward_range = (float(reward_range[0]), float(reward_range[1]))
        self.transition_assumption = transition_assumption
        self.allow_heuristic = allow_heuristic
        self._bounds_model = (
            bounds_model
            if bounds_model is not None
            else FunctionalManskiBounds(n_actions, reward_range=self.reward_range)
        )
        self._make_regressor = regressor if regressor is not None else RidgeRegressor
        self._buffer: list[FeatureTransition] = []
        self._lower_cont: dict[tuple[int, int], Regressor] = {}
        self._upper_cont: dict[tuple[int, int], Regressor] = {}
        self._fitted = False
        self._rng = np.random.default_rng(seed)

    @property
    def is_certified(self) -> bool:
        """Whether the propagated interval is a bound rather than heuristic value propagation.

        True at horizon 1 (nothing is propagated, so only the reward bound is used) or when the
        caller asserts unconfounded transitions. Mirrors :attr:`causalrl.DOVI.is_certified`.

        Note this is a statement about the *transition* assumption only. Even when it is ``True``,
        the interval still rests on the outcome and propensity models being correctly specified —
        an assumption the tabular bound does not need, and one every certificate here records.
        """
        return self.horizon <= 1 or self.transition_assumption == "unconfounded"

    def fit(
        self, transitions: Sequence[FeatureTransition] | None = None
    ) -> BoundedFittedQIteration:
        """Fit the reward bounds and propagate the envelope back from the horizon.

        Passing ``transitions`` replaces the observed buffer, matching
        :meth:`causalrl.FittedQIteration.fit`; omitting it refits from whatever
        :meth:`observe_step` has accumulated.
        """
        if not self.is_certified and not self.allow_heuristic:
            raise UnverifiedAssumptionError(
                "multi-step envelope propagation requires transition_assumption='unconfounded': "
                "the continuation expectation is taken over the LOGGED successor distribution, so "
                "if the behaviour policy's hidden confounder also drives the dynamics the "
                "propagated interval bounds nothing. Set allow_heuristic=True to run it anyway as "
                "value propagation (the certificate then reports EMPIRICAL, not BOUNDED)."
            )
        if transitions is not None:
            self._buffer = list(transitions)
        if not self._buffer:
            raise ValueError(
                "BoundedFittedQIteration.fit needs transitions: pass them explicitly or feed "
                "them in through observe_step first. There is no envelope over an empty log."
            )
        batch = unpack_transitions(self._buffer, encoder=self.encoder, n_actions=self.n_actions)
        states, next_states = batch.states, batch.next_states
        actions, rewards, not_done = batch.actions, batch.rewards, batch.not_done

        self._bounds_model.fit(states, actions, rewards)
        reward_low_next, reward_high_next = self._bounds_model.bounds(next_states)

        self._lower_cont = {}
        self._upper_cont = {}
        # Continuation value of the step above, evaluated at each row's successor. Zero at H+1.
        lower_at_successor = np.zeros(len(self._buffer))
        upper_at_successor = np.zeros(len(self._buffer))
        for step in range(self.horizon, 0, -1):
            lower_target = not_done * lower_at_successor
            upper_target = not_done * upper_at_successor
            for action in range(self.n_actions):
                rows = actions == action
                if not rows.any():
                    continue  # no data: the continuation defaults to 0 for this (step, action)
                self._lower_cont[(step, action)] = self._make_regressor().fit(
                    states[rows], lower_target[rows]
                )
                self._upper_cont[(step, action)] = self._make_regressor().fit(
                    states[rows], upper_target[rows]
                )
            low, high = self._envelope(next_states, step, reward_low_next, reward_high_next)
            lower_at_successor = low.max(axis=1)
            upper_at_successor = high.max(axis=1)
        self._fitted = True
        return self

    def _envelope(
        self,
        features: FloatArray,
        step: int,
        reward_low: FloatArray,
        reward_high: FloatArray,
    ) -> tuple[FloatArray, FloatArray]:
        """``(lower, upper)`` envelopes at ``step``, each ``(n_rows, n_actions)``, range-clipped."""
        remaining = self.horizon - step + 1
        floor = self.reward_range[0] * remaining
        cap = self.reward_range[1] * remaining
        low = np.array(reward_low, dtype=np.float64, copy=True)
        high = np.array(reward_high, dtype=np.float64, copy=True)
        for action in range(self.n_actions):
            lower_model = self._lower_cont.get((step, action))
            if lower_model is not None:
                low[:, action] += np.asarray(lower_model.predict(features)).reshape(-1)
            upper_model = self._upper_cont.get((step, action))
            if upper_model is not None:
                high[:, action] += np.asarray(upper_model.predict(features)).reshape(-1)
        # A return over `remaining` steps cannot leave the scaled reward range, whatever the
        # regressions extrapolate to; and the envelope must not invert.
        low = np.clip(low, floor, cap)
        high = np.clip(high, floor, cap)
        return low, np.maximum(high, low)

    def envelope(
        self, observation: Mapping[str, Any], step: int = 1
    ) -> tuple[FloatArray, FloatArray]:
        """``(lower, upper)`` arrays over actions at ``observation`` and ``step``."""
        self._require_fit()
        if not 1 <= step <= self.horizon:
            raise ValueError(f"step={step} must lie in [1, {self.horizon}]")
        features = np.asarray(self.encoder.encode(observation), dtype=np.float64).reshape(1, -1)
        reward_low, reward_high = self._bounds_model.bounds(features)
        low, high = self._envelope(features, step, reward_low, reward_high)
        return low[0], high[0]

    def interval(self, observation: Mapping[str, Any], action: int, step: int = 1) -> Interval:
        """The bound on ``Q_step(observation, action)`` as an :class:`Interval`."""
        low, high = self.envelope(observation, step)
        if not 0 <= action < self.n_actions:
            raise ValueError(f"action={action} must lie in [0, {self.n_actions})")
        return Interval(float(low[action]), float(high[action]))

    def non_dominated(self, observation: Mapping[str, Any], step: int = 1) -> list[int]:
        """Actions whose upper bound reaches the best lower bound — the survivors.

        The feature-space counterpart of
        :func:`causalrl.agents.primitives.non_dominated_actions`. An action whose whole interval
        lies below another's floor cannot be optimal at this state and can be dropped before any
        exploration. As in the tabular case, natural Manski bounds are wide and this is often a
        no-op returning every action, which is correct rather than a defect.
        """
        low, high = self.envelope(observation, step)
        best_lower = float(low.max())
        return [a for a in range(self.n_actions) if float(high[a]) >= best_lower]

    def act(self, observation: dict[str, Any]) -> int:
        """Optimism in the face of confounding: the action with the highest upper bound."""
        step = min(int(observation.get("t", 0)) + 1, self.horizon)
        _, high = self.envelope(observation, step)
        best = float(high.max())
        winners = [a for a in range(self.n_actions) if float(high[a]) >= best - 1e-12]
        return int(self._rng.choice(winners))

    def observe_transition(self, state: int, action: int, next_state: int, done: bool) -> None:
        """Refused: states here are feature vectors, not indices."""
        raise NotImplementedError(
            "BoundedFittedQIteration takes feature-typed transitions; the inherited int-typed "
            "hook cannot express a continuous state. Encode both endpoints and pass "
            "FeatureTransition objects to fit()."
        )

    def observe_step(
        self,
        observation: dict[str, Any],
        action: int,
        reward: float,
        next_observation: dict[str, Any],
        done: bool,
    ) -> None:
        """Encode both endpoints and buffer the transition.

        See :meth:`causalrl.FittedQIteration.observe_step`.
        """
        self._buffer.append(
            FeatureTransition(
                state=self.encoder.encode(observation),
                action=action,
                reward=float(reward),
                next_state=self.encoder.encode(next_observation),
                done=done,
            )
        )
        self._fitted = False

    def buffered_transitions(self) -> tuple[FeatureTransition, ...]:
        """The transitions observed so far and not yet discarded by a ``fit(transitions=...)``.

        The online buffer is otherwise write-only: a caller driving this agent through
        :meth:`observe_step` had no way to see what it had actually collected, which makes
        "did my driver wire the hook up correctly" unanswerable without reaching into a private
        attribute.
        """
        return tuple(self._buffer)

    def _require_fit(self) -> None:
        if not self._fitted:
            raise RuntimeError(
                "BoundedFittedQIteration has no envelope yet; call fit() with transitions first."
            )

    def certificate(self) -> Certificate:
        """A ``BOUNDED`` certificate when the transition assumption holds, else ``EMPIRICAL``.

        Either way the assumptions are recorded explicitly, including the two the tabular path does
        not need — that the outcome regression and the propensity model are correctly specified.
        The overlap diagnostic rides along on the propensity assumption, since it is the one
        checkable thing about it: where estimated propensities are near zero the interval is wide
        and honest, and where a flexible model has driven them near one the interval is tight and
        only as good as that model.
        """
        self._require_fit()
        diagnostic = self._bounds_model.diagnostic()
        assumptions = (
            Assumption(
                name="reward-range",
                params={"low": self.reward_range[0], "high": self.reward_range[1]},
                checkable=False,
            ),
            Assumption(name="outcome-model-specification", checkable=False),
            Assumption(
                name="propensity-model-specification",
                checkable=True,
                diagnostic={
                    "min_propensity": diagnostic.min_propensity,
                    "mean_propensity": diagnostic.mean_propensity,
                    "vacuous_fraction": diagnostic.vacuous_fraction,
                },
            ),
            Assumption(
                name="unconfounded-transitions",
                params={"transition_assumption": self.transition_assumption},
                checkable=False,
            ),
        )
        claim = (
            f"Manski envelope on Q over {self.encoder.dim}-dimensional state features "
            f"(horizon={self.horizon}, actions={self.n_actions})"
        )
        if self.is_certified:
            return Certificate(
                claim=claim,
                estimand=EstimandSpec(query="policy_value", target="mean"),
                kind=Kind.BOUNDED,
                value=None,
                alpha=None,
                assumptions=assumptions,
                method="bounded_fitted_q_iteration",
                witness=None,
                hedge=Hedge(
                    reason=(
                        "the interval is a bound only up to correct specification of the outcome "
                        "and propensity models: unlike the tabular per-cell bound, whose inputs "
                        "are counts, these are fitted functions and their misspecification "
                        "narrows the interval"
                    ),
                    detail={"overlap": diagnostic.summary()},
                ),
                provenance=Provenance.create(),
            )
        return Certificate(
            claim=claim,
            estimand=EstimandSpec(query="policy_value", target="mean"),
            kind=Kind.EMPIRICAL,
            value=None,
            alpha=None,
            assumptions=assumptions,
            method="bounded_fitted_q_iteration",
            witness=None,
            hedge=Hedge(
                reason=(
                    "transition_assumption='unknown' with horizon > 1: the continuation "
                    "expectation is taken over the logged successor distribution, which a "
                    "confounder driving the dynamics would make non-interventional, so this is "
                    "value propagation rather than a bound"
                ),
                detail={"overlap": diagnostic.summary(), "allow_heuristic": self.allow_heuristic},
                downgraded_from="bounded",
            ),
            provenance=Provenance.create(),
        )
