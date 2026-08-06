"""Fitted value iteration over state *features* — the continuous-state counterpart of DOVI.

:class:`causalrl.DOVI` plans over `(horizon, n_states, n_actions)` tables and holds an explicit
`(n_states, n_actions, n_states)` transition tensor. Both are tabular by construction. This module
keeps the backward induction and replaces the table with a regressor, so the state may be any
feature vector a :class:`~causalrl.state.StateEncoder` produces.

The regressor is the one the estimation core already ships:
:class:`causalrl.estimate.nuisance.Regressor` is a duck-typed ``fit``/``predict`` protocol that any
scikit-learn estimator satisfies, and which the DML estimators already use for their nuisances. The
control layer needed no new dependency to go continuous — only a type that was already there.

**What is lost, stated plainly.** DOVI caps its optimism with a Manski upper bound *per*
``(state, action)`` cell, computed from the offline log. In feature space there are no cells, so
that bound has no direct analogue: a bound holding uniformly over a function class is a different
and weaker object, and estimating one from data is a research question rather than a refactor. This
class therefore falls back to a **global** cap — with a per-step reward bounded by ``reward_max``,
the return from step ``h`` of a horizon-``H`` problem cannot exceed ``reward_max * (H - h + 1)``.
That cap is valid without any function-class assumption, and it is much weaker than the per-cell
bound. :meth:`FittedQIteration.certificate` says so: the claim is ``EMPIRICAL``, explicitly
downgraded from the ``BOUNDED`` status the tabular path enjoys, and nothing here silently inherits
DOVI's guarantee.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from causalrl.agents.base import Agent
from causalrl.certify.certificate import (
    Assumption,
    Certificate,
    EstimandSpec,
    Hedge,
    Kind,
    Provenance,
)
from causalrl.estimate.nuisance import Regressor, RidgeRegressor
from causalrl.state import FeatureTransition, FloatArray, StateEncoder

__all__ = ["FittedQIteration"]


class FittedQIteration(Agent):
    """Finite-horizon fitted Q iteration over encoded state features.

    Backward induction, identical in shape to :class:`causalrl.DOVI`'s but with the table replaced
    by one fitted regressor per ``(step, action)``::

        V_{H+1}(s)  = 0
        Q_h(·, a)   = fit( features(s) -> r + [not done] · V_{h+1}(s') )   over rows with A = a
        Q_h         = min(Q_h, reward_max · (H - h + 1))      # the global optimism cap
        V_h(s)      = max_a Q_h(s, a)

    ``encoder`` decides what a state *is*. With :class:`~causalrl.state.OneHotEncoder` and a
    least-squares regressor the fitted backup reproduces the tabular one, because the indicator
    basis spans every function on a finite state set — the tabular agent is a special case of this
    one, not a separate algorithm.

    An action never taken at some step has no data to fit, so its Q is set to the cap. That keeps
    the backup optimistic in the unexplored direction, matching how the tabular agent treats an
    unvisited cell.

    ``reward_max`` must be a genuine per-step upper bound on the reward; the cap is only as valid
    as that number. Rewards are not otherwise assumed bounded below.
    """

    def __init__(
        self,
        n_actions: int,
        horizon: int,
        encoder: StateEncoder,
        *,
        regressor: Callable[[], Regressor] | None = None,
        reward_max: float = 1.0,
        seed: int | None = None,
    ) -> None:
        if n_actions < 1:
            raise ValueError(f"n_actions={n_actions} must be at least 1")
        if horizon < 1:
            raise ValueError(f"horizon={horizon} must be at least 1")
        self.n_actions = n_actions
        self.horizon = horizon
        self.encoder = encoder
        self.reward_max = float(reward_max)
        self._make_regressor = regressor if regressor is not None else RidgeRegressor
        self._buffer: list[FeatureTransition] = []
        self._models: dict[tuple[int, int], Regressor] = {}
        self._fitted = False
        self._rng = np.random.default_rng(seed)

    @property
    def is_certified(self) -> bool:
        """Always ``False``: a fitted backup carries no per-cell causal bound.

        Named to match :attr:`causalrl.DOVI.is_certified` so the two are directly comparable. The
        tabular agent can answer ``True`` because its optimism is capped by a Manski bound computed
        per ``(state, action)``; this one caps optimism globally by horizon and reward range, which
        is valid but is not an identification guarantee.
        """
        return False

    def cap(self, step: int) -> float:
        """The global optimism cap at step ``step``: ``reward_max * (horizon - step + 1)``.

        The largest return achievable from ``step`` onward when each of the remaining steps pays at
        most ``reward_max``. Valid with no assumption on the function class, and much weaker than a
        per-cell bound.
        """
        if not 1 <= step <= self.horizon:
            raise ValueError(f"step={step} must lie in [1, {self.horizon}]")
        return self.reward_max * (self.horizon - step + 1)

    def observe(self, transition: FeatureTransition) -> None:
        """Add one feature-typed transition to the buffer and invalidate the current plan."""
        self._buffer.append(transition)
        self._fitted = False

    def observe_transition(self, state: int, action: int, next_state: int, done: bool) -> None:
        """Refused: this agent's states are feature vectors, not indices.

        The tabular hook is overridden rather than left as the inherited no-op, which would accept
        the call and silently discard the transition. Encode the endpoints and call
        :meth:`observe` with a :class:`~causalrl.state.FeatureTransition` instead.
        """
        raise NotImplementedError(
            "FittedQIteration takes feature-typed transitions: the inherited "
            "observe_transition(state: int, ...) hook cannot express a continuous state, and "
            "accepting it here would silently drop the transition. Encode both endpoints with "
            "this agent's encoder and call observe(FeatureTransition(...))."
        )

    def fit(self, transitions: Sequence[FeatureTransition] | None = None) -> FittedQIteration:
        """Run the backward induction over ``transitions`` (or the observed buffer) and plan.

        Passing ``transitions`` replaces the buffer, so a caller can refit from a fresh batch
        without carrying stale online data. Raises ``ValueError`` on an empty dataset — there is no
        such thing as a fitted value function with nothing to fit it on.
        """
        if transitions is not None:
            self._buffer = list(transitions)
        if not self._buffer:
            raise ValueError(
                "no transitions to fit: pass a non-empty sequence to fit(), or call observe() "
                "before fitting. A fitted backup with no data would return the optimism cap "
                "everywhere, which is a vacuous plan rather than a learned one."
            )

        dim = self.encoder.dim
        states = np.stack([t.state for t in self._buffer])
        next_states = np.stack([t.next_state for t in self._buffer])
        if states.shape[1] != dim:
            raise ValueError(
                f"transitions carry {states.shape[1]}-dimensional features but the encoder "
                f"produces {dim}: they were built with a different encoder."
            )
        actions = np.array([t.action for t in self._buffer], dtype=int)
        rewards = np.array([t.reward for t in self._buffer], dtype=np.float64)
        not_done = np.array([not t.done for t in self._buffer], dtype=np.float64)
        if actions.min() < 0 or actions.max() >= self.n_actions:
            raise ValueError(
                f"transitions contain action(s) outside [0, {self.n_actions}): "
                f"observed range [{actions.min()}, {actions.max()}]"
            )

        self._models = {}
        value_at_next = np.zeros(len(self._buffer), dtype=np.float64)
        for step in range(self.horizon, 0, -1):
            targets = np.minimum(rewards + not_done * value_at_next, self.cap(step))
            for action in range(self.n_actions):
                rows = actions == action
                if not rows.any():
                    continue  # no data: q_values() falls back to the cap for this (step, action)
                model = self._make_regressor()
                model.fit(states[rows], targets[rows])
                self._models[(step, action)] = model
            value_at_next = self._value(next_states, step)
        self._fitted = True
        return self

    def _q_matrix(self, features: FloatArray, step: int) -> FloatArray:
        """``(n_rows, n_actions)`` capped Q values at ``step`` for a batch of feature rows."""
        cap = self.cap(step)
        out = np.full((features.shape[0], self.n_actions), cap, dtype=np.float64)
        for action in range(self.n_actions):
            model = self._models.get((step, action))
            if model is not None:
                out[:, action] = np.minimum(np.asarray(model.predict(features)).reshape(-1), cap)
        return out

    def _value(self, features: FloatArray, step: int) -> FloatArray:
        return self._q_matrix(features, step).max(axis=1)

    def q_values(self, observation: Mapping[str, Any], step: int = 1) -> FloatArray:
        """Capped Q values for every action at ``observation``, as a ``(n_actions,)`` array."""
        self._require_fit()
        features = np.asarray(self.encoder.encode(observation), dtype=np.float64).reshape(1, -1)
        return self._q_matrix(features, step)[0]

    def value(self, observation: Mapping[str, Any], step: int = 1) -> float:
        """``max_a Q_step(observation, a)`` — the fitted value at this observation."""
        return float(self.q_values(observation, step).max())

    def act(self, observation: dict[str, Any]) -> int:
        """Greedy action at the observation's step, ties broken uniformly at random.

        The step is read from ``observation["t"]`` (0-indexed, as the shipped environments emit it)
        and clamped to the horizon, matching :meth:`causalrl.DOVI.act`.
        """
        step = min(int(observation.get("t", 0)) + 1, self.horizon)
        scores = self.q_values(observation, step)
        best = float(scores.max())
        winners = [a for a in range(self.n_actions) if scores[a] >= best - 1e-12]
        return int(self._rng.choice(winners))

    def update(self, observation: dict[str, Any], action: int, reward: float) -> None:
        """No-op: this is a batch method, and a reward alone cannot update a fitted backup.

        The backup needs the successor state, which this signature does not carry. Record the whole
        transition with :meth:`observe` and call :meth:`fit` to replan; silently accepting the
        reward here would suggest the plan had moved when it had not.
        """

    def _require_fit(self) -> None:
        if not self._fitted:
            raise RuntimeError(
                "FittedQIteration has no plan yet: call fit() after supplying transitions. "
                "Acting before fitting would return the optimism cap for every action, which is "
                "an arbitrary tie rather than a decision."
            )

    def certificate(self) -> Certificate:
        """An ``EMPIRICAL`` certificate recording exactly which guarantee was given up.

        The hedge carries ``downgraded_from="bounded"``: the tabular path is partially identified
        by a per-cell Manski bound, and replacing the table with a function class forfeits that in
        exchange for a global horizon cap. Reporting this as ``BOUNDED`` would claim an
        identification status the fitted backup does not have.
        """
        return Certificate(
            claim=(
                f"fitted Q iteration over {self.encoder.dim}-dimensional state features "
                f"(horizon={self.horizon}, actions={self.n_actions})"
            ),
            estimand=EstimandSpec(query="policy_value", target="mean"),
            kind=Kind.EMPIRICAL,
            value=None,
            alpha=None,
            assumptions=(
                Assumption(
                    name="reward-ceiling",
                    params={"reward_max": self.reward_max, "horizon": self.horizon},
                    checkable=False,
                ),
                Assumption(
                    name="function-class",
                    params={
                        "encoder": type(self.encoder).__name__,
                        "dim": self.encoder.dim,
                        "regressor": getattr(self._make_regressor, "__name__", "callable"),
                    },
                    checkable=False,
                ),
            ),
            method="fitted_q_iteration",
            witness=None,
            hedge=Hedge(
                reason=(
                    "optimism is capped globally by reward_max * steps remaining, not by a "
                    "per-(state, action) Manski bound: in feature space there are no cells to "
                    "bound, so the tabular partial-identification guarantee does not carry over"
                ),
                detail={"tabular_equivalent": "DOVI", "global_cap_at_step_1": self.cap(1)},
                downgraded_from="bounded",
            ),
            provenance=Provenance.create(),
        )
