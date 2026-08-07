"""What the certificate layer actually needs from a log.

3.0 gave the *planning* half of the library types richer than an arm index --
:class:`causalrl.state.FeatureTransition` for states, :data:`causalrl.Intervention` for actions --
and stopped there. :func:`causalrl.certify_policy` and :func:`causalrl.conformal_action_value`
still took ``Sequence[int]``, so the output of an :class:`~causalrl.agents.interventional.
InterventionalAgent` could not be handed to the very functions the README points it at without
round-tripping through an arm codebook. That inverted the library's headline claim, and it
inverted its own maturity split: the certificate layer is the half that is supposed to run on real
data.

This module supplies the missing seam. The observation is that off-policy certification needs
almost nothing from a log: the outcome of each logged decision, the probability the *logging*
policy gave the action it took, and -- given a target policy's actions -- which logged decisions
that policy would have reproduced. None of that mentions how a state or an action is represented.
:class:`LoggedDecisions` states exactly that much, generically in the action type, and the two
concrete logs (tabular :class:`~causalrl.data.dataset.ConfoundedTrajectoryDataset`, feature-space
:class:`FeatureDecisionLog`) each satisfy it. The certificate functions became generic rather than
growing a ``Sequence[int] | Sequence[Intervention]`` union, which would have pushed the
representation question into every call site instead of settling it once.

Positivity is the one place the two logs genuinely differ, and the difference is not cosmetic. A
tabular log counts, so it can look up the behaviour propensity of an action *nobody took in that
state* and report the gap. A feature-space log cannot: it knows the propensity of the action that
was logged and nothing else. Reporting "no positivity gaps found" in that case would be a false
negative dressed as a check, so :class:`PositivityReport` carries whether the check was possible
at all, and a log with no behaviour model says so instead of passing quietly.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Generic, Protocol, TypeVar, cast

import numpy as np

from causalrl.intervention import canonical
from causalrl.state import FloatArray

__all__ = [
    "FeatureDecisionLog",
    "LoggedDecision",
    "LoggedDecisions",
    "PositivityReport",
    "action_key",
]

ActionT = TypeVar("ActionT")
ActionT_contra = TypeVar("ActionT_contra", contravariant=True)


def action_key(action: object) -> Any:
    """A hashable, comparison-stable key for an action of any supported type.

    An arm index is its own key. An :data:`~causalrl.Intervention` is a ``Mapping`` and therefore
    neither hashable nor order-independent under ``==`` for the purpose of dict keys, so it goes
    through :func:`causalrl.intervention.canonical` -- the library's single canonical form. Using
    one function for both is what lets :class:`LoggedDecisions` be generic in the action type
    without every implementation re-deciding what "the same action" means.
    """
    if isinstance(action, Mapping):
        return canonical(cast("Mapping[str, Any]", action))
    return action


@dataclass(frozen=True)
class PositivityReport:
    """Whether a target policy strays outside the logs' support -- and whether that is knowable.

    ``checkable`` is the load-bearing field. A tabular log counts every ``(state, action)`` cell,
    so it can answer the question and ``checkable`` is ``True``; a feature-space log without a
    behaviour-propensity model cannot answer it at all, and reports ``False`` with empty ``gaps``
    rather than an empty ``gaps`` that reads as a clean bill of health. Consumers must not treat
    the two the same: the first licenses a checkable positivity assumption, the second a hedge.
    """

    checkable: bool
    gaps: tuple[str, ...] = ()

    @property
    def violated(self) -> bool:
        """Whether a gap was actually found. ``False`` when the check could not be run."""
        return bool(self.gaps)


class LoggedDecisions(Protocol[ActionT_contra]):
    """The off-policy certification interface: outcomes, logging propensities, target matches.

    Deliberately smaller than any concrete log. Nothing here mentions a state representation,
    because none of the certificate machinery needs one -- the marginal sensitivity model reweights
    *units*, and a unit is identified by its position in the log.
    """

    def __len__(self) -> int:
        """Number of logged decisions."""
        ...

    def outcomes(self) -> Sequence[float]:
        """The logged return of each decision, in log order."""
        ...

    def logging_propensities(self) -> Sequence[float]:
        """``pi_behavior(a_i | s_i)`` for the action actually logged at each decision."""
        ...

    def matches(self, target_actions: Sequence[ActionT_contra]) -> Sequence[bool]:
        """Whether the target policy would have taken the logged action, per decision."""
        ...

    def target_propensities(
        self, target_actions: Sequence[ActionT_contra]
    ) -> Sequence[float] | None:
        """``pi_behavior(pi_target(s_i) | s_i)`` per decision, or ``None`` if unknowable here.

        Note this is the behaviour propensity of the action the *target* policy would take, which
        is a strictly harder thing to know than :meth:`logging_propensities`: it asks about an
        action that may never have been played. A tabular log counts cells and can answer; a
        feature-space log without a behaviour model returns ``None``. The conformal band needs it
        to bound the likelihood ratio at a fresh test point, and ``None`` correctly forces that
        bound to infinity rather than letting an unknown pass as a small number.
        """
        ...

    def positivity(self, target_actions: Sequence[ActionT_contra]) -> PositivityReport:
        """Where the target policy leaves the logs' support, if that is knowable at all.

        Must agree with :meth:`target_propensities`: ``checkable`` exactly when that returns a
        sequence, and non-empty ``gaps`` exactly where its entries are non-positive. The two are
        separate methods because one supplies a bound and the other a human-readable diagnostic,
        and collapsing them would cost the cell labels that make a positivity failure actionable.
        """
        ...


@dataclass(frozen=True)
class LoggedDecision(Generic[ActionT]):
    """One logged decision in feature space, with the propensity that produced it.

    The feature-space counterpart of :class:`~causalrl.data.dataset.Transition` for the
    *certification* path rather than the planning path: it carries no successor state, because
    nothing in off-policy certification of a terminal return looks at one. ``propensity`` is
    supplied rather than counted -- in feature space there are no cells to count.
    """

    state: FloatArray
    action: ActionT
    reward: float
    propensity: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "state", np.asarray(self.state, dtype=np.float64).reshape(-1))
        if not 0.0 < self.propensity <= 1.0:
            raise ValueError(
                f"propensity={self.propensity} must lie in (0, 1]: it is the logging policy's "
                "probability of the action it actually took, which is positive by construction "
                "for any action that appears in the log. A zero here would divide by zero in "
                "every importance weight downstream."
            )


class FeatureDecisionLog(Generic[ActionT]):
    """A :class:`LoggedDecisions` over feature-space states and any action type.

    Build one from an agent's own logs when its states are vectors or its actions are
    :data:`~causalrl.Intervention` assignments. Actions are compared through :func:`action_key`,
    so a policy that returns ``{"dose": 2.0}`` matches a logged ``{"dose": 2.0}`` regardless of
    mapping identity or key order.

    ``behavior_propensity`` is optional and exists solely to make positivity *checkable*: given
    ``pi_behavior(a | s)`` it can ask whether the target policy's action had support at each logged
    state. Without it the log is still fully usable -- every weight the certificate layer forms
    uses the logged action's own propensity, which is always present -- but
    :meth:`positivity` reports that it could not check rather than that it found nothing.
    """

    def __init__(
        self,
        decisions: Sequence[LoggedDecision[ActionT]],
        *,
        behavior_propensity: Callable[[FloatArray, ActionT], float] | None = None,
    ) -> None:
        if not decisions:
            raise ValueError(
                "FeatureDecisionLog needs at least one logged decision: every quantity the "
                "certificate layer computes is an average over the log, and there is no "
                "off-policy claim to make from an empty one."
            )
        self._decisions = tuple(decisions)
        self._behavior_propensity = behavior_propensity

    def __len__(self) -> int:
        return len(self._decisions)

    @property
    def decisions(self) -> tuple[LoggedDecision[ActionT], ...]:
        """The logged decisions, in order."""
        return self._decisions

    def outcomes(self) -> Sequence[float]:
        return [d.reward for d in self._decisions]

    def logging_propensities(self) -> Sequence[float]:
        return [d.propensity for d in self._decisions]

    def matches(self, target_actions: Sequence[ActionT]) -> Sequence[bool]:
        self._check_length(target_actions)
        return [
            action_key(a) == action_key(d.action)
            for a, d in zip(target_actions, self._decisions, strict=True)
        ]

    def target_propensities(self, target_actions: Sequence[ActionT]) -> Sequence[float] | None:
        self._check_length(target_actions)
        if self._behavior_propensity is None:
            return None
        return [
            float(self._behavior_propensity(d.state, a))
            for a, d in zip(target_actions, self._decisions, strict=True)
        ]

    def positivity(self, target_actions: Sequence[ActionT]) -> PositivityReport:
        propensities = self.target_propensities(target_actions)
        if propensities is None:
            return PositivityReport(checkable=False)
        gaps = tuple(
            sorted(
                {
                    f"action {target_actions[i]!r} unsupported at logged state {i}"
                    for i, e in enumerate(propensities)
                    if e <= 0.0
                }
            )
        )
        return PositivityReport(checkable=True, gaps=gaps)

    def _check_length(self, target_actions: Sequence[ActionT]) -> None:
        if len(target_actions) != len(self._decisions):
            raise ValueError(
                f"target_actions has {len(target_actions)} entries but the log has "
                f"{len(self._decisions)} decisions: the certificate layer needs the action the "
                "target policy would take at each logged decision, one for one."
            )
