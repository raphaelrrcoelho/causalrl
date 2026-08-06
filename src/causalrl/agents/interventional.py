"""The set-valued agent contract, and an adapter that lifts an arm-indexed agent into it.

:class:`causalrl.Agent` answers with an ``int``. That is the right type for a bandit over a fixed
arm list and the wrong one everywhere the decision is *which variables to set, and to what* --
which is how the rest of causalrl states the problem: ``do`` takes a mapping, POMIS returns sets.
:class:`InterventionalAgent` closes that gap, and :class:`ScalarAgentAdapter` lets every existing
arm-indexed agent satisfy it without being rewritten.

Two things the contract adds beyond the return type:

* the admissible :class:`~causalrl.intervention.InterventionSpace` is supplied **per decision**,
  because which levers are available is a property of the current state, not of the graph; and
* an optional :class:`~causalrl.deadline.Deadline`, because an agent in a live loop is asked for
  the best action it can name by a fixed time.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any

from causalrl.agents.base import Agent
from causalrl.deadline import Deadline
from causalrl.intervention import Intervention, InterventionSpace, canonical


class InterventionalAgent(ABC):
    """An agent whose action is an intervention — an assignment, not an arm index.

    Implementations must return an intervention that ``space`` permits. The base class does not
    enforce that (a subclass may have a cheaper way to stay in bounds than re-checking), but
    :meth:`check_permitted` is provided for the ones that want it.

    ``deadline`` is advisory and cooperative: nothing here interrupts a running computation, so an
    implementation that ignores it will simply overrun. An agent that *does* honour it should keep
    a usable incumbent answer at all times and return it once
    :meth:`~causalrl.deadline.Deadline.expired` reports the budget is gone. ``None`` means no
    budget, which is the default so that offline and test call sites stay free of timing concerns.
    """

    @abstractmethod
    def act(
        self,
        observation: Mapping[str, Any],
        *,
        space: InterventionSpace,
        deadline: Deadline | None = None,
    ) -> Intervention:
        """Choose an intervention admissible in ``space``, ideally within ``deadline``."""

    @abstractmethod
    def update(
        self, observation: Mapping[str, Any], intervention: Intervention, reward: float
    ) -> None:
        """Learn from the reward that followed ``intervention``."""

    @staticmethod
    def check_permitted(intervention: Intervention, space: InterventionSpace) -> Intervention:
        """Return ``intervention`` if ``space`` permits it, else raise ``ValueError`` saying why.

        A helper rather than an enforced invariant: an agent that constructs its candidates from
        ``space`` in the first place cannot violate it and should not pay to re-check.
        """
        if not space.permits(intervention):
            offending = {
                name: value
                for name, value in intervention.items()
                if name not in space.variables or value not in space.values(name)
            }
            raise ValueError(
                f"intervention {dict(intervention)!r} is not admissible in this "
                f"InterventionSpace: {offending!r} names a variable that is not manipulable here, "
                f"or a value outside its domain (manipulable: {sorted(space.variables)})"
            )
        return intervention


class ScalarAgentAdapter(InterventionalAgent):
    """Present an arm-indexed :class:`causalrl.Agent` as an :class:`InterventionalAgent`.

    ``arms[i]`` is the intervention the wrapped agent's action ``i`` denotes, so the adapter is
    just a codebook in both directions: :meth:`act` maps the chosen index to its intervention, and
    :meth:`update` maps an intervention back to its index before forwarding the reward.

    **The wrapped agent's arm list is fixed, so ``space`` can only be checked, not honoured.** A
    scalar agent has no way to be told that arm 3 is unavailable this turn -- its policy is defined
    over all its arms. :meth:`act` therefore verifies that the arm it returns is admissible and
    raises if it is not, rather than silently emitting an inadmissible action or quietly
    substituting a different one. Where the arm list genuinely varies per decision, implement
    :class:`InterventionalAgent` directly instead of adapting a scalar agent.

    ``deadline`` is accepted and ignored for the same reason: :meth:`causalrl.Agent.act` takes no
    budget, so there is nothing to pass it to.
    """

    def __init__(self, agent: Agent, arms: Sequence[Intervention]) -> None:
        if not arms:
            raise ValueError("arms must be non-empty: an agent with no arms has nothing to choose")
        keys = [canonical(arm) for arm in arms]
        duplicates = sorted({key for key in keys if keys.count(key) > 1})
        if duplicates:
            raise ValueError(
                f"arms contains duplicate intervention(s) {[dict(k) for k in duplicates]!r}: the "
                "codebook must be one-to-one, or update() cannot tell which index a reward "
                "belongs to."
            )
        self._agent = agent
        self._arms = [dict(arm) for arm in arms]
        self._index_of = {key: i for i, key in enumerate(keys)}

    @property
    def arms(self) -> tuple[Intervention, ...]:
        """The codebook: ``arms[i]`` is the intervention denoted by the wrapped agent's action."""
        return tuple(dict(arm) for arm in self._arms)

    def act(
        self,
        observation: Mapping[str, Any],
        *,
        space: InterventionSpace,
        deadline: Deadline | None = None,
    ) -> Intervention:
        """Map the wrapped agent's chosen index to its intervention, checked against ``space``."""
        index = self._agent.act(dict(observation))
        if not 0 <= index < len(self._arms):
            raise IndexError(
                f"wrapped agent chose action {index}, outside the codebook of {len(self._arms)} "
                "arms: the agent's action space and the arms passed to ScalarAgentAdapter disagree"
            )
        return self.check_permitted(self._arms[index], space)

    def update(
        self, observation: Mapping[str, Any], intervention: Intervention, reward: float
    ) -> None:
        """Forward ``reward`` to the wrapped agent under the index that ``intervention`` encodes."""
        key = canonical(intervention)
        if key not in self._index_of:
            raise KeyError(
                f"intervention {dict(intervention)!r} is not in this adapter's codebook, so there "
                "is no arm index to credit the reward to. update() must be called with an "
                "intervention that act() returned."
            )
        self._agent.update(dict(observation), self._index_of[key], reward)
