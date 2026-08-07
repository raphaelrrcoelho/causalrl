"""Agents that always have an answer, and say when the clock cut the search short.

:class:`causalrl.Deadline` shipped in 3.0 as a well-built type that nothing consumed. It appeared
in :class:`~causalrl.agents.interventional.InterventionalAgent`'s signature, whose docstring said a
deadline-honouring agent "should keep a usable incumbent answer at all times" -- but nothing typed
that requirement and no shipped agent met it, so the parameter was decoration. That is precisely
the kind of unreachable surface 3.0 existed to delete, shipped by 3.0 itself.

:class:`Anytime` states the contract: there is always an incumbent, and refinement is something the
caller drives with whatever time is left. :class:`AnytimeInterventionSearch` implements it over an
:class:`~causalrl.InterventionSpace`, which is what makes a continuous domain actionable -- an
interval cannot be enumerated into arms, so the only way to decide over one is to search it, and
the only way to search under a real-time budget is to be interruptible.

The honesty requirement is the other half. A search stopped by its budget has examined a *subset*
of what it was asked to examine, and reporting its answer as though the sweep were exhaustive is
the search-completeness analogue of claiming identification you do not have. :class:`SearchReport`
records whether the search converged or was truncated, and its certificate carries a
budget-truncated :class:`~causalrl.certify.Hedge` in the second case.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np

from causalrl.agents.interventional import InterventionalAgent
from causalrl.certify.certificate import (
    Assumption,
    Certificate,
    EstimandSpec,
    Hedge,
    Kind,
    Provenance,
)
from causalrl.deadline import Deadline
from causalrl.intervention import Continuous, Intervention, InterventionSpace

__all__ = ["Anytime", "AnytimeInterventionSearch", "SearchReport"]


@runtime_checkable
class Anytime(Protocol):
    """An agent with a usable answer at every instant, refinable while time remains.

    The two halves are inseparable: :meth:`incumbent` without :meth:`refine` is a fixed answer, and
    :meth:`refine` without :meth:`incumbent` is a computation you cannot interrupt. Together they
    are what lets a decision be handed a wall-clock budget instead of being run to convergence.
    """

    def incumbent(self) -> Intervention:
        """The best answer found so far. Defined before any refinement has happened."""
        ...

    def refine(self, deadline: Deadline | None = None) -> bool:
        """Improve the incumbent within ``deadline``; return ``False`` once converged.

        ``False`` means further calls cannot help -- the caller should stop rather than spin. A
        return of ``True`` says only that more work remains, not that the last call used its whole
        budget.
        """
        ...


@dataclass(frozen=True)
class SearchReport:
    """How much of the search actually ran, and whether the budget stopped it.

    ``exhausted`` is the load-bearing field: ``True`` means the search finished the work it
    intended, ``False`` means a deadline cut it off and the answer is the best of a *subset*.
    """

    rounds: int
    rounds_planned: int
    candidates: int
    exhausted: bool
    best_value: float

    def certificate(self) -> Certificate:
        """An ``EMPIRICAL`` certificate, hedged when the budget truncated the search.

        Always ``EMPIRICAL``: a search over sampled candidates is sample evidence about which
        intervention looks best, and carries no identification guarantee whatever its budget. What
        the deadline changes is completeness, and that is what the hedge reports.
        """
        truncated = not self.exhausted
        return Certificate(
            claim=(
                f"best of {self.candidates} sampled interventions over {self.rounds} of "
                f"{self.rounds_planned} planned refinement rounds "
                f"(value {self.best_value:.6g})"
            ),
            estimand=EstimandSpec(query="do", target="mean"),
            kind=Kind.EMPIRICAL,
            value=None,
            alpha=None,
            assumptions=(
                Assumption(
                    name="search-completeness",
                    params={"rounds": self.rounds, "rounds_planned": self.rounds_planned},
                    checkable=True,
                    diagnostic={"exhausted": self.exhausted, "candidates": self.candidates},
                ),
            ),
            method="anytime-random-shooting",
            witness=None,
            hedge=(
                Hedge(
                    reason=(
                        "budget-truncated: the deadline expired after "
                        f"{self.rounds} of {self.rounds_planned} refinement rounds, so this is "
                        "the best of a subset of the intended search, not of all of it"
                    ),
                    detail={"rounds": self.rounds, "rounds_planned": self.rounds_planned},
                    downgraded_from="exhaustive-search",
                )
                if truncated
                else None
            ),
            provenance=Provenance.create(),
        )


class AnytimeInterventionSearch(InterventionalAgent):
    """Deadline-honouring random-shooting search over an :class:`~causalrl.InterventionSpace`.

    Round 0 samples the whole space; each later round resamples around the incumbent inside a
    window that shrinks by ``shrink``, so the answer improves monotonically and is usable after any
    number of rounds. ``value_fn(observation, intervention)`` scores a candidate -- typically a
    fitted outcome model, a ``BoundedSCMFit`` interval's midpoint, or an SCM's ``do`` expectation.

    The deadline is checked between candidates, not only between rounds, so a budget that expires
    mid-round still returns the incumbent rather than overrunning to a round boundary. Nothing here
    interrupts ``value_fn`` itself: a single call that overruns the budget will overrun it, which
    is the cooperative contract :class:`causalrl.Deadline` documents.
    """

    def __init__(
        self,
        value_fn: Callable[[Mapping[str, Any], Intervention], float],
        *,
        rounds: int = 8,
        candidates_per_round: int = 32,
        shrink: float = 0.5,
        seed: int | None = None,
    ) -> None:
        if rounds < 1:
            raise ValueError(f"rounds={rounds} must be at least 1")
        if candidates_per_round < 1:
            raise ValueError(f"candidates_per_round={candidates_per_round} must be at least 1")
        if not 0.0 < shrink <= 1.0:
            raise ValueError(
                f"shrink={shrink} must lie in (0, 1]: it multiplies the sampling window each "
                "round, so a value above 1 would widen the search around the incumbent and a "
                "non-positive one would collapse it to a point immediately."
            )
        self.value_fn = value_fn
        self.rounds = rounds
        self.candidates_per_round = candidates_per_round
        self.shrink = float(shrink)
        self._rng = np.random.default_rng(seed)
        self._incumbent: Intervention = {}
        self._best_value = -float("inf")
        self._report = SearchReport(0, rounds, 0, exhausted=False, best_value=-float("inf"))

    def incumbent(self) -> Intervention:
        """The best intervention found by the most recent :meth:`act`."""
        return dict(self._incumbent)

    @property
    def last_search(self) -> SearchReport:
        """How the most recent :meth:`act` terminated. Its certificate carries the hedge."""
        return self._report

    def act(
        self,
        observation: Mapping[str, Any],
        *,
        space: InterventionSpace,
        deadline: Deadline | None = None,
    ) -> Intervention:
        """Search ``space`` for the highest-value intervention within ``deadline``."""
        variables = sorted(space.variables)
        if not variables:
            self._incumbent = {}
            self._report = SearchReport(0, self.rounds, 0, exhausted=True, best_value=0.0)
            return {}

        self._incumbent = space.sample(variables, self._rng)
        self._best_value = float(self.value_fn(observation, self._incumbent))
        candidates = 1
        completed = 0

        for round_index in range(self.rounds):
            width = self.shrink**round_index
            for _ in range(self.candidates_per_round):
                if deadline is not None and deadline.expired():
                    self._report = SearchReport(
                        completed, self.rounds, candidates, False, self._best_value
                    )
                    return dict(self._incumbent)
                candidate = self._propose(space, variables, width)
                candidates += 1
                value = float(self.value_fn(observation, candidate))
                if value > self._best_value:
                    self._best_value, self._incumbent = value, candidate
            completed = round_index + 1

        self._report = SearchReport(completed, self.rounds, candidates, True, self._best_value)
        return dict(self._incumbent)

    def refine(self, deadline: Deadline | None = None) -> bool:
        """Not separately drivable: this agent's search runs inside :meth:`act`.

        Present because :class:`Anytime` requires it. It reports convergence (``False``) whenever
        the last :meth:`act` ran to completion, so a caller driving the protocol generically stops
        instead of spinning on a search that has nowhere left to go.
        """
        return not self._report.exhausted

    def _propose(
        self, space: InterventionSpace, variables: list[str], width: float
    ) -> Intervention:
        """One candidate: continuous variables jittered around the incumbent, discrete resampled.

        The window is a fraction ``width`` of each continuous domain, centred on the incumbent and
        clipped back into the domain, so every proposal is admissible by construction and no
        candidate is wasted on a rejection.
        """
        proposal: dict[str, Any] = {}
        for name in variables:
            domain = space.domain(name)
            if isinstance(domain, Continuous) and width < 1.0:
                span = (domain.high - domain.low) * width
                centre = float(self._incumbent[name])
                proposal[name] = domain.project(
                    self._rng.uniform(centre - span / 2.0, centre + span / 2.0)
                )
            else:
                proposal[name] = domain.sample(self._rng)
        return proposal

    def update(
        self, observation: Mapping[str, Any], intervention: Intervention, reward: float
    ) -> None:
        """No-op: this agent's policy is its ``value_fn``, which it does not fit.

        Learning the value model is a separate concern with its own entry points (``fit_scm``,
        ``FittedQIteration``, ``fit_scm_bounded``); this class is the search over one.
        """
        return None
