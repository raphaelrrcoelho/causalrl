"""Deadline: a wall-clock budget for a single decision.

An agent embedded in a live loop is not asked "what is the best action" but "what is the best
action *you can name by time T*". This is the type that carries T. It is deliberately tiny and
free of any policy: a deadline reports how much time is left and whether it has passed, and it is
the *agent's* job to decide what to do about that (return its incumbent answer, shrink a search,
skip a refinement). Nothing here interrupts a running computation.

Time is read from :func:`time.monotonic`, so a deadline is unaffected by wall-clock adjustments
and is meaningful only within one process. ``None`` is the idiomatic "no budget" value throughout
causalrl rather than a sentinel instance, so an unbudgeted call site stays free of deadline
machinery entirely.
"""

from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass(frozen=True)
class Deadline:
    """A monotonic instant by which a decision must be returned.

    Construct with :meth:`after` (a duration from now) rather than the raw constructor, which
    takes an absolute :func:`time.monotonic` reading and is mainly useful in tests where the clock
    is supplied explicitly.
    """

    at: float
    """Absolute :func:`time.monotonic` value at which the budget expires."""

    @classmethod
    def after(cls, seconds: float) -> Deadline:
        """A deadline ``seconds`` from now.

        A non-positive duration is allowed and yields an already-expired deadline: callers that
        compute a remaining budget by subtraction should get "no time left", not an exception.
        """
        return cls(time.monotonic() + seconds)

    def remaining(self) -> float:
        """Seconds left, clamped at ``0.0`` once the deadline has passed."""
        return max(0.0, self.at - time.monotonic())

    def expired(self) -> bool:
        """Whether the budget is exhausted."""
        return time.monotonic() >= self.at

    def fraction_of(self, share: float) -> float:
        """``share`` of the remaining budget, in seconds — for splitting time across sub-steps.

        ``share`` must lie in ``[0, 1]``: it is a fraction of what is left, not a multiplier, and
        a value above 1 would hand out a slice larger than the remaining budget under a name that
        says otherwise.
        """
        if not 0.0 <= share <= 1.0:
            raise ValueError(
                f"share={share} must lie in [0, 1]: fraction_of hands out that fraction of the "
                "remaining budget, so a value above 1 would return more time than the deadline "
                "allows while claiming to be a share of it."
            )
        return self.remaining() * share
