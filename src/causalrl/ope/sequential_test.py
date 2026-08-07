"""Anytime-valid bands: policy comparison that survives being watched.

Everything else in :mod:`causalrl.ope` makes a fixed-sample claim -- ``certify_policy``,
``conformal_action_value``, the MSM bounds. A fixed-sample interval is valid at the sample size you
declared *before* looking. In practice nobody works that way: you run some episodes, look, run
more, and stop when the comparison looks decisive. That procedure inflates the error rate without
limit, and "stop when the interval excludes zero" is the single most common way an evaluation
fools itself -- especially in self-play, where the temptation is to stop as soon as the new agent
looks better.

A confidence sequence fixes it by construction. Where a confidence interval satisfies
``P(mu in CI_t) >= 1 - alpha`` for one fixed ``t``, a confidence sequence satisfies

    P(for all t: mu in CS_t) >= 1 - alpha,

so the band is valid at *every* sample size simultaneously and a caller may look as often as they
like, stop whenever they like, and still hold the coverage guarantee. "Run until the band excludes
zero" becomes a valid stopping rule rather than a bias generator.

The boundary is the normal-mixture (conjugate-mixture sub-Gaussian) one of Howard, Ramdas,
McAuliffe & Sekhon, *Time-uniform, nonasymptotic confidence sequences* (Annals of Statistics, 2021).
It is non-asymptotic and exact under the sub-Gaussian condition, which bounded scores satisfy with
variance proxy ``(high - low) / 2``. Formula-level implementation; no code is ported.

There is a symmetry worth naming: :mod:`causalrl.magames`'s finite-time regret certificate is
already anytime-valid in the *game* sense -- it holds at every round of a no-regret run. This is its
statistical twin.

Honest scope. The guarantee here is about SAMPLING error under repeated looking. It says nothing
about confounding: a confounded off-policy estimate converges to the wrong number, and a
time-uniform band around it just tracks the wrong number more reliably. Compose with
:func:`~causalrl.certify_policy` for the confounding layer -- and note the two guarantees do not
multiply into a single joint one, which is why this returns its own certificate rather than
silently widening an MSM bound.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeVar

import numpy as np

from causalrl.certify.certificate import (
    Assumption,
    Certificate,
    EstimandSpec,
    Hedge,
    Kind,
    Provenance,
)
from causalrl.data.logged import LoggedDecisions
from causalrl.identification.bounds import Interval

__all__ = [
    "ConfidenceSequence",
    "SequentialVerdict",
    "confidence_sequence",
    "sequential_policy_comparison",
]

ActionT = TypeVar("ActionT")


def _mixture_radius(n: int, sigma: float, alpha: float, rho: float) -> float:
    """Normal-mixture boundary: the half-width valid simultaneously at every sample size.

    ``rho`` is the mixture variance, which tunes *where* the boundary is tightest without affecting
    its validity anywhere: the sequence is uniformly valid for any positive ``rho``.
    """
    if n <= 0:
        return float("inf")
    inner = (n * rho + 1.0) / (n * n * rho)
    return sigma * math.sqrt(2.0 * inner * math.log(math.sqrt(n * rho + 1.0) / alpha))


@dataclass(frozen=True)
class ConfidenceSequence:
    """A time-uniform band on a mean, and the record of what it took to get there.

    ``interval`` is deliberately NOT reported through ``Certificate.ci``'s usual fixed-sample
    reading. It is a different object with a stronger guarantee -- valid at all sample sizes at
    once -- and conflating the two would let a reader assume the wrong one.
    """

    interval: Interval
    n: int
    mean: float
    alpha: float
    sigma: float

    @property
    def excludes_zero(self) -> bool:
        """Whether the band has separated from zero -- a valid stopping condition, at any time."""
        return self.interval.lower > 0.0 or self.interval.upper < 0.0

    @property
    def width(self) -> float:
        return self.interval.upper - self.interval.lower


def confidence_sequence(
    values: Sequence[float],
    *,
    alpha: float = 0.05,
    value_range: tuple[float, float],
    rho: float | None = None,
) -> ConfidenceSequence:
    """A band on ``E[value]`` valid at every sample size simultaneously.

    ``value_range`` must genuinely contain every observation: it sets the sub-Gaussian variance
    proxy ``(high - low) / 2``, and a range that is too narrow silently voids the guarantee rather
    than degrading it. ``rho`` tunes where the boundary is tightest and defaults to a value
    optimised near the sample size actually supplied; validity does not depend on it.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha={alpha} must lie in (0, 1)")
    low, high = float(value_range[0]), float(value_range[1])
    if not low < high:
        raise ValueError(f"value_range={value_range} must satisfy low < high")
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    n = int(x.size)
    if n == 0:
        raise ValueError(
            "confidence_sequence needs at least one observation: a band over nothing is the whole "
            "real line, which is true but not a claim anyone can act on."
        )
    outside = float(np.max(np.abs(x - (low + high) / 2.0))) if n else 0.0
    if outside > (high - low) / 2.0 + 1e-9:
        raise ValueError(
            f"value_range={value_range} does not contain every observation (found one "
            f"{outside - (high - low) / 2.0:.6g} outside). The range is the sub-Gaussian variance "
            "proxy, so a range that is too narrow voids the time-uniform guarantee outright "
            "rather than making the band merely optimistic."
        )
    sigma = (high - low) / 2.0
    # Tuned so the boundary is tightest near the sample size in hand; validity holds for any rho.
    tuned = rho if rho is not None else max(1.0 / max(n, 1), 1e-12)
    mean = float(x.mean())
    radius = _mixture_radius(n, sigma, alpha, tuned)
    return ConfidenceSequence(
        interval=Interval(mean - radius, mean + radius),
        n=n,
        mean=mean,
        alpha=alpha,
        sigma=sigma,
    )


@dataclass(frozen=True)
class SequentialVerdict:
    """The result of an anytime-valid policy comparison, and whether it is safe to stop."""

    sequence: ConfidenceSequence
    certificate: Certificate

    @property
    def stop(self) -> bool:
        """Whether the band has separated from zero, so stopping now is a valid decision."""
        return self.sequence.excludes_zero

    @property
    def better(self) -> str | None:
        """``"target"``, ``"behavior"``, or ``None`` while the comparison is undecided."""
        if not self.stop:
            return None
        return "target" if self.sequence.interval.lower > 0.0 else "behavior"


def sequential_policy_comparison(
    dataset: LoggedDecisions[ActionT],
    target_actions: Sequence[ActionT],
    *,
    alpha: float = 0.05,
    reward_range: tuple[float, float],
    weight_cap: float = 100.0,
) -> SequentialVerdict:
    """Compare a target policy to the logging policy with a band valid under repeated looking.

    The per-decision score is the clipped importance-weighted contrast
    ``(w_i - 1) * r_i`` with ``w_i = 1[a_i = pi(s_i)] / e0(a_i | s_i)`` capped at ``weight_cap``,
    whose mean is the value improvement ``V(pi) - V(behaviour)``. Clipping is what makes the score
    bounded and hence sub-Gaussian; it also biases the estimate toward the behaviour policy, which
    is the conservative direction for a "should I ship this" question and is recorded as an
    assumption rather than left implicit.

    Read :attr:`SequentialVerdict.stop` as the stopping rule: it may be consulted after every new
    batch of logs without spending any error budget.
    """
    outcomes = list(dataset.outcomes())
    e0 = list(dataset.logging_propensities())
    matched = dataset.matches(target_actions)
    weights = [
        min(weight_cap, 1.0 / p) if m and p > 0.0 else 0.0 for m, p in zip(matched, e0, strict=True)
    ]
    scores = [(w - 1.0) * r for w, r in zip(weights, outcomes, strict=True)]

    low, high = float(reward_range[0]), float(reward_range[1])
    bound = max(abs(low), abs(high)) * max(weight_cap - 1.0, 1.0)
    sequence = confidence_sequence(scores, alpha=alpha, value_range=(-bound, bound))

    decided = sequence.excludes_zero
    certificate = Certificate(
        claim=(
            f"V(target) - V(behavior) in [{sequence.interval.lower:.4g}, "
            f"{sequence.interval.upper:.4g}] simultaneously at every sample size "
            f"(confidence sequence, alpha={alpha:g}, n={sequence.n})"
        ),
        estimand=EstimandSpec(query="policy_value", target="mean", policy="target"),
        kind=Kind.EMPIRICAL,
        value=None,
        alpha=alpha,
        assumptions=(
            Assumption(
                name="sub-gaussian-scores",
                params={"bound": bound, "weight_cap": weight_cap},
                checkable=True,
                diagnostic={"n": sequence.n},
            ),
            Assumption(name="no-unmeasured-confounding", params={}, checkable=False),
            Assumption(
                name="clipped-importance-weights",
                params={"weight_cap": weight_cap},
                checkable=True,
                diagnostic={"clipped": sum(1 for w in weights if w >= weight_cap)},
            ),
        ),
        method="normal-mixture-confidence-sequence",
        witness=None,
        hedge=(
            None
            if decided
            else Hedge(
                reason=(
                    "undecided: the band still contains zero, so stopping now would report a "
                    "difference the data have not separated from no difference"
                ),
                detail={"n": sequence.n, "width": sequence.width},
            )
        ),
        provenance=Provenance.create(),
    )
    return SequentialVerdict(sequence=sequence, certificate=certificate)
