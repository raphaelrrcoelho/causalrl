"""EXPERIMENTAL (unstable, evaluation-only): mean-field limit of a symmetric population (plan §8.2).

Not API-frozen — this package is outside causalrl's semver guarantees until promoted (§14), and is
reachable only as ``causalrl.meanfield`` (deliberately not in the top-level public API). It
evaluates the ``N → ∞`` limit of a symmetric two-action
:class:`~causalrl.magames.population.Population`: an agent best-responds to the *aggregate* fraction
of the population playing action ``1`` rather than to named opponents. A mean-field equilibrium is a
fixed point ``p*`` of the best-response-to-``p`` map. Evaluation-only (no learning); every
certificate is ``kind=EMPIRICAL`` (limiting-dynamics evidence).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from causalrl.certify.certificate import (
    Assumption,
    Certificate,
    EstimandSpec,
    Kind,
    Provenance,
)

__all__ = [
    "UNSTABLE",
    "MeanFieldGame",
    "best_response_fraction",
    "certify_mean_field_equilibrium",
    "mean_field_equilibria",
]

#: This package is experimental and may change or be removed without a deprecation window.
UNSTABLE = True

# payoff(own_action in {0, 1}, fraction of the population playing action 1) -> payoff
MeanFieldPayoff = Callable[[int, float], float]


@dataclass(frozen=True)
class MeanFieldGame:
    """A symmetric two-action mean-field game: ``payoff(own_action, fraction_playing_1)``."""

    payoff: MeanFieldPayoff


def best_response_fraction(game: MeanFieldGame, p: float) -> float:
    """Fraction playing ``1`` in a best response to population mix ``p`` (0, 0.5, or 1)."""
    u0 = game.payoff(0, p)
    u1 = game.payoff(1, p)
    if u1 > u0:
        return 1.0
    if u0 > u1:
        return 0.0
    return 0.5  # indifferent between the two actions


def mean_field_equilibria(game: MeanFieldGame, *, grid: int = 201) -> list[float]:
    """Fixed points ``p*`` of the best-response map: pure ``p ∈ {0, 1}`` plus interior indifference.

    An interior equilibrium is a root of ``d(p) = payoff(1, p) - payoff(0, p)`` (every agent is
    indifferent, so any mix is sustained); the pure ``0`` / ``1`` are equilibria when the matching
    action is a best response there. Found by a grid scan (evaluation-only).
    """
    eqs: list[float] = []
    if game.payoff(0, 0.0) >= game.payoff(1, 0.0):
        eqs.append(0.0)
    if game.payoff(1, 1.0) >= game.payoff(0, 1.0):
        eqs.append(1.0)
    ps = np.linspace(0.0, 1.0, grid)
    d = np.array([game.payoff(1, float(p)) - game.payoff(0, float(p)) for p in ps])
    for i in range(len(ps)):
        if d[i] == 0.0:  # indifference exactly on the grid point
            eqs.append(round(float(ps[i]), 6))
        elif i < len(ps) - 1 and d[i] * d[i + 1] < 0.0:  # sign change straddles a root
            p_star = float(ps[i] - d[i] * (ps[i + 1] - ps[i]) / (d[i + 1] - d[i]))
            eqs.append(round(p_star, 6))
    return sorted(set(eqs))


def certify_mean_field_equilibrium(game: MeanFieldGame, *, grid: int = 201) -> Certificate:
    """An ``EMPIRICAL`` certificate reporting the mean-field equilibria (experimental, unstable)."""
    eqs = mean_field_equilibria(game, grid=grid)
    return Certificate(
        claim=f"mean-field equilibria (fraction playing 1): {eqs}",
        estimand=EstimandSpec(query="equilibrium", target="mean"),
        kind=Kind.EMPIRICAL,
        value=None,
        alpha=None,
        assumptions=(
            Assumption(
                name="mean-field-limit",
                params={"symmetric": True, "actions": 2, "grid": grid, "unstable": UNSTABLE},
                checkable=False,
            ),
        ),
        method="mean-field-fixed-point",
        witness=None,
        hedge=None,
        provenance=Provenance.create(),
    )
