"""Coarse-correlated-equilibrium partial identification for finite games (T2 instrument).

For a population of no-regret learners playing a finite :class:`~causalrl.games.CausalGame`,
time-averaged play approaches the coarse-correlated-equilibrium (CCE) set; an intervention ``do``
that pins some agents' actions induces the intervened game. Any long-run time-averaged linear
functional of play therefore lies between the min and max of that functional over the CCE polytope
of the intervened game — both linear programs (:func:`cce_bounds`).

The *finite-time, anytime-valid* form needs no asymptotics at all: at any horizon, the realized
empirical joint distribution is — by construction — feasible for the ``epsilon``-relaxed polytope
with ``epsilon`` equal to its *measured* maximal deviation gain (:func:`cce_regret`). Passing that
measured value to :func:`certify_cce_do` therefore discharges the no-regret assumption at the run
horizon instead of assuming it in the limit.

:func:`certify_cce_do` returns the certificate ladder:

* ``IDENTIFIED`` — the functional is constant over the polytope (width <= tol): the equilibrium
  point prediction is valid for *every* no-regret population;
* ``BOUNDED`` — no-regret assumed, or a measured ``epsilon`` supplied: the LP interval is a sound
  partial identification of the population's time-averaged behaviour;
* ``EMPIRICAL`` — the interval is vacuous at the given regret level (abstention), or no-regret is
  neither assumed nor measured: the interval is reported as evidence, not a guarantee.

The interval *width* is itself a diagnostic: how much predictive content the equilibrium analysis
has for adaptive populations under that intervention.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from itertools import product

import numpy as np

from causalrl.certify.certificate import (
    Assumption,
    Certificate,
    EstimandSpec,
    Hedge,
    Kind,
    Provenance,
    Witness,
)
from causalrl.games import CausalGame
from causalrl.graphs import graph_hash
from causalrl.identification.bounds import Interval
from causalrl.magames._lp import FloatArray, solve_lp

__all__ = ["CCEPolytope", "cce_bounds", "cce_polytope", "cce_regret", "certify_cce_do"]

_METHOD = "cce-polytope-lp"

Functional = Callable[[Mapping[str, int]], float]
Epsilon = float | Mapping[str, float]


@dataclass(frozen=True)
class CCEPolytope:
    """The CCE polytope of a (possibly intervened) finite game, in distribution-over-profiles form.

    ``profiles`` are the joint actions (in ``agents`` order) consistent with the intervention;
    ``deviation_gains[k, j]`` is the payoff gain of ``constraint_labels[k] = (agent, action)``
    switching unilaterally to ``action`` when the realized profile is ``profiles[j]``. A
    distribution ``mu`` over ``profiles`` is an ``epsilon``-CCE iff ``deviation_gains @ mu`` is
    componentwise at most ``epsilon``.
    """

    agents: tuple[str, ...]
    profiles: tuple[tuple[int, ...], ...]
    deviation_gains: FloatArray
    constraint_labels: tuple[tuple[str, int], ...]


def cce_polytope(game: CausalGame, *, do: Mapping[str, int] | None = None) -> CCEPolytope:
    """Deviation-gain representation of the CCE polytope of ``game`` under intervention ``do``.

    ``do`` pins the given agents' actions: profiles are restricted to the consistent ones and only
    the free agents contribute no-deviation constraints (the intervened game).
    """
    do = dict(do or {})
    for name, action in do.items():
        if name not in game.agents:
            raise KeyError(f"unknown agent: {name!r}")
        if action not in game.actions[name]:
            raise ValueError(f"action {action!r} not available to agent {name!r}")
    profiles = tuple(
        p
        for p in product(*(game.actions[a] for a in game.agents))
        if all(p[game.agents.index(a)] == do[a] for a in do)
    )
    free = [a for a in game.agents if a not in do]
    rows: list[list[float]] = []
    labels: list[tuple[str, int]] = []
    for agent in free:
        idx = game.agents.index(agent)
        table = game.utilities[agent]
        for deviation in game.actions[agent]:
            rows.append([table[(*p[:idx], deviation, *p[idx + 1 :])] - table[p] for p in profiles])
            labels.append((agent, deviation))
    gains = np.asarray(rows, dtype=np.float64) if rows else np.zeros((0, len(profiles)))
    return CCEPolytope(tuple(game.agents), profiles, gains, tuple(labels))


def _epsilon_vector(polytope: CCEPolytope, epsilon: Epsilon) -> FloatArray:
    if isinstance(epsilon, Mapping):
        unknown = set(epsilon) - set(polytope.agents)
        if unknown:
            raise KeyError(f"epsilon given for unknown agents: {sorted(unknown)}")
        values = np.array(
            [float(epsilon.get(agent, 0.0)) for agent, _ in polytope.constraint_labels]
        )
    else:
        values = np.full(len(polytope.constraint_labels), float(epsilon))
    if np.any(values < 0):
        raise ValueError("epsilon must be nonnegative")
    return values


def _functional_vector(polytope: CCEPolytope, functional: Functional) -> FloatArray:
    return np.array(
        [float(functional(dict(zip(polytope.agents, p, strict=True)))) for p in polytope.profiles]
    )


def _polytope_bounds(polytope: CCEPolytope, values: FloatArray, epsilon: Epsilon) -> Interval:
    n = len(polytope.profiles)
    a_ub, b_ub = (
        (polytope.deviation_gains, _epsilon_vector(polytope, epsilon))
        if len(polytope.constraint_labels)
        else (None, None)
    )
    a_eq, b_eq = np.ones((1, n)), np.array([1.0])
    low = solve_lp(values, a_ub=a_ub, b_ub=b_ub, a_eq=a_eq, b_eq=b_eq)
    high = solve_lp(-values, a_ub=a_ub, b_ub=b_ub, a_eq=a_eq, b_eq=b_eq)
    if low.status != "optimal" or high.status != "optimal":
        raise RuntimeError(  # pragma: no cover - a nonempty bounded polytope always solves
            f"CCE bound LPs did not solve: ({low.status}, {high.status})"
        )
    assert low.value is not None and high.value is not None
    return Interval(low.value, -high.value)


def cce_bounds(
    game: CausalGame,
    functional: Functional,
    *,
    do: Mapping[str, int] | None = None,
    epsilon: Epsilon = 0.0,
) -> Interval:
    """Min/max of ``functional``'s expectation over the ``epsilon``-CCE polytope under ``do``.

    ``functional`` maps a joint profile (``{agent: action}``) to a real value; ``epsilon`` is a
    uniform or per-agent relaxation of the no-deviation constraints — pass a *measured* realized
    regret (:func:`cce_regret`) for the finite-time form.
    """
    polytope = cce_polytope(game, do=do)
    return _polytope_bounds(polytope, _functional_vector(polytope, functional), epsilon)


def cce_regret(
    game: CausalGame,
    weights: Mapping[tuple[int, ...], float] | Sequence[float] | FloatArray,
    *,
    do: Mapping[str, int] | None = None,
) -> float:
    """Maximal deviation gain of a joint distribution over profiles: the measured realized regret.

    ``weights`` is either a mapping from joint-action tuples (in ``game.agents`` order) to
    probabilities or a sequence aligned with ``cce_polytope(game, do=do).profiles``. The result is
    the smallest ``epsilon`` for which ``weights`` is ``epsilon``-CCE-feasible (0 for an exact CCE);
    it is what a no-regret population drives to 0 and what :func:`certify_cce_do` accepts as the
    measured ``epsilon``.
    """
    polytope = cce_polytope(game, do=do)
    if isinstance(weights, Mapping):
        mu = np.array([float(weights.get(p, 0.0)) for p in polytope.profiles])
    else:
        mu = np.asarray(weights, dtype=np.float64)
        if mu.shape != (len(polytope.profiles),):
            raise ValueError(
                f"weights must align with the {len(polytope.profiles)} polytope profiles"
            )
    if np.any(mu < 0) or abs(float(mu.sum()) - 1.0) > 1e-6:
        raise ValueError("weights must be nonnegative and sum to 1 over the polytope profiles")
    if not len(polytope.constraint_labels):
        return 0.0
    return float(np.max(polytope.deviation_gains @ mu))


def certify_cce_do(
    game: CausalGame,
    functional: Functional,
    *,
    do: Mapping[str, int] | None = None,
    no_regret: bool = True,
    epsilon: Epsilon | None = None,
    tol: float = 1e-9,
    seed: int = 0,
) -> Certificate:
    """Certify what ``functional``'s time average can be for a learning population under ``do``.

    ``no_regret`` asserts the population's time-averaged play approaches the exact CCE set (the
    asymptotic route); ``epsilon`` instead supplies a *measured* realized regret at the horizon
    actually run (the finite-time route, which needs no assumption). See the module docstring for
    the resulting ``IDENTIFIED``/``BOUNDED``/``EMPIRICAL`` ladder.
    """
    polytope = cce_polytope(game, do=do)
    values = _functional_vector(polytope, functional)
    interval = _polytope_bounds(polytope, values, epsilon if epsilon is not None else 0.0)
    width = interval.upper - interval.lower
    full = Interval(float(np.min(values)), float(np.max(values)))
    full_width = full.upper - full.lower
    vacuous = full_width > tol and width >= full_width - tol
    licensed = no_regret or epsilon is not None

    measured = None if epsilon is None else epsilon
    witness = Witness(
        "cce-interval",
        {
            "interval": [interval.lower, interval.upper],
            "width": width,
            "epsilon": measured if not isinstance(measured, Mapping) else dict(measured),
            "n_profiles": len(polytope.profiles),
            "n_constraints": len(polytope.constraint_labels),
            "do": dict(do or {}),
        },
    )
    assumptions = (
        Assumption(name="finite-game", params={"agents": list(game.agents)}, checkable=True),
        Assumption(
            name="no-regret",
            params={
                "assumed": no_regret,
                "measured_epsilon": None if isinstance(measured, Mapping) else measured,
            },
            checkable=True,
            diagnostic={"check": "cce_regret(game, realized_joint_distribution, do=do)"},
        ),
    )
    provenance = Provenance.create(seeds=(seed,), graph_hash=graph_hash(game.graph))
    estimand = EstimandSpec(query="equilibrium", target="mean")
    intervened = f" | do={dict(do)}" if do else ""

    if vacuous:
        return Certificate(
            claim=f"CCE interval is vacuous at this regret level{intervened}: abstaining",
            estimand=estimand,
            kind=Kind.EMPIRICAL,
            value=interval,
            alpha=None,
            assumptions=assumptions,
            method=_METHOD,
            witness=witness,
            hedge=Hedge(
                reason=(
                    "the epsilon-CCE interval spans the functional's full range — vacuous; "
                    "abstaining rather than certifying an uninformative bound"
                ),
                detail={"width": width, "full_range": [full.lower, full.upper]},
            ),
            provenance=provenance,
        )
    if licensed and width <= tol:
        midpoint = 0.5 * (interval.lower + interval.upper)
        return Certificate(
            claim=(
                "functional is constant over the CCE polytope: the equilibrium point prediction "
                f"is valid for every no-regret population{intervened}"
            ),
            estimand=estimand,
            kind=Kind.IDENTIFIED,
            value=midpoint,
            alpha=None,
            assumptions=assumptions,
            method=_METHOD,
            witness=witness,
            hedge=None,
            provenance=provenance,
        )
    if licensed:
        finite_time = epsilon is not None
        route = (
            "measured realized regret (finite-time, no asymptotic assumption)"
            if finite_time
            else "no-regret learning (time-averaged play approaches the CCE set)"
        )
        return Certificate(
            claim=(
                f"time-averaged functional of the learning population lies in the CCE interval"
                f"{intervened} [{route}]"
            ),
            estimand=estimand,
            kind=Kind.BOUNDED,
            value=interval,
            alpha=None,
            assumptions=assumptions,
            method=_METHOD,
            witness=witness,
            hedge=None,
            provenance=provenance,
        )
    return Certificate(
        claim=f"CCE interval reported as evidence only{intervened}",
        estimand=estimand,
        kind=Kind.EMPIRICAL,
        value=interval,
        alpha=None,
        assumptions=assumptions,
        method=_METHOD,
        witness=witness,
        hedge=Hedge(
            reason=(
                "no-regret neither assumed nor measured for this population: the CCE interval "
                "is evidence, not a guarantee"
            ),
            detail={"width": width},
        ),
        provenance=provenance,
    )
