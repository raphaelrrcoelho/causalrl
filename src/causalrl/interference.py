"""Interference: outcomes coupled through *other* units' treatments (Task-2 estimands under SUTVA
failure).

Everything else in causalrl assumes a unit's outcome responds to its own intervention alone. That
is SUTVA, and it is false wherever units share something -- a queue, a contagion, a limited supply,
a leaderboard. There the estimand splits in two: the **direct** effect of a unit's own treatment,
and the **spillover** effect of everyone else's.

The standard way to make that tractable is an **exposure mapping** (Aronow & Samii, *Ann. Appl.
Stat.* 2017): rather than let the outcome depend on the whole treatment vector -- which has
``2 ** n`` potential outcomes per unit and no hope of estimation -- assume it depends on that
vector only through a low-dimensional summary,

    ``Y_i(A) = Y_i(A_i, f(i, A))``

for a known ``f``. :class:`ExposureMapping` is that ``f``; :func:`neighbourhood_count`,
:func:`neighbourhood_fraction`, :func:`any_neighbour_treated` and :func:`population_share` are the
common ones. Given the mapping, the estimands below are ordinary stratified contrasts:

* :func:`direct_effect` — vary the unit's own treatment, hold exposure fixed.
* :func:`spillover_effect` — hold the unit's own treatment, vary exposure.
* :func:`total_effect` — vary both, from one regime to another.

**The exposure mapping is an assumption, and it is not testable from the data.** Choosing ``f``
asserts that no further feature of the treatment vector matters; if it does, these contrasts are
biased and nothing here will say so. The estimators verify only what data can verify -- that the
cells a contrast needs are actually populated -- and raise
:class:`~causalrl.exceptions.NotIdentifiableError` naming the empty cell when they are not. This
module deliberately reports no standard errors: under interference the rows within a group are
dependent, so the usual two-sample formula does not apply, and a variance estimate has to come
from the randomisation design that produced the data rather than from the outcome column.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np

from causalrl.exceptions import NotIdentifiableError

Adjacency = Sequence[Iterable[int]]
"""Peer structure: ``adjacency[i]`` is the set of units whose treatment can reach unit ``i``.

Not required to be symmetric -- influence often is not -- and a unit listing itself is refused by
the built-in mappings, which are all defined over a unit's *peers* and would otherwise
double-count its own treatment into its own exposure.
"""


def adjacency_from_matrix(matrix: np.ndarray) -> Adjacency:
    """Convert an ``n``-by-``n`` boolean/0-1 matrix to :data:`Adjacency`, reading row ``i`` as
    unit ``i``'s peers.

    Raises ``ValueError`` on a non-square matrix or a nonzero diagonal — a unit that is its own
    peer would fold its own treatment into its exposure, which is exactly the split the exposure
    mapping exists to keep clean.
    """
    array = np.asarray(matrix)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError(f"adjacency matrix must be square; got shape {array.shape}")
    if bool(np.any(np.diag(array))):
        offenders = sorted(int(i) for i in np.flatnonzero(np.diag(array)))
        raise ValueError(
            f"adjacency matrix has a nonzero diagonal at unit(s) {offenders}: a unit listed as its "
            "own peer folds its own treatment into its exposure, which collapses the direct and "
            "spillover effects this module exists to separate. Zero the diagonal."
        )
    return [[int(j) for j in np.flatnonzero(row)] for row in array]


@dataclass(frozen=True)
class ExposureMapping:
    """A known summary ``f(i, A)`` of the treatment vector, from unit ``i``'s point of view.

    ``name`` labels the mapping in errors and reports. ``fn`` receives the unit index and the full
    treatment vector and returns that unit's exposure, which may be any value that groups by
    equality (an int, a float, a bool, a label).

    Prefer the constructors below to writing ``fn`` by hand; they already exclude the unit's own
    treatment, which is the easy thing to get wrong.
    """

    name: str
    fn: Callable[[int, np.ndarray], Any]

    def __call__(self, unit: int, treatments: np.ndarray) -> Any:
        return self.fn(unit, np.asarray(treatments))

    def column(self, treatments: np.ndarray) -> np.ndarray:
        """Every unit's exposure under one treatment vector, as an array aligned with it.

        For repeated draws over the same structure, call this once per draw and concatenate; the
        estimators take flat per-observation arrays and are indifferent to how the rows were
        stacked.
        """
        vector = np.asarray(treatments)
        if vector.ndim != 1:
            raise ValueError(
                f"treatments must be a 1-D vector of one entry per unit; got shape {vector.shape}. "
                "For repeated draws, call column() per draw and concatenate the results."
            )
        return np.array([self.fn(i, vector) for i in range(len(vector))])


def _check_peers(adjacency: Adjacency) -> list[list[int]]:
    peers = [[int(j) for j in row] for row in adjacency]
    for i, row in enumerate(peers):
        if i in row:
            raise ValueError(
                f"unit {i} is listed as its own peer: its own treatment would be counted into its "
                "own exposure, collapsing the direct and spillover effects. Remove the self-edge."
            )
    return peers


def neighbourhood_count(adjacency: Adjacency) -> ExposureMapping:
    """Exposure is the **number** of treated peers — the coarsest useful count summary."""
    peers = _check_peers(adjacency)

    def fn(unit: int, treatments: np.ndarray) -> int:
        return int(sum(int(treatments[j] != 0) for j in peers[unit]))

    return ExposureMapping("neighbourhood_count", fn)


def neighbourhood_fraction(adjacency: Adjacency, *, decimals: int = 2) -> ExposureMapping:
    """Exposure is the **fraction** of a unit's peers that are treated, rounded to ``decimals``.

    Rounding is not cosmetic: exposure strata are formed by equality, so an unrounded ratio would
    put units with 3/7 and 4/9 treated peers in different cells and leave most cells with one
    observation in them. ``decimals`` trades resolution against how populated the cells are. A unit
    with no peers has no fraction to report and is given exposure ``0.0``.
    """
    peers = _check_peers(adjacency)

    def fn(unit: int, treatments: np.ndarray) -> float:
        row = peers[unit]
        if not row:
            return 0.0
        treated = sum(int(treatments[j] != 0) for j in row)
        return round(treated / len(row), decimals)

    return ExposureMapping("neighbourhood_fraction", fn)


def any_neighbour_treated(adjacency: Adjacency) -> ExposureMapping:
    """Exposure is ``True`` when at least one peer is treated — the binary spillover regime."""
    peers = _check_peers(adjacency)

    def fn(unit: int, treatments: np.ndarray) -> bool:
        return any(int(treatments[j] != 0) for j in peers[unit])

    return ExposureMapping("any_neighbour_treated", fn)


def population_share(*, decimals: int = 2) -> ExposureMapping:
    """Exposure is the share of **all other** units that are treated — no peer structure needed.

    The fully-connected case, where every unit is coupled to every other through one shared
    aggregate rather than through named neighbours. This is the mapping that matches a mean-field
    reading of a population (compare :mod:`causalrl.meanfield`, which takes the same aggregate as
    the thing an agent best-responds to). ``decimals`` rounds the share into strata for the same
    reason as :func:`neighbourhood_fraction`. A lone unit has no others and is given ``0.0``.
    """

    def fn(unit: int, treatments: np.ndarray) -> float:
        n = len(treatments)
        if n <= 1:
            return 0.0
        treated = int(np.sum(treatments != 0)) - int(treatments[unit] != 0)
        return round(treated / (n - 1), decimals)

    return ExposureMapping("population_share", fn)


class Cell(NamedTuple):
    """One stratum of a contrast: the mean outcome there, and how many rows it rests on."""

    treatment: Any
    exposure: Any
    mean: float
    n: int


class ExposureContrast(NamedTuple):
    """A difference of two exposure-stratified cell means, with both cells shown.

    ``estimate`` is ``high.mean - low.mean``. Both cells are carried so a caller can see the
    footing the number rests on -- a contrast between a 400-row cell and a 3-row cell is a very
    different claim from one between two 200-row cells, and the point estimate alone does not say
    which it is.
    """

    estimand: str
    estimate: float
    high: Cell
    low: Cell

    def summary(self) -> str:
        return (
            f"{self.estimand}: {self.estimate:+.4f}  "
            f"[A={self.high.treatment!r}, E={self.high.exposure!r}] n={self.high.n} "
            f"mean={self.high.mean:.4f}  -  "
            f"[A={self.low.treatment!r}, E={self.low.exposure!r}] n={self.low.n} "
            f"mean={self.low.mean:.4f}"
        )


def _cell(
    outcome: np.ndarray,
    treatment: np.ndarray,
    exposure: np.ndarray,
    *,
    at_treatment: Any,
    at_exposure: Any,
    estimand: str,
) -> Cell:
    mask = (treatment == at_treatment) & (exposure == at_exposure)
    count = int(np.count_nonzero(mask))
    if count == 0:
        raise NotIdentifiableError(
            f"{estimand} needs the cell (treatment={at_treatment!r}, exposure={at_exposure!r}), "
            "which no observation falls in. This is a positivity failure, not a modelling choice: "
            "the data never realised that combination of own-treatment and exposure, so its mean "
            "outcome is not estimable from this sample. Widen the exposure strata (a coarser "
            "mapping, or fewer decimals), or supply data whose randomisation reaches that cell.",
            witness=(at_treatment, at_exposure),
        )
    return Cell(
        treatment=at_treatment,
        exposure=at_exposure,
        mean=float(np.asarray(outcome, dtype=float)[mask].mean()),
        n=count,
    )


def _aligned(
    outcome: np.ndarray, treatment: np.ndarray, exposure: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y, a, e = np.asarray(outcome), np.asarray(treatment), np.asarray(exposure)
    if not (y.ndim == a.ndim == e.ndim == 1):
        raise ValueError(
            f"outcome, treatment and exposure must each be 1-D, one entry per observation; got "
            f"shapes {y.shape}, {a.shape}, {e.shape}"
        )
    if not (len(y) == len(a) == len(e)):
        raise ValueError(
            f"outcome, treatment and exposure must be the same length; got {len(y)}, {len(a)}, "
            f"{len(e)}. Each row is one unit under one draw -- if you stacked several draws, "
            "stack all three columns the same way."
        )
    if len(y) == 0:
        raise ValueError("no observations supplied")
    return y, a, e


def direct_effect(
    outcome: np.ndarray,
    treatment: np.ndarray,
    exposure: np.ndarray,
    *,
    at_exposure: Any,
    treated: Any = 1,
    control: Any = 0,
) -> ExposureContrast:
    """The effect of a unit's **own** treatment, holding exposure fixed at ``at_exposure``.

    ``E[Y(treated, at_exposure)] - E[Y(control, at_exposure)]``, estimated by the difference in
    cell means. Holding exposure fixed is what makes this the direct effect: the contrast between
    marginal means over all exposures would confound it with the spillover, since a unit's own
    treatment and its peers' are rarely independent.
    """
    y, a, e = _aligned(outcome, treatment, exposure)
    estimand = "direct_effect"
    high = _cell(y, a, e, at_treatment=treated, at_exposure=at_exposure, estimand=estimand)
    low = _cell(y, a, e, at_treatment=control, at_exposure=at_exposure, estimand=estimand)
    return ExposureContrast(estimand, high.mean - low.mean, high, low)


def spillover_effect(
    outcome: np.ndarray,
    treatment: np.ndarray,
    exposure: np.ndarray,
    *,
    at_treatment: Any,
    exposed: Any,
    unexposed: Any,
) -> ExposureContrast:
    """The effect of **others'** treatments, holding the unit's own treatment at ``at_treatment``.

    ``E[Y(at_treatment, exposed)] - E[Y(at_treatment, unexposed)]``. This is the quantity SUTVA
    asserts is zero; estimating it is how you find out whether that assertion was safe. Note that
    it is reported *at a given own-treatment*: the spillover onto treated units and onto untreated
    ones are different estimands and need not agree in sign.
    """
    y, a, e = _aligned(outcome, treatment, exposure)
    estimand = "spillover_effect"
    high = _cell(y, a, e, at_treatment=at_treatment, at_exposure=exposed, estimand=estimand)
    low = _cell(y, a, e, at_treatment=at_treatment, at_exposure=unexposed, estimand=estimand)
    return ExposureContrast(estimand, high.mean - low.mean, high, low)


def total_effect(
    outcome: np.ndarray,
    treatment: np.ndarray,
    exposure: np.ndarray,
    *,
    treated: Any = 1,
    treated_exposure: Any,
    control: Any = 0,
    control_exposure: Any,
) -> ExposureContrast:
    """The effect of moving a unit from one whole regime to another — own treatment *and* exposure.

    ``E[Y(treated, treated_exposure)] - E[Y(control, control_exposure)]``. This is the policy-
    relevant number when an intervention changes everyone's treatment at once, so a unit's exposure
    moves with its own assignment rather than staying fixed. It is not the sum of
    :func:`direct_effect` and :func:`spillover_effect` unless the two are additive, which the
    exposure mapping does not assume and this function does not check.
    """
    y, a, e = _aligned(outcome, treatment, exposure)
    estimand = "total_effect"
    high = _cell(y, a, e, at_treatment=treated, at_exposure=treated_exposure, estimand=estimand)
    low = _cell(y, a, e, at_treatment=control, at_exposure=control_exposure, estimand=estimand)
    return ExposureContrast(estimand, high.mean - low.mean, high, low)
