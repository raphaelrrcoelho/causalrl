"""Dense two-phase simplex in pure NumPy (internal; plan: CCE bounds are linear programs).

The core deliberately ships without scipy (see ``estimate/_stats.py``); the polytopes solved here —
deviation-constraint sets of small finite games — have a handful of variables and constraints, so a
dense tableau simplex with Bland's anti-cycling rule is exact enough and dependency-free.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

__all__ = ["LPResult", "solve_lp"]

_Matrix = FloatArray | Sequence[Sequence[float]]
_Vector = FloatArray | Sequence[float]


@dataclass(frozen=True)
class LPResult:
    """Outcome of :func:`solve_lp` (``status``: ``optimal`` / ``infeasible`` / ``unbounded``).

    On ``"optimal"``, ``x`` is the minimiser (original variables only), ``value`` is ``c @ x``,
    and ``dual_ub`` holds the sensitivities ``d value / d b_ub`` (one per ``<=`` row; a valid
    subgradient at degenerate optima; ``None`` when no inequality rows were given). Otherwise
    ``x``/``value``/``dual_ub`` are ``None``.
    """

    status: str
    x: FloatArray | None
    value: float | None
    dual_ub: FloatArray | None = None


def solve_lp(
    c: _Vector,
    *,
    a_ub: _Matrix | None = None,
    b_ub: _Vector | None = None,
    a_eq: _Matrix | None = None,
    b_eq: _Vector | None = None,
    tol: float = 1e-9,
) -> LPResult:
    """Minimise ``c @ x`` subject to ``a_ub @ x <= b_ub``, ``a_eq @ x == b_eq`` and ``x >= 0``."""
    cost = np.asarray(c, dtype=np.float64)
    n = cost.size
    ub_rows = np.zeros((0, n)) if a_ub is None else np.asarray(a_ub, dtype=np.float64)
    ub_rhs = np.zeros(0) if b_ub is None else np.asarray(b_ub, dtype=np.float64)
    eq_rows = np.zeros((0, n)) if a_eq is None else np.asarray(a_eq, dtype=np.float64)
    eq_rhs = np.zeros(0) if b_eq is None else np.asarray(b_eq, dtype=np.float64)
    n_ub = ub_rows.shape[0]

    # Standard form: append one slack per <= row, then flip rows so every RHS is nonnegative.
    table = np.block(
        [
            [ub_rows, np.eye(n_ub)],
            [eq_rows, np.zeros((eq_rows.shape[0], n_ub))],
        ]
    )
    rhs = np.concatenate([ub_rhs, eq_rhs])
    negative = rhs < 0
    table[negative] *= -1.0
    rhs = np.abs(rhs)
    m, width = table.shape

    # Phase 1: minimise the sum of one artificial variable per row.
    phase1 = np.hstack([table, np.eye(m)])
    cost1 = np.concatenate([np.zeros(width), np.ones(m)])
    basis = list(range(width, width + m))
    status = _simplex(phase1, rhs, cost1, basis, tol)
    if status != "optimal" or float(cost1[basis] @ rhs) > np.sqrt(tol):
        return LPResult("infeasible", None, None)
    _pivot_out_artificials(phase1, rhs, basis, width, tol)
    keep = [i for i in range(m) if basis[i] < width]
    table, rhs, basis = phase1[keep, :width], rhs[keep], [basis[i] for i in keep]

    # Phase 2: minimise the real objective from the feasible basis.
    cost2 = np.concatenate([cost, np.zeros(n_ub)])
    status = _simplex(table, rhs, cost2, basis, tol)
    if status != "optimal":
        return LPResult(status, None, None)
    solution = np.zeros(width)
    solution[basis] = rhs
    x = solution[:n]
    # Duals of the <= rows: the slack column for row j in the final tableau is (+-)B^-1 e_j, and
    # the row-flip sign cancels in c_B @ B^-1 e_j, so this is d value / d b_j directly.
    dual_ub = cost2[basis] @ table[:, n:] if n_ub else None
    return LPResult("optimal", x, float(cost @ x), dual_ub)


def _simplex(
    table: FloatArray, rhs: FloatArray, cost: FloatArray, basis: list[int], tol: float
) -> str:
    """Tableau simplex with Bland's rule; mutates ``table``/``rhs``/``basis`` in place."""
    while True:
        reduced = cost - cost[basis] @ table
        candidates = np.flatnonzero(reduced < -tol)
        if candidates.size == 0:
            return "optimal"
        entering = int(candidates[0])  # Bland: smallest eligible index
        column = table[:, entering]
        rows = np.flatnonzero(column > tol)
        if rows.size == 0:
            return "unbounded"
        ratios = rhs[rows] / column[rows]
        best = float(np.min(ratios))
        ties = rows[ratios <= best + tol]
        leaving = int(ties[np.argmin(np.asarray(basis)[ties])])  # Bland: smallest basis index
        _pivot(table, rhs, leaving, entering)
        basis[leaving] = entering


def _pivot(table: FloatArray, rhs: FloatArray, row: int, col: int) -> None:
    pivot = table[row, col]
    table[row] /= pivot
    rhs[row] /= pivot
    others = np.flatnonzero(np.abs(table[:, col]) > 0)
    for i in others:
        if i != row:
            factor = table[i, col]
            table[i] -= factor * table[row]
            rhs[i] -= factor * rhs[row]


def _pivot_out_artificials(
    table: FloatArray, rhs: FloatArray, basis: list[int], width: int, tol: float
) -> None:
    """Replace basic artificials (at zero level) with real columns; redundant rows stay flagged."""
    for i, b in enumerate(basis):
        if b < width:
            continue
        columns = np.flatnonzero(np.abs(table[i, :width]) > tol)
        if columns.size:
            entering = int(columns[0])
            _pivot(table, rhs, i, entering)
            basis[i] = entering
