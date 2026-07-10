"""Streaming quantile sketch with a deterministic rank-error bound (plan §9; invariants I3/I8).

:class:`GKQuantileSketch` is a Greenwald-Khanna ε-approximate quantile summary: it answers any
quantile query from a stream of arbitrary length in ``O((1/ε)·log(εn))`` space, and every answer's
*true* rank is within ``ε·n`` of the requested rank — a hard, data-independent guarantee (unlike a
sampling sketch). That bound is exposed as :attr:`error_bound` so a certificate can record it in
provenance rather than assert an unqualified quantile (I3/I8). It is the tail-target backbone of the
streaming certificate kernels: heavy-tailed / quantile functionals over logs too large to sort.

References: M. Greenwald & S. Khanna, *Space-Efficient Online Computation of Quantile Summaries*
(SIGMOD 2001). Formula-level implementation; no third-party code is ported.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = ["GKQuantileSketch"]

FloatArray = NDArray[np.float64]


@dataclass
class _Entry:
    """One summary tuple: a value ``v`` with min-rank gap ``g`` and rank uncertainty ``delta``."""

    v: float
    g: int
    delta: int


class GKQuantileSketch:
    """A Greenwald-Khanna ε-approximate streaming quantile summary.

    ``epsilon`` is the guaranteed maximum rank error as a fraction of ``n``: :meth:`quantile`
    returns a value whose true rank lies in ``[q·n - ε·n, q·n + ε·n]``. Summaries are mergeable; a
    merged summary keeps the ε guarantee by inflating each tuple's uncertainty with the adjacent
    uncertainty from the other summary (the standard mergeable-GK combine).
    """

    def __init__(self, epsilon: float = 0.01) -> None:
        if not 0.0 < epsilon < 1.0:
            raise ValueError("epsilon must lie in (0, 1)")
        self.epsilon = float(epsilon)
        self._entries: list[_Entry] = []
        self._n = 0
        self._compress_every = max(1, math.floor(1.0 / (2.0 * epsilon)))
        self._since_compress = 0

    @property
    def count(self) -> int:
        return self._n

    @property
    def error_bound(self) -> float:
        """Guaranteed maximum rank error as a fraction of ``n`` (record this in provenance)."""
        return self.epsilon

    def update(self, values: FloatArray | Sequence[float]) -> GKQuantileSketch:
        """Absorb a batch of observations; returns ``self`` for chaining."""
        x = np.asarray(values, dtype=np.float64).reshape(-1)
        for val in x.tolist():
            self._insert_one(float(val))
        return self

    def _insert_one(self, v: float) -> None:
        i = self._bisect(v)
        delta = 0 if (i == 0 or i == len(self._entries)) else self._band_capacity() - 1
        self._entries.insert(i, _Entry(v, 1, max(delta, 0)))
        self._n += 1
        self._since_compress += 1
        if self._since_compress >= self._compress_every:
            self._compress()
            self._since_compress = 0

    def _bisect(self, v: float) -> int:
        """First index ``i`` with ``entries[i].v >= v`` (entries stay sorted by ``v``)."""
        lo, hi = 0, len(self._entries)
        while lo < hi:
            mid = (lo + hi) // 2
            if self._entries[mid].v < v:
                lo = mid + 1
            else:
                hi = mid
        return lo

    def _band_capacity(self) -> int:
        return math.floor(2.0 * self.epsilon * self._n)

    def _compress(self) -> None:
        if self._n == 0:
            return
        threshold = self._band_capacity()
        i = len(self._entries) - 2
        while i >= 1:
            cur, nxt = self._entries[i], self._entries[i + 1]
            if cur.g + nxt.g + nxt.delta <= threshold:
                nxt.g += cur.g
                del self._entries[i]
            i -= 1

    def merge(self, other: GKQuantileSketch) -> GKQuantileSketch:
        """Fold another sketch into this one, preserving the ε guarantee; returns ``self``."""
        if other._n == 0:
            return self
        if self._n == 0:
            self._entries = [_Entry(e.v, e.g, e.delta) for e in other._entries]
            self._n = other._n
            self._recompress_after_merge()
            return self
        a = [_Entry(e.v, e.g, e.delta) for e in self._entries]
        b = [_Entry(e.v, e.g, e.delta) for e in other._entries]
        merged = _combine(a, b)
        self._entries = merged
        self._n = self._n + other._n
        self._recompress_after_merge()
        return self

    def _recompress_after_merge(self) -> None:
        self._since_compress = 0
        self._compress()

    def quantile(self, q: float) -> float:
        """Return a value whose true rank is within ``ε·n`` of ``q·n`` (``q`` in ``[0, 1]``)."""
        if self._n == 0:
            raise ValueError("cannot query an empty sketch")
        q = min(max(q, 0.0), 1.0)
        if q <= 0.0:
            return self._entries[0].v
        if q >= 1.0:
            return self._entries[-1].v
        rank = q * self._n
        margin = self.epsilon * self._n
        rmin = 0
        for e in self._entries:
            rmin_i = rmin + e.g
            rmax_i = rmin_i + e.delta
            if rank - rmin_i <= margin and rmax_i - rank <= margin:
                return e.v
            rmin = rmin_i
        return self._entries[-1].v


def _combine(a: list[_Entry], b: list[_Entry]) -> list[_Entry]:
    """Mergeable-GK combine: union both summaries, inflating each tuple's ``delta`` by the adjacent
    uncertainty (``g + delta`` of the next strictly-greater tuple) contributed by the other summary.
    """
    # Precompute, for every entry, the (g + delta) of the first strictly-greater entry in the OTHER
    # list — the extra max-rank the other summary could add at that value.
    b_vals = [e.v for e in b]
    a_vals = [e.v for e in a]
    out: list[_Entry] = []
    for e in a:
        out.append(_Entry(e.v, e.g, e.delta + _adjacent_uncertainty(e.v, b, b_vals)))
    for e in b:
        out.append(_Entry(e.v, e.g, e.delta + _adjacent_uncertainty(e.v, a, a_vals)))
    out.sort(key=lambda t: t.v)
    return out


def _adjacent_uncertainty(v: float, other: list[_Entry], other_vals: list[float]) -> int:
    """``g + delta`` of the first entry in ``other`` whose value is strictly greater than ``v``."""
    lo, hi = 0, len(other_vals)
    while lo < hi:
        mid = (lo + hi) // 2
        if other_vals[mid] <= v:
            lo = mid + 1
        else:
            hi = mid
    if lo >= len(other):
        return 0
    nxt = other[lo]
    return max(nxt.g + nxt.delta - 1, 0)
