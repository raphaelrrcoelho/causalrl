"""Phase 3 §9: Greenwald-Khanna streaming quantile sketch — the ε rank-error guarantee (numpy)."""

from __future__ import annotations

import numpy as np
import pytest

from causalrl.backends.quantile_sketch import GKQuantileSketch


def _true_rank(sorted_data: np.ndarray, v: float) -> int:
    """Max rank of ``v`` in ``sorted_data`` (count of elements <= v)."""
    return int(np.searchsorted(sorted_data, v, side="right"))


@pytest.mark.parametrize("eps", [0.05, 0.01])
def test_rank_error_within_epsilon_single_stream(eps: float) -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal(100_000)
    sketch = GKQuantileSketch(epsilon=eps)
    for batch in np.array_split(x, 50):
        sketch.update(batch)
    assert sketch.count == x.shape[0]
    srt = np.sort(x)
    n = x.shape[0]
    for q in (0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99):
        v = sketch.quantile(q)
        rank = _true_rank(srt, v)
        assert abs(rank - q * n) <= eps * n + 2  # +2 absorbs the searchsorted-side / floor slack


def test_space_is_sublinear() -> None:
    x = np.random.default_rng(1).standard_normal(50_000)
    sketch = GKQuantileSketch(epsilon=0.01)
    sketch.update(x)
    # O((1/eps) log(eps n)) entries — a tiny fraction of the 50k stream.
    assert len(sketch._entries) < 2_000


def test_extremes_and_error_bound() -> None:
    x = np.linspace(-3.0, 5.0, 4_000)
    sketch = GKQuantileSketch(epsilon=0.02).update(x)
    assert sketch.quantile(0.0) == pytest.approx(-3.0, abs=1e-9)
    assert sketch.quantile(1.0) == pytest.approx(5.0, abs=1e-9)
    assert sketch.error_bound == 0.02


def test_empty_sketch_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        GKQuantileSketch().quantile(0.5)


def test_bad_epsilon_rejected() -> None:
    for bad in (0.0, 1.0, -0.1, 2.0):
        with pytest.raises(ValueError, match="epsilon"):
            GKQuantileSketch(epsilon=bad)


def test_heavy_tailed_quantiles() -> None:
    # Pareto (infinite variance for shape < 2): the median is still well-estimated by rank.
    rng = np.random.default_rng(2)
    x = rng.pareto(1.5, size=80_000)
    sketch = GKQuantileSketch(epsilon=0.01).update(x)
    srt = np.sort(x)
    n = x.shape[0]
    for q in (0.5, 0.9, 0.99):
        rank = _true_rank(srt, sketch.quantile(q))
        assert abs(rank - q * n) <= 0.01 * n + 2


def test_merge_preserves_rank_guarantee() -> None:
    rng = np.random.default_rng(3)
    a = rng.standard_normal(40_000)
    b = rng.standard_normal(60_000) + 1.5  # different location -> nontrivial interleave
    sa = GKQuantileSketch(epsilon=0.01).update(a)
    sb = GKQuantileSketch(epsilon=0.01).update(b)
    sa.merge(sb)
    combined = np.sort(np.concatenate([a, b]))
    n = combined.shape[0]
    assert sa.count == n
    for q in (0.1, 0.25, 0.5, 0.75, 0.9):
        rank = _true_rank(combined, sa.quantile(q))
        assert abs(rank - q * n) <= 2.0 * 0.01 * n + 2  # merged: bounded rank error (<= 2 eps n)
