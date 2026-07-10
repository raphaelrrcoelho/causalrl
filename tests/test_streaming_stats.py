"""Phase 3 §9: streaming sufficient statistics — exact parity with one-shot NumPy (pure numpy)."""

from __future__ import annotations

import math

import numpy as np

from causalrl.backends.streaming import StreamingMoments, WeightedStreamingRatio


def test_streaming_moments_match_one_shot() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal(10_000)
    acc = StreamingMoments()
    for batch in np.array_split(x, 37):  # ragged batch sizes
        acc.update(batch)
    assert acc.count == x.shape[0]
    assert math.isclose(acc.mean, float(x.mean()), rel_tol=0, abs_tol=1e-9)
    assert math.isclose(acc.variance, float(x.var(ddof=1)), rel_tol=1e-9)
    assert math.isclose(acc.se, float(x.std(ddof=1) / math.sqrt(x.shape[0])), rel_tol=1e-9)


def test_streaming_moments_merge_is_associative() -> None:
    rng = np.random.default_rng(1)
    x = rng.standard_normal(5_000)
    a, b, c = np.array_split(x, 3)
    bc = StreamingMoments().update(b).merge(StreamingMoments().update(c))
    left = StreamingMoments().update(a).merge(bc)
    flat = StreamingMoments().update(x)
    assert math.isclose(left.mean, flat.mean, rel_tol=1e-9, abs_tol=1e-12)
    assert math.isclose(left.variance, flat.variance, rel_tol=1e-9)


def test_streaming_moments_ci_brackets_mean() -> None:
    acc = StreamingMoments().update(np.arange(100, dtype=float))
    ci = acc.ci(1.96)
    assert ci.lower < acc.mean < ci.upper
    assert math.isclose((ci.upper - ci.lower) / 2.0, 1.96 * acc.se, rel_tol=1e-12)


def test_empty_and_singleton_moments_are_nan() -> None:
    empty = StreamingMoments()
    assert empty.count == 0 and math.isnan(empty.mean) and math.isnan(empty.variance)
    one = StreamingMoments().update([3.0])
    assert one.count == 1 and one.mean == 3.0 and math.isnan(one.variance) and math.isnan(one.se)


def test_weighted_ratio_matches_hajek_full_data() -> None:
    rng = np.random.default_rng(2)
    w = rng.uniform(0.1, 4.0, size=8_000)
    y = rng.standard_normal(8_000)
    acc = WeightedStreamingRatio()
    for wb, yb in zip(np.array_split(w, 19), np.array_split(y, 19), strict=True):
        acc.update(wb, yb)
    # Self-normalised (Hájek) weighted mean.
    assert math.isclose(acc.value, float(np.average(y, weights=w)), rel_tol=1e-9)
    # Influence-function SE == closed form √(Σ w²(y-V)²)/Σw computed on the full data.
    v = float(np.average(y, weights=w))
    se_full = math.sqrt(float((w * w * (y - v) ** 2).sum())) / float(w.sum())
    assert math.isclose(acc.se, se_full, rel_tol=1e-9)


def test_weighted_ratio_uniform_weights_reduce_to_mean() -> None:
    y = np.arange(1, 51, dtype=float)
    acc = WeightedStreamingRatio().update(np.ones_like(y), y)
    assert math.isclose(acc.value, float(y.mean()), rel_tol=1e-12)
    # Uniform weights: ESS == n exactly.
    assert math.isclose(acc.effective_sample_size, float(y.shape[0]), rel_tol=1e-12)


def test_weighted_ratio_ess_drops_with_weight_skew() -> None:
    y = np.zeros(1000)
    skewed = np.concatenate([np.full(1, 1000.0), np.full(999, 1e-3)])
    acc = WeightedStreamingRatio().update(skewed, y)
    assert acc.effective_sample_size < 2.0  # one weight dominates -> tiny ESS


def test_weighted_ratio_merge_matches_single_pass() -> None:
    rng = np.random.default_rng(3)
    w = rng.uniform(0.2, 3.0, size=4_000)
    y = rng.standard_normal(4_000)
    merged = WeightedStreamingRatio()
    for wb, yb in zip(np.array_split(w, 8), np.array_split(y, 8), strict=True):
        part = WeightedStreamingRatio().update(wb, yb)
        merged.merge(part)
    flat = WeightedStreamingRatio().update(w, y)
    assert math.isclose(merged.value, flat.value, rel_tol=1e-12)
    assert math.isclose(merged.se, flat.se, rel_tol=1e-12)
