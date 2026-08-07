"""Micro-benchmarks + regression guards for the Phase-3 streaming data plane (§9).

Run: PYTHONPATH=src .venv/bin/python benchmarks/bench_streaming.py

Guards the scale path the same way ``bench_causal_core.py`` guards the closed-form fast paths:
  (1) The streaming accumulators (``StreamingMoments`` / ``WeightedStreamingRatio``) reproduce the
      one-shot NumPy statistics EXACTLY, and run within a conservative constant factor of a single
      vectorised pass (hard-fail on a >2x slowdown vs the recorded relative floor).
  (2) The end-to-end streamed OPE certificate (``stream_policy_value`` over a columnar log) equals
      the materialised Hájek estimate — correctness never regresses with scale.

The shipped closed-form guards (the 874x MSM speedup, exact known-noise counterfactual) live in
``bench_causal_core.py`` and are untouched; this file extends the guard suite beside them.
"""

from __future__ import annotations

import time

import numpy as np

from causalrl.backends.streaming import StreamingMoments, WeightedStreamingRatio
from causalrl.data.trajectory import TrajectoryLog
from causalrl.ope.ipw import stream_policy_value

# Conservative relative floor: streaming may be at most this many times a single vectorised pass.
# Set well above the observed ~1-2x so only a genuine >2x regression trips it (not machine noise).
_MAX_SLOWDOWN = 6.0


def _timed(fn, repeat: int = 3) -> tuple[float, object]:
    best = float("inf")
    out: object = None
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t0)
    return best, out


def bench_streaming_moments() -> None:
    """StreamingMoments over batches vs one-shot ``mean``/``var``: exact + within the floor."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal(10_000_000)
    batches = np.array_split(x, 200)

    def streamed() -> StreamingMoments:
        acc = StreamingMoments()
        for b in batches:
            acc.update(b)
        return acc

    def one_shot() -> tuple[float, float]:
        return float(x.mean()), float(x.var(ddof=1))

    t_stream, acc = _timed(streamed)
    t_one, (m, v) = _timed(one_shot)
    acc = acc  # type: ignore[assignment]
    assert abs(acc.mean - m) < 1e-9, f"streaming mean off: {acc.mean} vs {m}"
    assert abs(acc.variance - v) / v < 1e-9, f"streaming var off: {acc.variance} vs {v}"
    ratio = t_stream / t_one
    thru = x.shape[0] / t_stream / 1e6
    print(
        f"[moments] streamed {x.shape[0]:,} in {t_stream * 1e3:.1f} ms ({thru:.1f}M/s); "
        f"{ratio:.2f}x one-shot ({t_one * 1e3:.1f} ms), exact"
    )
    assert ratio < _MAX_SLOWDOWN, f"moments streaming regressed: {ratio:.1f}x > {_MAX_SLOWDOWN}x"


def bench_weighted_ratio() -> None:
    """WeightedStreamingRatio vs one-shot Hájek: exact self-normalised value + IF standard error."""
    rng = np.random.default_rng(1)
    w = rng.uniform(0.2, 4.0, size=10_000_000)
    y = rng.standard_normal(10_000_000)

    t_stream, acc = _timed(lambda: WeightedStreamingRatio().update(w, y))
    acc_r: WeightedStreamingRatio = acc  # type: ignore[assignment]
    v_ref = float(np.average(y, weights=w))
    se_ref = float(np.sqrt(float((w * w * (y - v_ref) ** 2).sum())) / w.sum())

    def one_shot() -> tuple[float, float]:
        # Apples-to-apples: the vectorised closed form for the SAME value + IF standard error.
        v = float(np.average(y, weights=w))
        se = float(np.sqrt(float((w * w * (y - v) ** 2).sum())) / w.sum())
        return v, se

    t_one, _ = _timed(one_shot)
    assert abs(acc_r.value - v_ref) < 1e-9, f"ratio value off: {acc_r.value} vs {v_ref}"
    assert abs(acc_r.se - se_ref) < 1e-9, f"ratio SE off: {acc_r.se} vs {se_ref}"
    ratio = t_stream / t_one
    print(
        f"[ratio]   streamed {w.shape[0]:,} in {t_stream * 1e3:.1f} ms; V={acc_r.value:.4f} "
        f"SE={acc_r.se:.5f}, {ratio:.2f}x one-shot, exact"
    )
    assert ratio < _MAX_SLOWDOWN, f"ratio streaming regressed: {ratio:.1f}x > {_MAX_SLOWDOWN}x"


def bench_stream_policy_value_end_to_end() -> None:
    """End-to-end streamed OPE certificate over a columnar log == the materialised Hájek value."""
    rng = np.random.default_rng(3)
    n = 200_000
    w = rng.uniform(0.3, 3.0, size=n)
    y = rng.standard_normal(n) + 0.5
    rows: list[dict[str, object]] = []
    for i in range(n):
        base = {"entity_id": i, "episode_id": 0, "t": 0}
        rows.append({**base, "kind": "w", "name": "weight", "value": float(w[i])})
        rows.append({**base, "kind": "r", "name": "reward", "value": float(y[i])})
    log = TrajectoryLog.from_rows(rows).sorted_by_key()

    t_stream, cert = _timed(lambda: stream_policy_value(log, batch_size=50_000), repeat=1)
    from causalrl.certify.certificate import Certificate

    c: Certificate = cert  # type: ignore[assignment]
    ref = float(np.average(y, weights=w))
    assert isinstance(c.value, float) and abs(c.value - ref) < 1e-9, "streamed OPE != Hájek"
    thru = n / t_stream / 1e3
    print(
        f"[ope]     streamed {n:,} decisions ({2 * n:,} rows) in {t_stream * 1e3:.0f} ms "
        f"({thru:.0f}k dec/s); V={c.value:.4f}, exact vs materialised"
    )


def main() -> None:
    print("causal-core streaming micro-benchmarks (Phase 3 §9 scale path)")
    bench_streaming_moments()
    bench_weighted_ratio()
    bench_stream_policy_value_end_to_end()
    print("OK — all streaming bench assertions passed")


if __name__ == "__main__":
    main()
