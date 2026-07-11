"""Streaming certificate kernels over columnar logs (plan §9; invariants I3/I8).

Single-pass estimators that consume a :class:`~causalrl.data.trajectory.TrajectoryLog` — or an
on-disk Parquet log streamed batch-by-batch — and emit a unified
:class:`~causalrl.certify.certificate.Certificate`, never materialising the whole log. The numerics
live in the mergeable accumulators of :mod:`causalrl.backends`; this module joins the columnar cells
into per-decision records (:class:`~causalrl.data.streaming_join.KeyJoiner`) and wraps the result in
an identification-aware certificate.

* :func:`stream_policy_value` — self-normalised (Hájek) importance-sampling off-policy value with a
  confidence interval and an effective-sample-size overlap hedge (I3).
* :func:`stream_quantile_certificate` — a distributional quantile / tail functional via the GK
  sketch, recording the ε rank-error budget in the certificate (I8).
"""

from __future__ import annotations

import hashlib

from causalrl.backends.quantile_sketch import GKQuantileSketch
from causalrl.backends.streaming import WeightedStreamingRatio
from causalrl.certify.certificate import (
    Assumption,
    Certificate,
    EstimandSpec,
    Hedge,
    Kind,
    Provenance,
    Witness,
)
from causalrl.data.streaming_join import KeyJoiner, LogSource, iter_log_batches
from causalrl.estimate._stats import norm_ppf

__all__ = ["stream_policy_value", "stream_quantile_certificate"]


def _stream_fingerprint(*parts: float) -> str:
    """A cheap O(1) content signature of a streamed pass (streamed sums, not the raw log)."""
    blob = ",".join(f"{p:.10g}" for p in parts)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _hedged_policy_value(
    alpha: float, policy: str | None, hedge: Hedge, fingerprint: str
) -> Certificate:
    return Certificate(
        claim=f"off-policy value refused: {hedge.reason}",
        estimand=EstimandSpec(query="policy_value", target="mean", policy=policy),
        kind=Kind.IDENTIFIED,
        value=None,
        alpha=alpha,
        assumptions=(),
        method="refused",
        witness=None,
        hedge=hedge,
        provenance=Provenance.create(data_fingerprint=fingerprint),
        ci=None,
    )


def stream_policy_value(
    source: LogSource,
    *,
    weight: str = "weight",
    reward: str = "reward",
    batch_size: int = 100_000,
    alpha: float = 0.05,
    min_ess_fraction: float = 0.1,
    policy: str | None = None,
) -> Certificate:
    """Certify a policy's self-normalised importance-sampling value, streamed over a columnar log.

    Each decision contributes an importance weight (the ``weight`` value cell,
    ``rho = π_target/π_behaviour``) and a ``reward`` cell; the estimate is the Hájek ratio
    ``V = Σ(rho·r) / Σ rho`` with an influence-function CI. Returns a ``kind=IDENTIFIED``
    :class:`Certificate` (value + ``ci``) under logged-propensity positivity, else a hedge
    when the Kish effective sample size falls below ``min_ess_fraction`` of the decision count — the
    streaming positivity guard (I3). ``source`` is a :class:`TrajectoryLog` or a Parquet path,
    streamed in row batches of ``batch_size`` without materialising it.
    """
    joiner = KeyJoiner((weight, reward))
    ratio = WeightedStreamingRatio()
    n_batches = 0
    for log in iter_log_batches(source, batch_size):
        n_batches += 1
        cols = joiner.drain(log)
        w = cols[weight]
        if w.shape[0]:
            ratio.update(w, cols[reward])

    n = ratio.count
    fingerprint = _stream_fingerprint(float(n), ratio.sum_weights, ratio.value if n else 0.0)
    if n == 0:
        return _hedged_policy_value(
            alpha, policy, Hedge("no-decisions", {"weight": weight, "reward": reward}), fingerprint
        )
    ess = ratio.effective_sample_size
    ess_fraction = ess / n
    if ess_fraction < min_ess_fraction:
        return _hedged_policy_value(
            alpha,
            policy,
            Hedge(
                "overlap-violation",
                {
                    "ess": ess,
                    "n": n,
                    "ess_fraction": ess_fraction,
                    "min_ess_fraction": min_ess_fraction,
                },
            ),
            fingerprint,
        )

    value = ratio.value
    z = float(norm_ppf(1.0 - alpha / 2.0))
    return Certificate(
        claim=f"IS off-policy value V(π) = {value:.4g}",
        estimand=EstimandSpec(query="policy_value", target="mean", policy=policy),
        kind=Kind.IDENTIFIED,
        value=value,
        alpha=alpha,
        assumptions=(
            Assumption(
                name="overlap",
                params={"min_ess_fraction": min_ess_fraction},
                checkable=True,
                diagnostic={"ess": ess, "ess_fraction": ess_fraction, "n": n},
            ),
            Assumption(name="logged-propensities", params={}, checkable=False),
        ),
        method=f"self-normalised IPW (streaming, {n_batches} batch(es), dropped={joiner.dropped})",
        witness=Witness(kind="importance-weighting", detail={"weight": weight, "reward": reward}),
        hedge=None,
        provenance=Provenance.create(data_fingerprint=fingerprint),
        ci=ratio.ci(z),
    )


def stream_quantile_certificate(
    source: LogSource,
    *,
    name: str,
    q: float,
    epsilon: float = 0.01,
    batch_size: int = 100_000,
    alpha: float | None = None,
) -> Certificate:
    """Certify a distributional ``q``-quantile of the ``name`` column via a streaming GK sketch.

    Returns a ``kind=IDENTIFIED`` :class:`Certificate` whose ``value`` is the sketch's quantile
    estimate; the ε rank-error budget is recorded as a checkable ``quantile-sketch`` assumption so
    the guarantee (true rank within ``ε·n`` of ``q·n``) travels with the claim (I8). The quantile is
    a distributional summary of the observed ``name`` column, not a causal effect, so the query is
    observational (``see``). ``source`` streams in row batches without materialising the log.
    """
    sketch = GKQuantileSketch(epsilon)
    n_rows = 0
    for log in iter_log_batches(source, batch_size):
        cells = log.values_by_name(name)
        if cells.shape[0]:
            values = [float(v) for v in cells.tolist()]
            sketch.update(values)
            n_rows += len(values)

    fingerprint = _stream_fingerprint(float(n_rows), q, epsilon)
    if sketch.count == 0:
        return Certificate(
            claim=f"quantile refused: column {name!r} absent",
            estimand=EstimandSpec(query="see", target="quantile"),
            kind=Kind.IDENTIFIED,
            value=None,
            alpha=alpha,
            assumptions=(),
            method="refused",
            witness=None,
            hedge=Hedge("no-observations", {"name": name}),
            provenance=Provenance.create(data_fingerprint=fingerprint),
            ci=None,
        )

    q_hat = sketch.quantile(q)
    return Certificate(
        claim=f"quantile_{q:g}({name}) = {q_hat:.4g}",
        estimand=EstimandSpec(query="see", target="quantile"),
        kind=Kind.IDENTIFIED,
        value=q_hat,
        alpha=alpha,
        assumptions=(
            Assumption(
                name="quantile-sketch",
                params={"epsilon": epsilon, "q": q},
                checkable=True,
                diagnostic={"rank_error_fraction": sketch.error_bound, "n": sketch.count},
            ),
        ),
        method=f"GK streaming sketch (ε={epsilon:g})",
        witness=Witness(
            kind="empirical-quantile", detail={"name": name, "q": q, "n": sketch.count}
        ),
        hedge=None,
        provenance=Provenance.create(data_fingerprint=fingerprint),
        ci=None,
    )
