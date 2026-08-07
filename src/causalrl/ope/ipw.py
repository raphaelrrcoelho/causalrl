"""Importance-sampling off-policy value: the in-memory kernel and its streaming certificate.

Two ways to answer "what would this target policy have scored on these logs?" by reweighting the
logged rewards:

* :func:`ipw_value` — the plain inverse-propensity-weighted estimate over in-memory lists.
* :func:`stream_policy_value` — the self-normalised (Hájek) importance-sampling value with a
  confidence interval and an effective-sample-size overlap hedge (I3), streamed single-pass over a
  :class:`~causalrl.data.trajectory.TrajectoryLog` — or an on-disk Parquet log consumed
  batch-by-batch — and returned as a unified
  :class:`~causalrl.certify.certificate.Certificate`, never materialising the whole log. The
  numerics live in the mergeable accumulators of :mod:`causalrl.backends`; this module joins the
  columnar cells into per-decision records
  (:class:`~causalrl.data.streaming_join.KeyJoiner`) and wraps the result in an
  identification-aware certificate.
"""

from __future__ import annotations

import hashlib

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

__all__ = ["ipw_value", "stream_policy_value"]


def ipw_value(
    actions: list[int],
    rewards: list[float],
    behavior_probs: list[float],
    target_probs: list[float],
) -> float:
    """Inverse-propensity-weighted off-policy value estimate.

    Samples with zero behavior propensity (``b == 0``) contribute 0 to the sum but are
    still counted in the ``n`` denominator, which biases the estimate toward 0 — pass only
    samples with positive behavior propensity for an unbiased estimate.
    """
    n = len(actions)
    total = 0.0
    for _a, r, b, t in zip(actions, rewards, behavior_probs, target_probs, strict=True):
        total += (t / b) * r if b > 0 else 0.0
    return total / n


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
