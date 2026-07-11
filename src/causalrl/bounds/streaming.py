"""Streaming marginal-sensitivity-model bounds over a columnar log (plan §9; invariant I2).

:func:`stream_msm_bounds` streams the log, extracting only the two float columns the closed-form Tan
bound needs — the treated units' outcomes and nominal propensities — instead of holding the whole
long log, then applies the exact ``O(n log n)`` closed form
(:func:`causalrl.identification.bounds.ipw_sensitivity_bounds`). The result is a ``kind=BOUNDED``
:class:`~causalrl.certify.certificate.Certificate`: partial identification of the treated
counterfactual mean ``E[Y(1)]`` under an odds-ratio confounding budget ``gamma`` — never a point
estimate (I2/I3).
"""

from __future__ import annotations

import hashlib

import numpy as np
from numpy.typing import NDArray

from causalrl.certify.certificate import (
    Assumption,
    Certificate,
    EstimandSpec,
    Hedge,
    Kind,
    Provenance,
)
from causalrl.data.streaming_join import KeyJoiner, LogSource, iter_log_batches
from causalrl.identification.bounds import ipw_sensitivity_bounds

__all__ = ["stream_msm_bounds"]

FloatArray = NDArray[np.float64]


def _fingerprint(n: float, gamma: float) -> str:
    return hashlib.sha256(f"{n:.10g},{gamma:.10g}".encode()).hexdigest()[:16]


def stream_msm_bounds(
    source: LogSource,
    *,
    gamma: float,
    outcome: str = "reward",
    propensity: str = "propensity",
    treatment: str | None = None,
    batch_size: int = 100_000,
) -> Certificate:
    """Certify MSM bounds on ``E[Y(1)]`` from a streamed log under confounding budget ``gamma``.

    Each decision supplies an ``outcome`` cell and a nominal ``propensity`` cell; when ``treatment``
    is given, only decisions with that indicator cell ``> 0.5`` (the treated units) enter the bound.
    Returns a ``kind=BOUNDED`` :class:`Certificate` whose ``value`` is the Tan interval: it holds
    ``E[Y(1)]`` whenever the true confounding odds ratio is at most ``gamma``, collapses to the IPW
    point at ``gamma = 1``, and widens monotonically. ``source`` streams in row batches of
    ``batch_size`` without materialising the log (only the two needed columns accumulate).
    """
    if gamma < 1.0:
        raise ValueError("gamma must be >= 1")
    names = (outcome, propensity) if treatment is None else (outcome, propensity, treatment)
    joiner = KeyJoiner(names)
    y_chunks: list[FloatArray] = []
    e_chunks: list[FloatArray] = []
    n_batches = 0
    for log in iter_log_batches(source, batch_size):
        n_batches += 1
        cols = joiner.drain(log)
        y = cols[outcome]
        e = cols[propensity]
        if treatment is not None:
            mask = cols[treatment] > 0.5
            y = y[mask]
            e = e[mask]
        if y.shape[0]:
            y_chunks.append(y)
            e_chunks.append(e)

    y = np.concatenate(y_chunks) if y_chunks else np.empty(0, dtype=np.float64)
    e = np.concatenate(e_chunks) if e_chunks else np.empty(0, dtype=np.float64)
    n = int(y.shape[0])
    fingerprint = _fingerprint(float(n), gamma)
    if n == 0:
        return Certificate(
            claim="MSM bounds refused: no treated units in stream",
            estimand=EstimandSpec(query="do", target="mean"),
            kind=Kind.BOUNDED,
            value=None,
            alpha=None,
            assumptions=(),
            method="refused",
            witness=None,
            hedge=Hedge("no-treated-units", {"outcome": outcome, "propensity": propensity}),
            provenance=Provenance.create(data_fingerprint=fingerprint),
            ci=None,
        )

    interval = ipw_sensitivity_bounds(y.tolist(), e.tolist(), gamma=gamma, return_certificate=False)
    return Certificate(
        claim=f"E[Y(1)] ∈ [{interval.lower:.4g}, {interval.upper:.4g}] under Γ={gamma:g}",
        estimand=EstimandSpec(query="do", target="mean"),
        kind=Kind.BOUNDED,
        value=interval,
        alpha=None,
        assumptions=(
            Assumption(name="MSM", params={"gamma": gamma}, checkable=False),
            Assumption(name="logged-propensities", params={}, checkable=False),
        ),
        method=(
            f"Tan MSM closed form (streamed, {n_batches} batches, n={n}, dropped={joiner.dropped})"
        ),
        witness=None,
        hedge=None,
        provenance=Provenance.create(data_fingerprint=fingerprint),
        ci=None,
    )
