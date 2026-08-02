"""Functional connectivity with a common-input sensitivity certificate.

A discovered functional edge between two neurons is only as trustworthy as the assumption that
nothing *unrecorded* drives both of them. On a Utah array that assumption is false by
construction: a few hundred electrodes sample a vanishing fraction of the local network, so every
reported edge is exposed to common input from neurons that were never observed.

This module answers the question that exposure raises — **how strong would an unrecorded common
input have to be, to explain this edge away entirely?** — and reports the answer as a
:class:`~causalrl.certify.Certificate`.

The instrument (:func:`common_input_tipping_point`) is a sensitivity analysis in the sense of
Cinelli & Hazlett (*JRSS-B* 2020): rather than assuming no confounding, it computes the strength a
confounder needs in order to overturn the finding, and *benchmarks* that against the strength of
the shared inputs that were actually observed. If explaining an edge away requires a hidden common
input stronger than any shared drive visible among the recorded units, the edge survives with a
witness; if a hidden input no stronger than a typical observed one suffices, the certificate hedges
and says so. Neither outcome is a p-value, and neither pretends the sufficiency assumption holds.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from causalrl.certify.certificate import (
    Assumption,
    Certificate,
    EstimandSpec,
    Hedge,
    Kind,
    Provenance,
    Witness,
)
from causalrl.discovery import CITest
from causalrl.identification.bounds import Interval
from causalrl.neuro.citests import PartialCorrelationTest, PoissonGLMTest
from causalrl.neuro.recording import MultiScaleRecording
from causalrl.neuro.timeseries import LaggedGraph, discover_lagged, lag_name, lagged_frame

__all__ = [
    "CommonInputSensitivity",
    "FunctionalConnectivity",
    "certify_functional_edge",
    "common_input_tipping_point",
    "functional_connectivity",
    "observed_shared_variance",
]

FloatArray = NDArray[np.float64]


def common_input_tipping_point(partial_correlation: float) -> float:
    """Shared-variance fraction an unrecorded common input needs to explain away an edge.

    Take the null in which there is **no** direct link, and the observed association arises purely
    from one unrecorded common input ``Z``:

        X = a·Z + e_x,   Y = b·Z + e_y,   Z, e ~ N(0, 1) independent

    which induces ``rho = a·b / sqrt((1+a²)(1+b²))``. The fraction of ``X``'s variance explained by
    ``Z`` is ``R²_x = a²/(1+a²)``, and likewise for ``Y``. Among all ``(a, b)`` reproducing a given
    ``rho``, the one minimising the *larger* of the two requirements is the symmetric ``a = b``,
    where ``R²_x = R²_y = |rho|``.

    So the tipping point is ``|rho|``, read as: *an unrecorded common input must explain at least
    this fraction of the variance of **both** units to account for the edge on its own.* Being the
    minimum over the whole family, it is the conservative (easiest-to-satisfy) requirement — a
    weaker common input cannot do it, whatever its asymmetry.

    Exact for the linear-Gaussian null; for point-process edges it is applied to the partial
    correlation of the same conditioning set and is an approximation, recorded as such in the
    certificate's assumptions.
    """
    return float(min(abs(partial_correlation), 1.0))


def observed_shared_variance(
    data: Mapping[str, FloatArray],
    variables: Sequence[str],
    *,
    conditioning: Mapping[str, Sequence[str]] | None = None,
    quantile: float = 0.95,
) -> tuple[float, tuple[str, str] | None]:
    """Benchmark scale: the ``quantile`` of squared partial correlations among recorded pairs.

    This is the empirical answer to "how strongly do units in *this* recording actually share
    input?", and it is what a hypothetical unrecorded common input is compared against. A tipping
    point above this scale means the edge could only be explained away by a hidden drive stronger
    than nearly every shared drive that was actually observed.

    Returns ``(scale, argmax_pair)``; the pair is the most strongly shared observed pair, reported
    in the certificate so the benchmark is auditable rather than a bare number.
    """
    test = PartialCorrelationTest()
    names = list(variables)
    values: list[float] = []
    best: tuple[str, str] | None = None
    best_val = -1.0
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            z = list((conditioning or {}).get(a, ())) + list((conditioning or {}).get(b, ()))
            z = [v for v in dict.fromkeys(z) if v not in (a, b)]
            rho = test.partial_correlation(data, a, b, z)
            values.append(rho * rho)
            if rho * rho > best_val:
                best_val, best = rho * rho, (a, b)
    if not values:
        return 0.0, None
    return float(np.quantile(np.asarray(values), quantile)), best


@dataclass(frozen=True)
class CommonInputSensitivity:
    """How much unrecorded common input an edge can absorb before it is explained away."""

    source: str
    target: str
    lag: int
    partial_correlation: float
    tipping_point: float  # required shared-variance fraction in BOTH units
    benchmark: float  # observed shared-variance scale among recorded pairs
    benchmark_pair: tuple[str, str] | None
    robust: bool  # tipping_point > benchmark

    def summary(self) -> str:
        verdict = "survives" if self.robust else "explained away"
        return (
            f"{self.source}@t-{self.lag} -> {self.target}: an unrecorded common input must explain "
            f">= {self.tipping_point:.1%} of the variance of both units to erase this edge; the "
            f"observed shared-input benchmark is {self.benchmark:.1%} ({verdict})"
        )


def certify_functional_edge(
    data: Mapping[str, FloatArray],
    *,
    source: str,
    target: str,
    lag: int,
    conditioning: Sequence[str] = (),
    benchmark: float | None = None,
    benchmark_pair: tuple[str, str] | None = None,
    seeds: tuple[int, ...] = (),
    data_fingerprint: str | None = None,
) -> Certificate:
    """Certify one functional edge against unrecorded common input.

    ``data`` must already be lag-embedded (see :func:`~causalrl.neuro.timeseries.lagged_frame`), so
    ``source`` at ``lag`` and ``target`` at lag 0 are both columns. ``conditioning`` is the set the
    edge survived in discovery — the sensitivity statement is conditional on exactly that set, and
    it is recorded in the certificate.

    The certificate is always :attr:`~causalrl.certify.Kind.BOUNDED`: this is partial
    identification under an explicit sensitivity budget, never a point-identified effect.
    """
    test = PartialCorrelationTest()
    src = lag_name(source, lag)
    rho = test.partial_correlation(data, src, target, list(conditioning))
    tipping = common_input_tipping_point(rho)
    bench = 0.0 if benchmark is None else float(benchmark)
    robust = tipping > bench
    sens = CommonInputSensitivity(
        source=source,
        target=target,
        lag=lag,
        partial_correlation=rho,
        tipping_point=tipping,
        benchmark=bench,
        benchmark_pair=benchmark_pair,
        robust=robust,
    )
    assumptions = (
        Assumption(
            name="linear-gaussian-common-input",
            params={"conditioning": list(conditioning), "lag": lag},
            checkable=True,
            diagnostic={"partial_correlation": rho},
        ),
        Assumption(
            name="benchmark-shared-variance",
            params={
                "scale": bench,
                "pair": list(benchmark_pair) if benchmark_pair else None,
            },
            checkable=True,
        ),
    )
    witness = (
        Witness(
            kind="exceeds-observed-common-input",
            detail={
                "tipping_point": tipping,
                "benchmark": bench,
                "reading": (
                    "no shared drive this strong is visible among the recorded units, so a hidden "
                    "one would have to exceed everything observed"
                ),
            },
        )
        if robust
        else None
    )
    hedge = (
        None
        if robust
        else Hedge(
            reason=(
                "an unrecorded common input no stronger than the observed shared-input benchmark "
                "would account for this edge; it is not distinguishable from common input here"
            ),
            detail={"tipping_point": tipping, "benchmark": bench},
        )
    )
    return Certificate(
        claim=sens.summary(),
        estimand=EstimandSpec(query="see", target="functional-edge"),
        kind=Kind.BOUNDED,
        value=Interval(0.0, tipping),
        alpha=None,
        assumptions=assumptions,
        method="common-input-tipping-point",
        witness=witness,
        hedge=hedge,
        provenance=Provenance.create(seeds=seeds, data_fingerprint=data_fingerprint),
    )


@dataclass(frozen=True)
class FunctionalConnectivity:
    """A discovered functional graph plus its per-edge common-input sensitivity."""

    graph: LaggedGraph
    sensitivities: tuple[CommonInputSensitivity, ...]
    certificates: tuple[Certificate, ...]
    benchmark: float
    scale: str  # "micro" (spikes) or "meso" (population signals)

    def robust_edges(self) -> list[tuple[str, str]]:
        """Edges whose tipping point exceeds the observed shared-input benchmark."""
        return sorted({(s.source, s.target) for s in self.sensitivities if s.robust})

    def fragile_edges(self) -> list[tuple[str, str]]:
        """Edges a plausible unrecorded common input would explain away."""
        return sorted({(s.source, s.target) for s in self.sensitivities if not s.robust})

    def summary(self) -> str:
        n_edges = len(self.graph.lagged_edges())
        return (
            f"{self.scale}: {n_edges} lagged edges, {len(self.robust_edges())} robust to common "
            f"input at benchmark {self.benchmark:.1%}, {len(self.fragile_edges())} fragile; "
            f"{len(self.graph.latent_pairs())} contemporaneous <-> (latent or sub-bin)"
        )


def functional_connectivity(
    recording: MultiScaleRecording,
    *,
    scale: str = "micro",
    max_lag: int = 3,
    ci_test: CITest | None = None,
    alpha: float = 0.001,
    max_conditioning_size: int = 2,
    benchmark_quantile: float = 0.95,
) -> FunctionalConnectivity:
    """Discover functional connectivity at one scale and certify every edge against common input.

    ``scale="micro"`` runs on binned spike counts with a point-process GLM test by default;
    ``scale="meso"`` runs on the population signals with a partial-correlation test. Both go
    through :func:`~causalrl.neuro.timeseries.discover_lagged`, so both control for autocorrelation
    and return a contemporaneous PAG alongside the lagged links.
    """
    if scale == "micro":
        columns = recording.micro_columns()
        default_test: CITest = PoissonGLMTest(alpha=alpha)
    elif scale == "meso":
        columns = recording.meso_columns()
        if not columns:
            raise ValueError("recording carries no mesoscopic signals")
        default_test = PartialCorrelationTest(alpha=alpha)
    else:
        raise ValueError(f"scale must be 'micro' or 'meso', got {scale!r}")

    names = list(columns)
    test = ci_test if ci_test is not None else default_test
    graph = discover_lagged(
        columns,
        names,
        max_lag=max_lag,
        ci_test=test,
        contemporaneous_ci_test=PartialCorrelationTest(alpha=alpha),
        max_conditioning_size=max_conditioning_size,
    )

    frame = lagged_frame(columns, names, max_lag)
    # The benchmark asks "how strongly do units in this recording actually share input?", which is
    # an *unconditioned* quantity. Conditioning each unit on its own past — right for removing
    # autocorrelation when testing an edge — would absorb precisely the slow shared drive the
    # benchmark exists to measure, collapsing it to ~0 and making every edge trivially "robust".
    bench, bench_pair = observed_shared_variance(frame, names, quantile=benchmark_quantile)

    sensitivities: list[CommonInputSensitivity] = []
    certificates: list[Certificate] = []
    fingerprint = recording.fingerprint()
    for link in graph.links:
        if link.source == link.target:
            continue  # a unit's own past is not a connectivity claim
        conditioning = [
            lag_name(other.source, other.lag)
            for other in graph.parents(link.target)
            if not (other.source == link.source and other.lag == link.lag)
        ]
        cert = certify_functional_edge(
            frame,
            source=link.source,
            target=link.target,
            lag=link.lag,
            conditioning=conditioning,
            benchmark=bench,
            benchmark_pair=bench_pair,
            data_fingerprint=fingerprint,
        )
        certificates.append(cert)
        rho = PartialCorrelationTest().partial_correlation(
            frame, lag_name(link.source, link.lag), link.target, conditioning
        )
        tipping = common_input_tipping_point(rho)
        sensitivities.append(
            CommonInputSensitivity(
                source=link.source,
                target=link.target,
                lag=link.lag,
                partial_correlation=rho,
                tipping_point=tipping,
                benchmark=bench,
                benchmark_pair=bench_pair,
                robust=tipping > bench,
            )
        )
    return FunctionalConnectivity(
        graph=graph,
        sensitivities=tuple(sensitivities),
        certificates=tuple(certificates),
        benchmark=bench,
        scale=scale,
    )
