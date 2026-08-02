"""Time-series causal discovery: lagged links, contemporaneous latents, and a summary graph.

:func:`causalrl.discovery.discover` assumes i.i.d. draws. Neural recordings are neither
independent nor unordered: they are strongly autocorrelated time series, and autocorrelation alone
manufactures spurious conditional dependence between channels. This module supplies the missing
front-end.

:func:`discover_lagged` implements the PCMCI scheme of Runge et al. (*Sci. Adv.* 2019) over an
explicit lag embedding:

1. **PC₁ condition selection.** For each target, iteratively prune the candidate parent set
   ``{(source, lag)}`` by conditioning on the strongest parents found so far. Cheap, and it keeps
   later conditioning sets small.
2. **MCI test.** Re-test each surviving link conditioning on the parents of *both* endpoints (the
   source's parents shifted by the link's lag). Conditioning on the source's own past is what
   removes the false positives that autocorrelation would otherwise create — the reason a plain PC
   run on time series is not trustworthy.
3. **Orientation.** Links with lag ≥ 1 are oriented past → present by time order; no Meek
   ambiguity, no equivalence class. The contemporaneous (lag-0) slice is handed to the shipped FCI
   implementation with the lagged parents pinned into every conditioning set (via
   :class:`ConditionedCITest`), so it returns a **PAG** in which ``<->`` marks a latent common
   cause.

**Reading the contemporaneous slice.** At bin width ``dt`` an interaction faster than one bin
cannot be resolved in time, so a lag-0 edge means "common input, or an interaction faster than
``dt``" — never a resolved directed synaptic effect. That ambiguity is a property of the sampling,
not a defect of the algorithm, and the PAG's ``<->`` records it honestly instead of reporting a
direction the data cannot support. Narrowing ``dt`` moves interactions from the lag-0 slice into
the lagged slice; that sensitivity is worth reporting alongside any result.

The output :class:`LaggedGraph` converts into the library's own graph types —
:meth:`~LaggedGraph.unrolled_admg` (always acyclic, so the full identification stack applies) and
:meth:`~LaggedGraph.summary_graph` (cyclic, as cortex is) — which is what connects spike-train
functional connectivity to identification, transport and certification.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from causalrl.discovery import PAG, CITest, discover_latent
from causalrl.exceptions import CausalGraphError
from causalrl.neuro.citests import CITestResult, PartialCorrelationTest

__all__ = [
    "ConditionedCITest",
    "LaggedGraph",
    "LaggedLink",
    "discover_lagged",
    "lag_name",
    "lagged_frame",
]

FloatArray = NDArray[np.float64]


def lag_name(variable: str, lag: int) -> str:
    """Name of ``variable`` at lag ``lag`` in a lag embedding (``lag=0`` keeps the plain name)."""
    if lag < 0:
        raise CausalGraphError(f"lag must be non-negative, got {lag}")
    return variable if lag == 0 else f"{variable}@t-{lag}"


def lagged_frame(
    data: Mapping[str, FloatArray], variables: Sequence[str], max_lag: int
) -> dict[str, FloatArray]:
    """Lag-embed ``data``: one column per ``(variable, lag)`` for ``lag = 0 … max_lag``.

    All columns are truncated to the ``n - max_lag`` rows on which every lag is defined, so the
    embedding is rectangular and every conditioning set is evaluated on the same sample.
    """
    if max_lag < 0:
        raise CausalGraphError("max_lag must be non-negative")
    cols: dict[str, FloatArray] = {}
    n = None
    for v in variables:
        if v not in data:
            raise CausalGraphError(f"variable not in data: {v!r}")
        arr = np.asarray(data[v], dtype=np.float64).reshape(-1)
        if n is None:
            n = arr.shape[0]
        elif arr.shape[0] != n:
            raise CausalGraphError("all series must have the same length")
        for lag in range(max_lag + 1):
            # x_{t-lag} aligned to t = max_lag … n-1, so every column has n - max_lag rows.
            cols[lag_name(v, lag)] = arr[max_lag - lag : arr.shape[0] - lag]
    if n is None:
        raise CausalGraphError("no variables given")
    if n <= max_lag:
        raise CausalGraphError(f"series of length {n} is too short for max_lag={max_lag}")
    return cols


@dataclass(frozen=True)
class ConditionedCITest:
    """A :class:`~causalrl.discovery.CITest` that pins extra conditioning variables into every test.

    ``always[v]`` lists variables added to the conditioning set whenever ``v`` is one of the two
    tested endpoints. In lagged discovery these are the lagged parents, which turns the shipped
    FCI into the contemporaneous phase of PCMCI⁺ without residualising the data or duplicating the
    orientation rules.
    """

    base: CITest
    always: Mapping[str, Sequence[str]]

    def __call__(
        self, data: Mapping[str, np.ndarray], x: str, y: str, z: Sequence[str]
    ) -> object:
        extra: list[str] = []
        for endpoint in (x, y):
            for v in self.always.get(endpoint, ()):
                if v not in (x, y) and v not in z and v not in extra:
                    extra.append(v)
        return self.base(data, x, y, [*z, *extra])


@dataclass(frozen=True)
class LaggedLink:
    """A directed link ``source@t-lag -> target@t`` surviving the MCI test."""

    source: str
    target: str
    lag: int
    statistic: float
    p_value: float | None

    def render(self) -> str:
        lag = f"@t-{self.lag}" if self.lag else "@t"
        p = "" if self.p_value is None else f", p={self.p_value:.2g}"
        return f"{self.source}{lag} -> {self.target}@t (stat={self.statistic:.3g}{p})"


@dataclass(frozen=True)
class LaggedGraph:
    """Result of :func:`discover_lagged`: time-ordered lagged links plus a contemporaneous PAG."""

    variables: tuple[str, ...]
    max_lag: int
    links: tuple[LaggedLink, ...]  # lag >= 1 only
    contemporaneous: PAG
    method: str = ""

    def parents(self, target: str) -> tuple[LaggedLink, ...]:
        """Lagged parents of ``target``, strongest first."""
        if target not in self.variables:
            raise CausalGraphError(f"unknown variable: {target!r}")
        got = [ln for ln in self.links if ln.target == target]
        return tuple(sorted(got, key=lambda ln: -abs(ln.statistic)))

    def lagged_edges(self, *, include_self: bool = False) -> list[tuple[str, str]]:
        """Distinct ``(source, target)`` pairs carrying at least one lagged link.

        Self-links are excluded by default: a unit's own past is always a strong predictor of its
        present (refractoriness, bursting, rate drift), so recovering it is not evidence about
        connectivity. They are kept in the model — conditioned on in every MCI test, which is what
        controls autocorrelation-driven false positives — and reported by :meth:`self_links`.
        """
        return sorted(
            {(ln.source, ln.target) for ln in self.links if include_self or ln.source != ln.target}
        )

    def self_links(self) -> tuple[LaggedLink, ...]:
        """Links from a variable's own past to its present (autocorrelation / refractoriness)."""
        return tuple(ln for ln in self.links if ln.source == ln.target)

    def latent_pairs(self) -> list[tuple[str, str]]:
        """Contemporaneous ``<->`` pairs: a **definite** latent common cause or sub-bin interaction.

        These are the edges FCI could orient with arrowheads at both ends, which requires enough
        surrounding structure (a collider) to rule out either direction. Absence from this list is
        not evidence against common input — see :meth:`contemporaneous_ambiguous`.
        """
        return sorted(
            (a, b) for a, b, _, _ in self.contemporaneous.edges()
            if self.contemporaneous.is_bidirected(a, b)
        )

    def contemporaneous_ambiguous(self) -> list[tuple[str, str]]:
        """Contemporaneous edges carrying a circle mark — direction undetermined by the data.

        An ``a o-o b`` edge is consistent with ``a -> b``, ``b -> a`` *and* ``a <-> b``. For
        connectivity that is the operative statement: the pair is associated at zero lag, and the
        recording cannot distinguish a synaptic effect faster than one bin from an unrecorded
        common input. Reporting it as a directed functional edge would be an overclaim, which is
        why these are listed separately from :meth:`contemporaneous_directed`.
        """
        out: list[tuple[str, str]] = []
        for a, b, mark_a, mark_b in self.contemporaneous.edges():
            if "o" in (mark_a, mark_b):
                out.append((a, b))
        return sorted(out)

    def common_input_candidates(self) -> list[tuple[str, str]]:
        """Contemporaneous pairs that may reflect common input: definite ``<->`` or ambiguous."""
        return sorted(set(self.latent_pairs()) | set(self.contemporaneous_ambiguous()))

    def contemporaneous_directed(self) -> list[tuple[str, str]]:
        """Contemporaneous edges FCI could orient as ``a -> b``."""
        return sorted(
            (a, b) if self.contemporaneous.is_directed(a, b) else (b, a)
            for a, b, _, _ in self.contemporaneous.edges()
            if self.contemporaneous.is_directed(a, b) or self.contemporaneous.is_directed(b, a)
        )

    def unrolled_admg(self) -> object:
        """The time-unrolled graph as an acyclic :class:`~causalrl.scm.graph.CausalGraph`.

        Nodes are ``lag_name(v, lag)``. The unrolling is acyclic *by construction* — every lagged
        edge points forward in time — so the whole identification stack (``identify_effect``,
        POMIS, transport, certification) applies to spike-train functional connectivity through
        this view, which the cyclic summary graph cannot offer. Contemporaneous ``<->`` edges
        become bidirected edges at lag 0.
        """
        from causalrl.scm.graph import CausalGraph

        nodes = [lag_name(v, lag) for lag in range(self.max_lag, -1, -1) for v in self.variables]
        directed: list[tuple[str, str]] = []
        for ln in self.links:
            for target_lag in range(0, self.max_lag - ln.lag + 1):
                directed.append(
                    (lag_name(ln.source, target_lag + ln.lag), lag_name(ln.target, target_lag))
                )
        for a, b in self.contemporaneous_directed():
            directed.append((lag_name(a, 0), lag_name(b, 0)))
        bidirected = [(lag_name(a, 0), lag_name(b, 0)) for a, b in self.latent_pairs()]
        return CausalGraph(directed, bidirected, nodes=nodes)

    def summary_graph(self) -> object:
        """The lag-collapsed summary graph as a ``CyclicCausalGraph`` (recurrence is preserved).

        Lazy import: :mod:`causalrl.experimental` is outside the stable API.
        """
        from causalrl.experimental.cyclic import CyclicCausalGraph

        directed = self.lagged_edges() + self.contemporaneous_directed()
        return CyclicCausalGraph(directed, self.latent_pairs(), nodes=list(self.variables))

    def render(self) -> str:
        lines = [ln.render() for ln in sorted(self.links, key=lambda x: (x.target, x.lag))]
        contemp = self.contemporaneous.render()
        if contemp:
            lines.append(f"contemporaneous: {contemp}")
        return "\n".join(lines) if lines else "(no links)"


def _statistic(result: object) -> tuple[float, float | None, bool]:
    """Normalise a CI-test return value to ``(|statistic|, p_value, independent)``."""
    if isinstance(result, CITestResult):
        return abs(result.statistic), result.p_value, result.independent
    return (0.0, None, True) if bool(result) else (float("inf"), None, False)


def _pc1_parents(
    frame: Mapping[str, FloatArray],
    target: str,
    candidates: Sequence[str],
    *,
    ci_test: CITest,
    max_conditioning_size: int,
) -> list[str]:
    """PCMCI stage 1: prune ``candidates`` by conditioning on the strongest parents so far."""
    surviving = list(candidates)
    strength: dict[str, float] = dict.fromkeys(surviving, float("inf"))
    for size in range(max_conditioning_size + 1):
        if len(surviving) <= size:
            break
        ordered = sorted(surviving, key=lambda c: -strength[c])
        dropped: list[str] = []
        for cand in list(surviving):
            others = [c for c in ordered if c != cand][:size]
            if len(others) < size:
                continue
            stat, _, independent = _statistic(ci_test(frame, cand, target, others))
            # Runge et al. rank by the *minimum* statistic seen so far: a parent is only as strong
            # as its weakest evidence, so a link explained away by one conditioning set sinks.
            strength[cand] = min(strength[cand], stat)
            if independent:
                dropped.append(cand)
        for cand in dropped:
            surviving.remove(cand)
        if not dropped and size > 0:
            break
    return sorted(surviving, key=lambda c: -strength[c])


def discover_lagged(
    data: Mapping[str, FloatArray],
    variables: Sequence[str],
    *,
    max_lag: int = 3,
    ci_test: CITest | None = None,
    contemporaneous_ci_test: CITest | None = None,
    contemporaneous: bool = True,
    max_conditioning_size: int = 3,
    max_contemporaneous_conditioning_size: int = 2,
) -> LaggedGraph:
    """Discover a lagged causal graph from multivariate time series (PCMCI + FCI at lag 0).

    ``data`` maps variable name to a 1-D series; all series must share a length and a time base.
    ``ci_test`` is the independence oracle for the lagged phase — use
    :class:`~causalrl.neuro.citests.PoissonGLMTest` for spike counts and
    :class:`~causalrl.neuro.citests.PartialCorrelationTest` for continuous mesoscopic signals
    (the default). ``contemporaneous_ci_test`` defaults to ``ci_test``.

    Set ``contemporaneous=False`` to skip the lag-0 phase entirely when the bin width is known to
    resolve every interaction of interest; the returned PAG is then empty.
    """
    names = list(variables)
    if len(set(names)) != len(names):
        raise CausalGraphError("variables must be unique")
    if max_lag < 1:
        raise CausalGraphError("max_lag must be at least 1 for lagged discovery")
    lag_test = ci_test if ci_test is not None else PartialCorrelationTest()
    frame = lagged_frame(data, names, max_lag)

    # Stage 1 + 2: lagged parents per target.
    candidates = {
        target: [lag_name(v, lag) for v in names for lag in range(1, max_lag + 1)]
        for target in names
    }
    pc1: dict[str, list[str]] = {
        target: _pc1_parents(
            frame,
            target,
            candidates[target],
            ci_test=lag_test,
            max_conditioning_size=max_conditioning_size,
        )
        for target in names
    }

    links: list[LaggedLink] = []
    for target in names:
        parents = pc1[target]
        for parent in parents:
            source, lag = _split_lag(parent, names)
            conditioning = [p for p in parents if p != parent][:max_conditioning_size]
            # MCI: also condition on the source's own parents, shifted into the target's frame.
            for sp in pc1[source][:max_conditioning_size]:
                sp_var, sp_lag = _split_lag(sp, names)
                shifted = sp_lag + lag
                if shifted > max_lag:
                    continue  # not represented in this embedding; widen max_lag to include it
                name = lag_name(sp_var, shifted)
                if name not in conditioning and name != parent and name != target:
                    conditioning.append(name)
            stat, p, independent = _statistic(lag_test(frame, parent, target, conditioning))
            if not independent:
                links.append(LaggedLink(source, target, lag, stat, p))

    # Stage 3: contemporaneous slice, with the lagged parents pinned into every conditioning set.
    if contemporaneous and len(names) > 1:
        base = contemporaneous_ci_test if contemporaneous_ci_test is not None else lag_test
        always = {
            target: [lag_name(ln.source, ln.lag) for ln in links if ln.target == target]
            for target in names
        }
        pag = discover_latent(
            frame,
            names,
            ci_test=ConditionedCITest(base, always),
            max_conditioning_size=max_contemporaneous_conditioning_size,
        )
    else:
        pag = PAG(tuple(names), {})

    method = f"pcmci(max_lag={max_lag}, ci={type(lag_test).__name__})"
    return LaggedGraph(tuple(names), max_lag, tuple(links), pag, method)


def _split_lag(name: str, variables: Sequence[str]) -> tuple[str, int]:
    """Inverse of :func:`lag_name`."""
    if "@t-" not in name:
        if name not in variables:
            raise CausalGraphError(f"unknown variable in lag embedding: {name!r}")
        return name, 0
    var, _, lag = name.partition("@t-")
    if var not in variables:
        raise CausalGraphError(f"unknown variable in lag embedding: {name!r}")
    return var, int(lag)
