"""Find WHICH mechanisms differ between regimes, instead of being told.

The library carries the whole vocabulary for "these two domains differ in these mechanisms" --
:class:`~causalrl.Regime`, selection diagrams, :func:`~causalrl.identify_transport`,
:func:`~causalrl.transported_effect` -- and every one of them takes the selection set as an
*input*. You have to already know which mechanisms shifted. That is the one end of the transport
loop that was open: nothing here helped you find it, even though the data usually can.

The test is a conditional independence, and it is exact rather than a heuristic. Write ``R`` for
the regime label. The mechanism at node ``V`` is invariant across regimes precisely when

    P(V | Pa(V)) is the same in every regime,  i.e.  V independent of R given Pa(V).

So localising a shift is running the library's own CI test with the regime indicator as one
argument, once per node. Nodes that reject are exactly the ones a selection diagram should mark,
and :attr:`ShiftReport.selection` is in the form :func:`~causalrl.identify_transport` consumes --
which is what closes the loop.

The scope this honestly has: it finds shifts in ``P(V | Pa(V))`` for the graph you supply. It does
not discover the graph, and it inherits that graph's correctness -- a missing edge can show up here
as a "shift" at a node whose parent set is simply wrong. It also tests each node separately, so
with many nodes the usual multiple-comparison caution applies; :func:`localize_mechanism_shift`
reports every node's p-value rather than only the rejections, so a caller can apply their own
correction.

Related to Invariant Causal Prediction (Peters, Buhlmann & Meinshausen, JRSS-B 2016), which runs
the same invariance test across candidate *parent sets* to discover causal structure. Here the
graph is given and the question is inverted: which mechanisms moved. No code is ported.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from causalrl.certify.certificate import (
    Assumption,
    Certificate,
    EstimandSpec,
    Kind,
    Provenance,
    Witness,
)
from causalrl.discovery import conditional_mutual_information
from causalrl.scm.graph import CausalGraph

__all__ = ["MechanismShift", "ShiftReport", "localize_mechanism_shift"]

_REGIME_COLUMN = "__regime__"


def _normal_sf(z: float) -> float:
    """``P(Z > z)`` for a standard normal, via :func:`math.erfc`."""
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def _chi2_sf(x: float, df: int) -> float:
    """Upper tail ``P(chi2_df > x)`` by the Wilson-Hilferty cube-root normal approximation.

    ``(X / k) ** (1/3)`` is close to normal with mean ``1 - 2/(9k)`` and variance ``2/(9k)``. The
    approximation is accurate to about three decimals over the range that matters for a screening
    test, and it keeps this module inside the core dependency set (numpy and networkx) rather than
    pulling in scipy for one tail probability.
    """
    if df <= 0:
        return 1.0
    if x <= 0.0:
        return 1.0
    t = (x / df) ** (1.0 / 3.0)
    mean = 1.0 - 2.0 / (9.0 * df)
    sd = math.sqrt(2.0 / (9.0 * df))
    return min(1.0, max(0.0, _normal_sf((t - mean) / sd)))


@dataclass(frozen=True)
class MechanismShift:
    """The invariance verdict at one node.

    ``statistic`` is the G-test statistic ``2 n I(V; R | Pa(V))`` in nats-times-samples, which is
    asymptotically chi-squared under invariance; ``p_value`` is its upper tail. ``shifted`` is the
    verdict at the caller's ``alpha``, and is what puts the node in the selection set.
    """

    node: str
    parents: tuple[str, ...]
    cmi: float
    statistic: float
    df: int
    p_value: float
    shifted: bool


class ShiftReport:
    """Which mechanisms moved between regimes, and the selection set that follows."""

    def __init__(self, shifts: Sequence[MechanismShift], *, regimes: Sequence[str], alpha: float):
        self._shifts = tuple(shifts)
        self.regimes = tuple(regimes)
        self.alpha = float(alpha)

    @property
    def shifts(self) -> tuple[MechanismShift, ...]:
        """Every node's verdict, in the graph's topological order -- not only the rejections."""
        return self._shifts

    @property
    def selection(self) -> frozenset[str]:
        """The shifted nodes, ready to hand to :func:`~causalrl.identify_transport`."""
        return frozenset(s.node for s in self._shifts if s.shifted)

    @property
    def invariant(self) -> frozenset[str]:
        """Nodes whose mechanism the data give no reason to think moved."""
        return frozenset(s.node for s in self._shifts if not s.shifted)

    def certificate(self) -> Certificate:
        """An ``EMPIRICAL`` certificate: this is a hypothesis test, not an identification result.

        Failing to reject invariance is not proof of it, and the certificate says so rather than
        reporting the invariant set as established. Anything built on top -- a transport formula
        keyed on :attr:`selection` -- inherits that.
        """
        shifted = sorted(self.selection)
        return Certificate(
            claim=(
                f"{len(shifted)} of {len(self._shifts)} mechanisms differ across regimes "
                f"{list(self.regimes)} at alpha={self.alpha:g}"
                + (f": {', '.join(shifted)}" if shifted else "")
            ),
            estimand=EstimandSpec(query="see", target="mean"),
            kind=Kind.EMPIRICAL,
            value=None,
            alpha=self.alpha,
            assumptions=(
                Assumption(
                    name="supplied-graph-is-correct",
                    params={"n_nodes": len(self._shifts)},
                    checkable=False,
                ),
                Assumption(
                    name="no-multiplicity-correction",
                    params={"n_tests": len(self._shifts)},
                    checkable=True,
                    diagnostic={s.node: s.p_value for s in self._shifts},
                ),
            ),
            method="conditional-invariance-test",
            witness=Witness("selection-set", {"selection": shifted}),
            hedge=None,
            provenance=Provenance.create(),
        )

    def summary(self) -> str:
        lines = [f"ShiftReport(regimes={list(self.regimes)}, alpha={self.alpha:g})"]
        for s in self._shifts:
            verdict = "SHIFTED " if s.shifted else "invariant"
            lines.append(
                f"  {verdict} {s.node}: p={s.p_value:.4f} (G={s.statistic:.2f}, df={s.df})"
            )
        return "\n".join(lines)


def _levels(column: np.ndarray) -> int:
    return len({int(v) for v in np.asarray(column).ravel()})


def localize_mechanism_shift(
    data_by_regime: Mapping[str, Mapping[str, np.ndarray]],
    *,
    graph: CausalGraph,
    alpha: float = 0.05,
) -> ShiftReport:
    """Test each node's mechanism for invariance across regimes; emit the selection set.

    ``data_by_regime`` maps a regime name to its columns -- at least two regimes, each carrying
    every node of ``graph``. Columns must be integer-coded (the CI test is the discrete one); bin
    continuous variables first, and be aware that the test then asks about the binned mechanism.

    Returns a :class:`ShiftReport` whose :attr:`~ShiftReport.selection` plugs straight into
    :func:`~causalrl.identify_transport`.
    """
    if len(data_by_regime) < 2:
        raise ValueError(
            f"localize_mechanism_shift needs at least 2 regimes, got {len(data_by_regime)}: "
            "invariance is a statement about a comparison, and there is nothing to compare a "
            "single regime with."
        )
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha={alpha} must lie in (0, 1)")

    regimes = list(data_by_regime)
    nodes = graph.topological_order()
    for name, columns in data_by_regime.items():
        missing = [node for node in nodes if node not in columns]
        if missing:
            raise KeyError(f"regime {name!r} is missing column(s) for graph node(s): {missing}")

    pooled: dict[str, np.ndarray] = {}
    for node in nodes:
        pooled[node] = np.concatenate(
            [np.asarray(data_by_regime[r][node]).ravel() for r in regimes]
        )
    pooled[_REGIME_COLUMN] = np.concatenate(
        [
            np.full(len(np.asarray(data_by_regime[r][nodes[0]]).ravel()), i)
            for i, r in enumerate(regimes)
        ]
    )
    n = len(pooled[_REGIME_COLUMN])

    shifts: list[MechanismShift] = []
    for node in nodes:
        parents = tuple(sorted(graph.parents(node)))
        cmi = conditional_mutual_information(pooled, node, _REGIME_COLUMN, list(parents))
        statistic = 2.0 * n * cmi
        # Degrees of freedom for the conditional G-test: (levels-1)(regimes-1) per observed
        # parent configuration. Counting only OBSERVED configurations matters -- an unobserved
        # cell contributes nothing to the statistic, so charging df for it would only make the
        # test conservative in proportion to how sparse the parent space is.
        if parents:
            observed_configs = len({tuple(int(pooled[p][i]) for p in parents) for i in range(n)})
        else:
            observed_configs = 1
        df = max(1, (_levels(pooled[node]) - 1) * (len(regimes) - 1) * observed_configs)
        p_value = _chi2_sf(statistic, df)
        shifts.append(
            MechanismShift(
                node=node,
                parents=parents,
                cmi=cmi,
                statistic=statistic,
                df=df,
                p_value=p_value,
                shifted=p_value < alpha,
            )
        )
    return ShiftReport(shifts, regimes=regimes, alpha=alpha)
