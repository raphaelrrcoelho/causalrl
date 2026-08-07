"""Fit what a confounded graph identifies, and *bound* what it does not.

:func:`causalrl.fit_scm` refuses a graph with bidirected edges, which is correct as far as it goes
-- under latent confounding a node's mechanism is genuinely not recoverable by regression on its
observed parents, and returning a point estimate would be a lie. But refusing outright is the one
move this library tells everyone else not to make. Partial identification is the whole thesis:
when the data do not pin a quantity down, report the set they *do* pin down and say so. Applying
that to the fitting layer is what this module does.

The split is per node, and it is finer than "the graph is confounded". A mechanism ``f_V`` is
recoverable by regression on ``Pa(V)`` exactly when the exogenous noise at ``V`` is independent of
those parents, so what breaks identification is a bidirected edge between ``V`` and one of *its own
parents* -- not any bidirected edge anywhere. A graph with ``A <-> B`` where ``B`` is not a parent
of ``A`` still has every mechanism identified, and this module fits all of them point-wise. Only
the nodes with a confounded parent fall back to an interval.

For those, the interval is the cross-fitted Manski bound already in
:class:`causalrl.FunctionalManskiBounds`: the confounded parents are the "treatment" (their joint
configuration indexes the arms), the remaining parents are the covariates, and the node's own value
is the outcome. That reuse is deliberate -- the bound a confounded mechanism admits is the same
bound a confounded action admits, and having one implementation means the overlap diagnostic and
the reward-range discipline come along for free.

What this does NOT return is a :class:`~causalrl.StructuralCausalModel`. A confounded fit has no
single mechanism at the bounded nodes, so there is no honest SCM to hand back, and returning one
whose type promises point answers would reintroduce exactly the problem. :class:`BoundedSCMFit`
answers interval queries instead, and says which nodes were identified.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import product

import numpy as np

from causalrl.bounds.functional import FunctionalManskiBounds, OverlapDiagnostic
from causalrl.certify.certificate import (
    Assumption,
    Certificate,
    EstimandSpec,
    Kind,
    Provenance,
)
from causalrl.exceptions import CausalGraphError
from causalrl.identification.bounds import Interval
from causalrl.scm.fit import fit_scm
from causalrl.scm.fitters import MechanismFitter
from causalrl.scm.graph import CausalGraph
from causalrl.scm.scm import StructuralCausalModel

__all__ = ["BoundedNodeFit", "BoundedSCMFit", "fit_scm_bounded"]

_MAX_ARMS = 64
"""Largest number of confounded-parent configurations treated as arms at one node.

The arms are the PRODUCT of the confounded parents' level counts, so this grows explosively and
each arm needs enough rows to fit an outcome model and a propensity. Past this the bound would be
vacuous at almost every arm anyway, which is a slower and less informative way of saying the data
cannot answer the question.
"""


@dataclass(frozen=True)
class BoundedNodeFit:
    """What one node's mechanism turned out to be: a point fit, or an interval.

    ``identified`` is the field to branch on. When ``True`` the node was fitted exactly as
    :func:`causalrl.fit_scm` would have, and ``confounded_parents`` is empty. When ``False`` the
    node has at least one parent it shares a latent cause with, and every query about it comes back
    as an :class:`~causalrl.Interval` whose width is the price of that confounding.
    """

    node: str
    parents: tuple[str, ...]
    confounded_parents: tuple[str, ...]
    identified: bool
    value_range: tuple[float, float] | None = None
    overlap: OverlapDiagnostic | None = None


class BoundedSCMFit:
    """A fit over a confounded graph: point mechanisms where identified, intervals elsewhere.

    Query it with :meth:`interval`. An identified node returns a degenerate interval (the point
    estimate, twice over) so that a caller can treat every node uniformly without first asking
    which kind it is -- and a caller that *does* care can ask :meth:`is_identified`.
    """

    def __init__(
        self,
        nodes: Sequence[BoundedNodeFit],
        *,
        n_samples: int,
        point_model: StructuralCausalModel,
        bounds: Mapping[str, FunctionalManskiBounds],
        arm_index: Mapping[str, Mapping[tuple[float, ...], int]],
        seed: int,
    ) -> None:
        self._nodes = tuple(nodes)
        self.n_samples = int(n_samples)
        self._point_model = point_model
        self._bounds = dict(bounds)
        self._arm_index = {k: dict(v) for k, v in arm_index.items()}
        self._seed = int(seed)

    @property
    def nodes(self) -> tuple[BoundedNodeFit, ...]:
        """Per-node results, in the graph's topological order."""
        return self._nodes

    @property
    def bounded_nodes(self) -> tuple[str, ...]:
        """Nodes whose mechanism is not identified, and so is reported as an interval."""
        return tuple(f.node for f in self._nodes if not f.identified)

    @property
    def identified_nodes(self) -> tuple[str, ...]:
        """Nodes whose mechanism the data identify point-wise."""
        return tuple(f.node for f in self._nodes if f.identified)

    def is_identified(self, node: str) -> bool:
        """Whether ``node``'s mechanism was point-identified."""
        return self._fit_for(node).identified

    @property
    def point_model(self) -> StructuralCausalModel:
        """The fitted SCM, whose mechanisms are trustworthy at :attr:`identified_nodes` ONLY.

        Exposed because an identified node deserves the whole library -- ``do``/``see``,
        transport, counterfactuals -- rather than the single scalar :meth:`interval` returns. Its
        mechanisms at :attr:`bounded_nodes` are the regression the confounding invalidates, and are
        not to be used; they exist because one fitting path is fitted for every node at once.
        """
        return self._point_model

    def interval(
        self, node: str, assignment: Mapping[str, float], *, n_samples: int = 4096
    ) -> Interval:
        """Bound ``E[node | do(parents = assignment)]``.

        ``assignment`` must give a value for every parent of ``node``. An identified node returns
        a degenerate interval, estimated from :attr:`point_model` with ``n_samples`` draws; a
        confounded one returns the cross-fitted Manski bound, which is as wide as the logs'
        overlap at that arm requires and is vacuous where they carry no information at all.
        """
        fit = self._fit_for(node)
        missing = [p for p in fit.parents if p not in assignment]
        if missing:
            raise KeyError(
                f"assignment is missing parent(s) {sorted(missing)} of {node!r}: bounding "
                f"E[{node} | do(...)] needs a value for every parent ({list(fit.parents)})."
            )
        if fit.identified:
            value = self._point_value(node, assignment, n_samples)
            return Interval(value, value)

        arm_key = tuple(float(assignment[p]) for p in fit.confounded_parents)
        arms = self._arm_index[node]
        if arm_key not in arms:
            low, high = fit.value_range or (0.0, 1.0)
            return Interval(low, high)
        covariates = [p for p in fit.parents if p not in fit.confounded_parents]
        features = np.array(
            [[float(assignment[p]) for p in covariates]] if covariates else [[0.0]],
            dtype=np.float64,
        )
        low, high = self._bounds[node].bounds(features)
        arm = arms[arm_key]
        return Interval(float(low[0, arm]), float(high[0, arm]))

    def _point_value(self, node: str, assignment: Mapping[str, float], n: int) -> float:
        """``E[node | do(parents)]`` from the fitted SCM, by intervening and averaging.

        Estimated by sampling rather than read off the mechanism, so that every family works the
        same way -- an additive-noise mechanism, a tabular CPT and a pinned equation all answer
        this question, and none of them answers it through the same attribute. Deterministic given
        the fit's seed.
        """
        parents = {p: float(assignment[p]) for p in self._fit_for(node).parents}
        drawn = self._point_model.do(parents).see(n, seed=self._seed)
        return float(np.asarray(drawn[node], dtype=np.float64).mean())

    def _fit_for(self, node: str) -> BoundedNodeFit:
        for fit in self._nodes:
            if fit.node == node:
                return fit
        raise CausalGraphError(f"unknown node: {node!r}")

    def certificate(self) -> Certificate:
        """A ``BOUNDED`` certificate naming which nodes cost the model its point identification."""
        bounded = self.bounded_nodes
        return Certificate(
            claim=(
                f"fitted SCM over {len(self._nodes)} nodes; {len(self.identified_nodes)} "
                f"identified point-wise, {len(bounded)} bounded under latent confounding"
                + (f" ({', '.join(bounded)})" if bounded else "")
            ),
            estimand=EstimandSpec(query="do", target="mean"),
            kind=Kind.BOUNDED if bounded else Kind.IDENTIFIED,
            value=None,
            alpha=None,
            assumptions=(
                Assumption(
                    name="bounded-outcome-range",
                    params={"nodes": list(bounded)},
                    checkable=True,
                ),
                Assumption(name="correctly-specified-nuisances", params={}, checkable=False),
            ),
            method="manski-bounds-on-confounded-mechanisms",
            witness=None,
            hedge=None,
            provenance=Provenance.create(),
            ci=None,
        )

    def summary(self) -> str:
        lines = [f"BoundedSCMFit(n={self.n_samples})"]
        for fit in self._nodes:
            if fit.identified:
                lines.append(f"  {fit.node}: IDENTIFIED parents=[{', '.join(fit.parents)}]")
            else:
                overlap = f" {fit.overlap.summary()}" if fit.overlap is not None else ""
                lines.append(
                    f"  {fit.node}: BOUNDED confounded_with="
                    f"[{', '.join(fit.confounded_parents)}]{overlap}"
                )
        return "\n".join(lines)


def _levels(column: np.ndarray) -> list[float]:
    return sorted({float(v) for v in np.asarray(column).ravel()})


def fit_scm_bounded(
    data: Mapping[str, np.ndarray],
    *,
    graph: CausalGraph,
    value_ranges: Mapping[str, tuple[float, float]] | None = None,
    families: Mapping[str, MechanismFitter] | None = None,
    holdout: float = 0.2,
    seed: int = 0,
) -> BoundedSCMFit:
    """Fit ``graph``'s identified mechanisms and bound the confounded ones.

    The confounded-graph counterpart of :func:`causalrl.fit_scm`, which refuses this input. A node
    is bounded exactly when it shares a bidirected edge with one of its own parents; every other
    node is fitted point-wise by the same machinery, with the same ``families`` overrides and the
    same held-out scoring.

    ``value_ranges`` gives ``(low, high)`` for each bounded node and is REQUIRED for them: a Manski
    bound is built out of the range the unobserved fraction could occupy, so a missing range is a
    missing bound rather than a default. Ranges for identified nodes are ignored.

    Bounded nodes' confounded parents must be discrete with few levels -- their joint configuration
    indexes the arms of the bound.
    """
    ranges = dict(value_ranges or {})
    confounded: dict[str, tuple[str, ...]] = {}
    for node in graph.topological_order():
        parents = tuple(sorted(graph.parents(node)))
        confounded[node] = tuple(p for p in parents if graph.is_confounded(node, p))

    bounded_nodes = [n for n, ps in confounded.items() if ps]
    missing_ranges = sorted(n for n in bounded_nodes if n not in ranges)
    if missing_ranges:
        raise ValueError(
            f"value_ranges is missing bounded node(s) {missing_ranges}: each is confounded with "
            f"one of its own parents (e.g. {missing_ranges[0]} <-> "
            f"{confounded[missing_ranges[0]][0]}), so its mechanism is not identified and the "
            "only thing the data support is a Manski interval -- which is built from the range "
            "the node's unobserved values could occupy. Without that range there is no bound."
        )

    # Every node whose own parents are unconfounded is fitted exactly as fit_scm would: strip the
    # bidirected edges it refuses on, and fit that graph. Bounded nodes are refitted below, so
    # their point mechanism is computed and discarded -- cheap, and it keeps one fitting path.
    point_graph = CausalGraph(directed_edges=graph.directed_edges, nodes=graph.nodes)
    point_model = fit_scm(data, graph=point_graph, families=families, holdout=holdout, seed=seed)

    bounds: dict[str, FunctionalManskiBounds] = {}
    arm_index: dict[str, dict[tuple[float, ...], int]] = {}
    fits: list[BoundedNodeFit] = []
    n_samples = len(next(iter(data.values())))

    for node in graph.topological_order():
        parents = tuple(sorted(graph.parents(node)))
        confounders = confounded[node]
        if not confounders:
            fits.append(
                BoundedNodeFit(node=node, parents=parents, confounded_parents=(), identified=True)
            )
            continue

        level_sets = [_levels(data[p]) for p in confounders]
        n_arms = int(np.prod([len(levels) for levels in level_sets]))
        if n_arms > _MAX_ARMS:
            raise ValueError(
                f"node {node!r} has {n_arms} confounded-parent configurations "
                f"({', '.join(confounders)}), above the limit of {_MAX_ARMS}. Each is an arm "
                "needing its own outcome and propensity fit, so the bound would be vacuous at "
                "almost all of them. Coarsen the confounded parents first."
            )
        index = {combo: i for i, combo in enumerate(product(*level_sets))}
        covariates = [p for p in parents if p not in confounders]
        features = (
            np.column_stack([np.asarray(data[p], dtype=np.float64) for p in covariates])
            if covariates
            else np.zeros((n_samples, 1))
        )
        actions = np.array(
            [index[tuple(float(data[p][i]) for p in confounders)] for i in range(n_samples)],
            dtype=np.int_,
        )
        model = FunctionalManskiBounds(
            n_actions=len(index), reward_range=ranges[node], seed=seed
        ).fit(features, actions, np.asarray(data[node], dtype=np.float64))
        bounds[node] = model
        arm_index[node] = index
        fits.append(
            BoundedNodeFit(
                node=node,
                parents=parents,
                confounded_parents=confounders,
                identified=False,
                value_range=ranges[node],
                overlap=model.diagnostic(),
            )
        )

    return BoundedSCMFit(
        fits,
        n_samples=n_samples,
        point_model=point_model,
        bounds=bounds,
        arm_index=arm_index,
        seed=seed,
    )
