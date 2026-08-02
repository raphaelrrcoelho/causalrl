"""Learn a StructuralCausalModel from data given a DAG.

Fits into the *existing* SCM type, so ``do`` / ``see`` / ``CausalEnvWrapper`` / transport /
certify all accept a learned model unchanged. The returned model carries ``provenance="fitted"``,
which gates L3 queries: L1 data identifies the mechanisms but not the noise-to-value coupling.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import NamedTuple

import numpy as np
from torch.distributions import Distribution

from causalrl.exceptions import NotIdentifiableError
from causalrl.scm.fitters import ANMFit, MechanismFitter, TabularCPT, evaluate_holdout
from causalrl.scm.graph import CausalGraph
from causalrl.scm.mechanisms import Mechanism
from causalrl.scm.scm import StructuralCausalModel

_MAX_DISCRETE_LEVELS = 20


class NodeFit(NamedTuple):
    """What was fitted at one node, and what that licenses.

    ``holdout_score`` evaluates the DEPLOYED (train-fitted) mechanism against data the fit never
    saw -- mean log-likelihood for a discrete (non-invertible) mechanism, R^2 for a continuous
    (invertible, additive-noise) one -- via :func:`causalrl.scm.fitters.evaluate_holdout`. It sits
    on the same scale as each family's in-sample ``score``, but is a genuine out-of-sample number:
    an overfit deployed mechanism scores worse here even though its in-sample ``score`` looks fine.
    """

    node: str
    family: str
    parents: tuple[str, ...]
    holdout_score: float
    invertible: bool


class FitReport(NamedTuple):
    """Per-node provenance for a fitted SCM -- see :class:`NodeFit` for what each field reports."""

    nodes: tuple[NodeFit, ...]
    n_samples: int

    def summary(self) -> str:
        lines = [f"FitReport(n={self.n_samples})"]
        for fit in self.nodes:
            parents = ", ".join(fit.parents) or "-"
            lines.append(
                f"  {fit.node}: family={fit.family} parents=[{parents}] "
                f"holdout={fit.holdout_score:.3f} invertible={fit.invertible}"
            )
        return "\n".join(lines)


_FAMILY_NAMES = {
    "TabularCPT": "tabular_cpt",
    "LinearGaussianFit": "linear_gaussian",
    "ANMFit": "anm",
    "NeuralFit": "neural",
}


def _family_name(fitter: MechanismFitter) -> str:
    return _FAMILY_NAMES.get(type(fitter).__name__, type(fitter).__name__)


def _is_discrete(column: np.ndarray) -> bool:
    """Integer-valued with few distinct levels — the tabular regime."""
    values = np.asarray(column)
    if not np.all(np.isfinite(values)):
        return False
    integral = np.allclose(values, np.round(values))
    return bool(integral and len(np.unique(values)) <= _MAX_DISCRETE_LEVELS)


def fit_scm(
    data: Mapping[str, np.ndarray],
    *,
    graph: CausalGraph,
    families: Mapping[str, MechanismFitter] | None = None,
    holdout: float = 0.2,
    seed: int = 0,
) -> StructuralCausalModel:
    """Fit every mechanism of ``graph`` from ``data``; return a learned SCM.

    Each node is fitted by ``families[node]`` when given, else by dtype: integer-valued columns
    with at most 20 levels get :class:`TabularCPT`, everything else :class:`ANMFit`. The result is
    an ordinary :class:`StructuralCausalModel` marked ``provenance="fitted"``.

    Raises :class:`NotIdentifiableError` on a graph with bidirected edges: under latent confounding
    a node's mechanism is not recoverable by regression on its observed parents.
    """
    if graph.has_bidirected_edges():
        raise NotIdentifiableError(
            "fit_scm cannot fit a graph with bidirected edges: under latent confounding a "
            "mechanism is not identified by regression on observed parents. Fitting confounded "
            "graphs needs the neural-causal-model construction (one latent per c-component).",
            witness=graph.bidirected_edges,
        )
    missing = [node for node in graph.nodes if node not in data]
    if missing:
        raise KeyError(f"data is missing column(s) for graph node(s): {sorted(missing)}")

    columns = {node: np.asarray(data[node]) for node in graph.nodes}
    n = len(next(iter(columns.values())))
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(n)
    cut = max(1, int(n * (1.0 - holdout)))
    train_index, test_index = permutation[:cut], permutation[cut:]

    mechanisms: dict[str, Mechanism] = {}
    exogenous: dict[str, Distribution] = {}
    fits: list[NodeFit] = []
    for node in graph.topological_order():
        parents = tuple(sorted(graph.parents(node)))
        fitter = (families or {}).get(node) or (
            TabularCPT() if _is_discrete(columns[node]) else ANMFit()
        )
        train_parents = {p: columns[p][train_index] for p in parents}
        fitted = fitter.fit(train_parents, columns[node][train_index])
        test_parents = {p: columns[p][test_index] for p in parents}
        holdout_score = (
            evaluate_holdout(fitted, test_parents, columns[node][test_index])
            if len(test_index) > 0
            else fitted.score
        )
        mechanisms[node] = fitted.mechanism
        exogenous[node] = fitted.noise
        fits.append(
            NodeFit(
                node=node,
                family=_family_name(fitter),
                parents=parents,
                holdout_score=holdout_score,
                invertible=fitted.invertible,
            )
        )
        fitted.mechanism.invertible = fitted.invertible  # type: ignore[attr-defined]

    return StructuralCausalModel(
        graph,
        mechanisms,
        exogenous,
        provenance="fitted",
        fit_report=FitReport(nodes=tuple(fits), n_samples=n),
    )
