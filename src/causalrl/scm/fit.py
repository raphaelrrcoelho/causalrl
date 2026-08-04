"""Learn a StructuralCausalModel from data given a DAG.

Fits into the *existing* SCM type, so ``do`` / ``see`` / ``CausalEnvWrapper`` / transport /
certify all accept a learned model unchanged. The returned model carries ``provenance="fitted"``,
which gates L3 queries: L1 data identifies the mechanisms but not the noise-to-value coupling.
"""

from __future__ import annotations

from collections.abc import Mapping
from itertools import combinations, product
from typing import Any, NamedTuple

import numpy as np
from torch.distributions import Distribution

from causalrl.discovery import CPDAG
from causalrl.exceptions import CausalGraphError, NotIdentifiableError
from causalrl.scm.fitters import ANMFit, MechanismFitter, TabularCPT, evaluate_holdout
from causalrl.scm.graph import CausalGraph
from causalrl.scm.mechanisms import Mechanism
from causalrl.scm.scm import StructuralCausalModel

_MAX_DISCRETE_LEVELS = 20

_MAX_MEC_ENUMERATION = 4096
"""Candidate orientations :func:`_enumerate_mec` will materialise before refusing.

An equivalence class is enumerated by trying all ``2 ** k`` orientations of its ``k`` undirected
edges, so the WORK is exponential in ``k`` whatever the resulting class size turns out to be.
``max_members`` caps the class, not the search, so it cannot bound this. 4096 candidates costs
about 0.15 s here; 2 ** 25 -- an ordinary PC output on 10 variables -- would not finish.
"""


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
    "PoissonGLMFit": "poisson_glm",
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

    **The deployed mechanisms are fitted on a fraction of the data.** ``data`` is split once, by a
    ``seed``-controlled permutation shared by every node: ``1 - holdout`` of the rows train the
    mechanisms the returned SCM actually carries, and the remaining ``holdout`` fraction is scored
    but never trained on. Nothing is refitted on the full sample afterwards, so at the default
    ``holdout=0.2`` every number this model reports — every ``do()``, every contrast — comes from
    an 80%-data fit. Pass a smaller ``holdout`` to trade the out-of-sample score for training rows.

    ``holdout`` must lie in ``(0, 1)``. ``0.0`` is refused rather than silently reporting an
    in-sample number under :attr:`NodeFit.holdout_score`, whose contract is data the fit never
    saw; ``>= 1.0`` would leave a one-row training set.

    Raises :class:`NotIdentifiableError` on a graph with bidirected edges: under latent confounding
    a node's mechanism is not recoverable by regression on its observed parents.
    """
    if not 0.0 < holdout < 1.0:
        raise ValueError(
            f"holdout={holdout} must lie in (0, 1): it is the fraction of rows held out of every "
            "mechanism's fit and scored as NodeFit.holdout_score. At 0.0 there is no such data, "
            "so the reported score would be an in-sample number under an out-of-sample name; at "
            "1.0 or above there is nothing left to fit on."
        )
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

    overrides = families or {}
    mechanisms: dict[str, Mechanism] = {}
    exogenous: dict[str, Distribution] = {}
    fits: list[NodeFit] = []
    for node in graph.topological_order():
        parents = tuple(sorted(graph.parents(node)))
        fitter = overrides.get(node) or (TabularCPT() if _is_discrete(columns[node]) else ANMFit())
        train_parents = {p: columns[p][train_index] for p in parents}
        fitted = fitter.fit(train_parents, columns[node][train_index])
        test_parents = {p: columns[p][test_index] for p in parents}
        holdout_score = (
            evaluate_holdout(fitted, test_parents, columns[node][test_index])
            # A validated holdout in (0, 1) leaves the test partition empty only at n == 1, where
            # the in-sample score is the only number there is.
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


def fit_scm_mec(
    data: Mapping[str, np.ndarray],
    *,
    cpdag: CPDAG,
    max_members: int = 32,
    **kwargs: Any,
) -> list[StructuralCausalModel]:
    """Fit one SCM per DAG in ``cpdag``'s Markov equivalence class.

    The returned list is the SCM *belief*: observational data cannot choose among these, so every
    member fits it equally well. Sub-project 3 (active interventional discovery) shrinks this set
    by intervening. Raises ``ValueError`` above ``max_members`` naming the true class size —
    equivalence classes are exponential and silently truncating one would misreport the belief.

    A second ``ValueError`` fires *before* enumerating when the CPDAG has too many undirected
    edges: finding the class means trying all ``2 ** k`` orientations of them, and that work is
    exponential in ``k`` no matter how small the class turns out to be, so ``max_members`` cannot
    bound it (see :data:`_MAX_MEC_ENUMERATION`). The pre-check can only name ``2 ** k``, an upper
    bound on the class size rather than the size itself, and so over-refuses: the honest reading
    is "this search is too big to run", not "this class is too big to fit".
    """
    undirected_count = len(cpdag.undirected_edges)
    candidates = 1 << undirected_count
    if candidates > _MAX_MEC_ENUMERATION:
        raise ValueError(
            f"cpdag has {undirected_count} undirected edges, so enumerating its equivalence class "
            f"means orienting them 2**{undirected_count} = {candidates} ways -- above the "
            f"_MAX_MEC_ENUMERATION={_MAX_MEC_ENUMERATION} search budget. That count is an upper "
            f"bound on the class size, not the size itself, which is only known after the search; "
            f"max_members caps the class and so cannot cap this. Narrow the CPDAG with tiers via "
            f"orient() (or supply interventional data) before fitting."
        )
    members = _enumerate_mec(cpdag)
    if len(members) > max_members:
        raise ValueError(
            f"equivalence class has {len(members)} members, above max_members={max_members}; "
            "raise the cap deliberately or narrow the CPDAG with tiers via orient()"
        )
    return [fit_scm(data, graph=graph, **kwargs) for graph in members]


def _enumerate_mec(cpdag: CPDAG) -> list[CausalGraph]:
    """Every acyclic orientation of the undirected edges that introduces no new v-structure."""
    undirected = [tuple(sorted(edge)) for edge in sorted(cpdag.undirected_edges, key=sorted)]
    base = set(cpdag.directed_edges)
    adjacency = {frozenset(e) for e in (*base, *undirected)}
    baseline = _v_structures(base, adjacency)
    graphs: list[CausalGraph] = []
    for choice in product([False, True], repeat=len(undirected)):
        edges = set(base)
        for (a, b), reversed_ in zip(undirected, choice, strict=True):
            edges.add((b, a) if reversed_ else (a, b))
        try:
            graph = CausalGraph(directed_edges=sorted(edges), nodes=list(cpdag.variables))
        except CausalGraphError:
            continue  # cyclic orientation
        if _v_structures(edges, adjacency) != baseline:
            continue
        graphs.append(graph)
    return graphs


def _v_structures(
    edges: set[tuple[str, str]], adjacency: set[frozenset[str]]
) -> set[tuple[str, str, str]]:
    """Unshielded colliders ``a -> c <- b`` with ``a``, ``b`` non-adjacent."""
    found: set[tuple[str, str, str]] = set()
    parents: dict[str, set[str]] = {}
    for u, v in edges:
        parents.setdefault(v, set()).add(u)
    for child, ps in parents.items():
        for a, b in combinations(sorted(ps), 2):
            if frozenset((a, b)) not in adjacency:
                found.add((a, child, b))
    return found
