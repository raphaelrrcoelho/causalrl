"""Learn a StructuralCausalModel from data given a DAG.

Fits into the *existing* SCM type, so ``do`` / ``see`` / ``CausalEnvWrapper`` / transport /
certify all accept a learned model unchanged. The returned model carries ``provenance="fitted"``,
which gates L3 queries: L1 data identifies the mechanisms but not the noise-to-value coupling.

Fitting mechanisms from a fixed table is supervised learning, but the object it produces is a
model-based-RL *world model*: hand it to
:class:`~causalrl.envs.suite.scbandit.StructuralCausalBanditEnv` and an agent can act in it
(``tests/test_learned_scm_as_env.py``, ``examples/learned_scm_policy.py``). Planning *inside* a
fitted model with :class:`~causalrl.agents.causal_mbrl.CausalMBRLAgent` is not wired -- that agent
builds its value table from columnar data and takes no SCM (sub-project 4 of
``docs/learned_scm/DESIGN.md``).
"""

from __future__ import annotations

from collections.abc import Mapping
from itertools import combinations, product
from typing import Any, Literal, NamedTuple

import numpy as np
from torch.distributions import Distribution

from causalrl.discovery import CPDAG
from causalrl.exceptions import CausalGraphError, NotIdentifiableError
from causalrl.scm.fitters import (
    ANMFit,
    MechanismFitter,
    PinnedMechanism,
    TabularCPT,
    evaluate_holdout,
)
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

    ``pinned`` marks a node whose equation was SUPPLIED rather than learned (see
    :class:`causalrl.scm.fitters.PinnedMechanism`). It changes how ``holdout_score`` should be
    read: for a learned node the score measures how well a family fitted, and for a pinned one it
    measures whether the equation the caller asserted survives contact with held-out data.
    """

    node: str
    family: str
    parents: tuple[str, ...]
    holdout_score: float
    invertible: bool
    pinned: bool = False


class FitReport(NamedTuple):
    """Per-node provenance for a fitted SCM -- see :class:`NodeFit` for what each field reports."""

    nodes: tuple[NodeFit, ...]
    n_samples: int

    def summary(self) -> str:
        lines = [f"FitReport(n={self.n_samples})"]
        for fit in self.nodes:
            parents = ", ".join(fit.parents) or "-"
            marker = " PINNED" if fit.pinned else ""
            lines.append(
                f"  {fit.node}: family={fit.family} parents=[{parents}] "
                f"holdout={fit.holdout_score:.3f} invertible={fit.invertible}{marker}"
            )
        return "\n".join(lines)

    @property
    def pinned_nodes(self) -> tuple[str, ...]:
        """Nodes whose equation was supplied rather than learned, in report order."""
        return tuple(fit.node for fit in self.nodes if fit.pinned)


_FAMILY_NAMES = {
    "TabularCPT": "tabular_cpt",
    "LinearGaussianFit": "linear_gaussian",
    "ANMFit": "anm",
    "NeuralFit": "neural",
    "PoissonGLMFit": "poisson_glm",
    "BayesianLinearFit": "bayesian_linear",
    "PinnedMechanism": "pinned",
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

    **Mechanisms may be pinned instead of learned.** Passing
    :class:`~causalrl.scm.fitters.PinnedMechanism` as a node's family deploys the equation you
    supply at that node while the rest of the graph is still fitted from ``data`` — the ordinary
    case where a documented rule sits next to behaviour that has no closed form. A model with both
    kinds of node carries ``provenance="mixed"``; one whose every node is pinned carries
    ``provenance="specified"``, since nothing about its equations came from the data. Pinned nodes
    are listed by :attr:`FitReport.pinned_nodes` and still receive a ``holdout_score``, which for
    them tests the asserted equation rather than measuring a fit.

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
                pinned=isinstance(fitter, PinnedMechanism),
            )
        )
        fitted.mechanism.invertible = fitted.invertible  # type: ignore[attr-defined]

    return StructuralCausalModel(
        graph,
        mechanisms,
        exogenous,
        provenance=_provenance(fits),
        fit_report=FitReport(nodes=tuple(fits), n_samples=n),
    )


def _provenance(fits: list[NodeFit]) -> Literal["specified", "fitted", "mixed"]:
    """Where the returned model's equations came from: all supplied, all learned, or both.

    ``"specified"`` for an all-pinned model matches a hand-built SCM, whose mechanisms are equally
    a caller's assertion. ``"mixed"`` is gated like ``"fitted"`` for L3 queries -- a model is only
    as identified as its weakest node, and a mix still contains learned ones.
    """
    if not fits:
        # Degenerate, but the label still has to be true: nothing was learned from data, so
        # "fitted" would assert a provenance the model does not have. An empty model has no
        # non-invertible node either, so the L3 guard is unaffected either way -- this is a
        # question of honest labelling, not of safety.
        return "specified"
    pinned = sum(1 for fit in fits if fit.pinned)
    if pinned == 0:
        return "fitted"
    return "specified" if pinned == len(fits) else "mixed"


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
    if not members:
        # No orientation of the undirected edges is acyclic, so the CPDAG's *directed* edges
        # already contain a cycle and no DAG is consistent with it. Returning [] here would hand
        # back an empty belief that reads exactly like "not fitted yet" -- a caller's next query
        # then fails with a misleading message about calling this function, when the real fault is
        # an inconsistent graph. This function already refuses to truncate a class it cannot
        # enumerate; refusing an empty one is the same principle.
        raise NotIdentifiableError(
            f"no DAG is consistent with this CPDAG, so its equivalence class is empty: the "
            f"directed edges {sorted(cpdag.directed_edges)} admit no acyclic orientation of the "
            f"{undirected_count} undirected one(s). A cyclic directed set usually means the "
            f"orientations came from separate sources that were not reconciled -- "
            f"discover_interventional orients the edges incident to each intervention target "
            f"independently, so two or more targets can disagree. Reconcile them (or drop a "
            f"target) before fitting.",
            witness=sorted(cpdag.directed_edges),
        )
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
