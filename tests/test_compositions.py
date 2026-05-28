"""Multi-primitive compositions on causalrl.

These tests exercise three compositions of library primitives that are not directly packaged as
single-shot calls in the public API or covered by the other tests. Each asserts a property of the
*composition*, not of any single primitive:

* **Discovery-driven intervention selection.** PC discovery + a conservative equivalence-class
  POMIS (union of ``pomis()`` over the acyclic orientations consistent with the discovered CPDAG) +
  a Thompson-sampling bandit over only the surviving arms. Asserts soundness (the discovered class
  contains the oracle POMIS), substantial arm pruning, and a sample-efficiency win in cumulative
  regret over a graph-blind baseline.
* **MSM Γ-lower bound as a diagnostic on offline action ranking.** Per-action MSM Γ-lower bound on
  ``E[Y(a)]`` via :func:`ipw_sensitivity_bounds` (built on the action's logged units and the
  per-stratum nominal propensity). Asserts that, under hidden confounding strong enough to flip the
  naive (Z-only) IPW ranking, at a Γ that brackets the true confounding odds ratio the MSM lower
  bound is valid on each action's truth AND is meaningfully below the naive point estimate on the
  action naive picks — exposing its over-optimism. The composition does not promise that
  argmax-lower picks the truly-best action; at moderate Γ on bounded outcomes the lower bound is
  often too loose to drive action selection.
* **Bootstrap confidence interval for general transport.** A paired percentile bootstrap around
  :func:`estimate_transported_effect` (resampling source and target rows with replacement) on a
  known covariate-shift SCM. Asserts non-degenerate width, coverage of the true target effect, and
  exclusion of a naive source-reuse point estimate from that interval.
"""

from __future__ import annotations

import itertools

import networkx as nx
import numpy as np
import torch
from torch.distributions import Bernoulli, Distribution, Uniform

from causalrl.discovery import discover
from causalrl.envs.suite.scbandit import StructuralCausalBanditEnv
from causalrl.identification.bounds import ipw_sensitivity_bounds
from causalrl.identification.id_algorithm import (
    estimate_effect,
    estimate_transported_effect,
)
from causalrl.identification.intervention_sets import pomis
from causalrl.scm.graph import CausalGraph
from causalrl.scm.mechanisms import FunctionalMechanism, Mechanism
from causalrl.scm.scm import StructuralCausalModel


# =================================================================================================
# Composition 1 — Discovery + equivalence-class POMIS + Thompson sampling
# =================================================================================================
def _build_distractor_scm() -> tuple[StructuralCausalModel, CausalGraph]:
    """One relevant lever A drives Y; B, C, D are isolated distractor levers."""
    nodes = ["A", "B", "C", "D", "Y"]
    graph = CausalGraph(directed_edges=[("A", "Y")], nodes=nodes)
    mechanisms: dict[str, Mechanism] = {
        "A": FunctionalMechanism([], lambda pa, u: u),
        "B": FunctionalMechanism([], lambda pa, u: u),
        "C": FunctionalMechanism([], lambda pa, u: u),
        "D": FunctionalMechanism([], lambda pa, u: u),
        "Y": FunctionalMechanism(["A"], lambda pa, u: (u < (0.15 + 0.70 * pa["A"])).float()),
    }
    exo: dict[str, Distribution] = {
        "A": Bernoulli(0.5),
        "B": Bernoulli(0.5),
        "C": Bernoulli(0.5),
        "D": Bernoulli(0.5),
        "Y": Uniform(0.0, 1.0),
    }
    return StructuralCausalModel(graph, mechanisms, exo), graph


def _equivalence_class_pomis(cpdag, reward: str, manipulable: set[str]) -> set[frozenset[str]]:
    """Union of ``pomis()`` over every acyclic orientation of the CPDAG's undirected edges.

    A conservative equivalence-class POMIS: a superset of the true-graph POMIS (so the optimum's
    arm is always included), and small whenever discovery has determined most edges.
    """
    und = [tuple(sorted(e)) for e in cpdag.undirected_edges]
    base = set(cpdag.directed_edges)
    out: set[frozenset[str]] = set()
    for bits in itertools.product([0, 1], repeat=len(und)):
        directed = set(base)
        for (a, b), bit in zip(und, bits, strict=True):
            directed.add((a, b) if bit == 0 else (b, a))
        if not nx.is_directed_acyclic_graph(nx.DiGraph(list(directed))):
            continue
        g = CausalGraph(directed_edges=sorted(directed), nodes=list(cpdag.variables))
        out |= set(pomis(g, reward, manipulable=manipulable))
    return out


class _ArmTS:
    """Beta-Bernoulli Thompson sampling over an explicit list of allowed global arm indices."""

    def __init__(self, allowed: list[int], seed: int) -> None:
        self.allowed = allowed
        self._a = np.ones(len(allowed))
        self._b = np.ones(len(allowed))
        self._rng = np.random.default_rng(seed)
        self._local = {global_i: i for i, global_i in enumerate(allowed)}

    def act(self) -> int:
        return self.allowed[int(np.argmax(self._rng.beta(self._a, self._b)))]

    def update(self, arm: int, reward: float) -> None:
        i = self._local[arm]
        self._a[i] += reward
        self._b[i] += 1.0 - reward


def _cumulative_regret(
    env: StructuralCausalBanditEnv, allowed: list[int], seed: int, steps: int
) -> float:
    agent = _ArmTS(allowed, seed)
    env.reset(seed=seed)
    total = 0.0
    for _ in range(steps):
        arm = agent.act()
        _, reward, _, _, _ = env.step(arm)
        agent.update(arm, reward)
        total += env.optimal_value - reward
    return total


def test_discovery_driven_intervention_selection() -> None:
    """PC + equivalence-class POMIS + TS substantially beats graph-blind TS in regret."""
    scm, true_graph = _build_distractor_scm()
    manipulable = {"A", "B", "C", "D"}
    domains = {v: [0, 1] for v in manipulable}
    env = StructuralCausalBanditEnv(
        scm, true_graph, "Y", list(manipulable), domains, n_mc=3000, seed=0
    )

    # Phase 1: discover the CPDAG from observational data, then take the equivalence-class POMIS.
    samples = scm.see(5000, seed=0)
    data = {v: samples[v].long().numpy() for v in true_graph.nodes}
    cpdag = discover(data, list(true_graph.nodes))
    learned = _equivalence_class_pomis(cpdag, "Y", manipulable)
    oracle = set(pomis(true_graph, "Y", manipulable=manipulable))

    # Soundness: the conservative class contains every truly-possibly-optimal set; the only
    # relevant lever A is present in every learned set or as itself.
    assert oracle <= learned, f"learned class {learned} drops oracle members {oracle - learned}"
    assert any("A" in s for s in learned)
    # Pruning: discovery rejected the three distractor levers B, C, D entirely.
    for distractor in ("B", "C", "D"):
        assert all(distractor not in s for s in learned), f"distractor {distractor} survived"

    # Phase 2: TS over the discovered class vs. brute-force TS over every arm.
    pruned_arms = [i for i, arm in enumerate(env.arms) if frozenset(arm) in learned]
    all_arms = list(range(len(env.arms)))
    assert len(all_arms) >= 5 * len(pruned_arms), (
        "discovery did not substantially prune the arm space"
    )

    steps, seeds = 1200, range(3)
    pruned_regret = float(
        np.mean([_cumulative_regret(env, pruned_arms, 10 + s, steps) for s in seeds])
    )
    brute_regret = float(np.mean([_cumulative_regret(env, all_arms, 10 + s, steps) for s in seeds]))
    assert brute_regret > 2.0 * pruned_regret, (
        f"composition did not deliver a sample-efficiency win: "
        f"brute={brute_regret:.1f} vs pruned={pruned_regret:.1f}"
    )


# =================================================================================================
# Composition 2 — MSM Γ-lower bound argmax for offline action selection
# =================================================================================================
def _generate_confounded_offline_data(n: int, seed: int) -> dict[str, np.ndarray]:
    """One observed covariate Z, one hidden confounder U, action A, outcome Y.

    Confounding pushes A=1 toward high-U units where Y is naturally high, so A=1 looks great by
    logged mean and by Z-only IPW, but the true effect prefers A=0.
    """
    rng = np.random.default_rng(seed)
    z = rng.integers(0, 2, n)
    u = rng.integers(0, 2, n)
    p_a1 = 0.10 + 0.40 * z + 0.40 * u  # P(A=1|Z,U) in {0.1, 0.5, 0.5, 0.9}
    a = (rng.random(n) < p_a1).astype(int)
    p_y1 = 0.10 + 0.60 * u + 0.05 * z + 0.15 * (1 - a)  # A=0 truly adds +0.15
    y = (rng.random(n) < p_y1).astype(float)
    return {"Z": z, "U": u, "A": a, "Y": y}


def _true_do_value(action: int) -> float:
    total = 0.0
    for z in (0, 1):
        for u in (0, 1):
            total += 0.25 * (0.10 + 0.60 * u + 0.05 * z + 0.15 * (1 - action))
    return total


def _msm_lower_action(data: dict[str, np.ndarray], action: int, gamma: float) -> float:
    """MSM Γ-lower bound on ``E[Y | do(A=action)]`` using ``ipw_sensitivity_bounds``.

    Per-unit nominal propensity ``e_a(Z)`` is the empirical share of ``A=action`` within the unit's
    Z stratum (the only adjustment available to a practitioner who does not observe U). Passed to
    :func:`ipw_sensitivity_bounds` over the units that actually took ``action``: at Γ=1 this returns
    the Hájek IPW point, widening monotonically with Γ.
    """
    z, a, y = data["Z"], data["A"], data["Y"]
    e_per_unit = np.where(z == 1, (a == action)[z == 1].mean(), (a == action)[z == 0].mean())
    mask = a == action
    return float(ipw_sensitivity_bounds(y[mask], e_per_unit[mask], gamma=gamma)[0])


def test_msm_lower_bound_diagnoses_naive_offline_ranking() -> None:
    """MSM Γ-lower bound is valid for each action and flags the naive winner as over-optimistic."""
    data = _generate_confounded_offline_data(n=8000, seed=0)
    truth = {a: _true_do_value(a) for a in (0, 1)}
    assert truth[0] > truth[1], f"setup invariant broken: truth={truth}"

    # The naive Z-adjusted IPW (= MSM at Γ=1, Hájek with Z-only nominal propensities) ranks the
    # actions backwards under hidden U confounding.
    naive_point = {a: _msm_lower_action(data, a, gamma=1.0) for a in (0, 1)}
    naive_pick = max(naive_point, key=naive_point.get)
    true_best = max(truth, key=truth.get)
    assert naive_pick != true_best, (
        f"setup invariant broken: naive already correct, naive={naive_point}, truth={truth}"
    )

    # Validity: at a Γ bracketing the worst-case true/nominal odds ratio (~3.86 by construction),
    # the MSM Γ-lower bound is a valid lower bound on E[Y(a)] for each action.
    gamma = 4.0
    msm_lower = {a: _msm_lower_action(data, a, gamma) for a in (0, 1)}
    for a in (0, 1):
        assert msm_lower[a] <= truth[a] + 0.02, (
            f"MSM lower bound violated truth at gamma={gamma} for a={a}: "
            f"lower={msm_lower[a]:.3f} vs truth={truth[a]:.3f}"
        )
    # Diagnostic value: the robust lower bound is meaningfully below the naive point estimate for
    # the confounded winner — it flags naive's value as over-optimistic.
    assert msm_lower[naive_pick] < naive_point[naive_pick] - 0.10, (
        f"MSM bound did not flag naive overoptimism: lower={msm_lower[naive_pick]:.3f} "
        f"vs naive_point={naive_point[naive_pick]:.3f}"
    )


# =================================================================================================
# Composition 3 — Bootstrap confidence interval for general transport
# =================================================================================================
_GRAPH_TRANSPORT = CausalGraph(directed_edges=[("Z", "X"), ("Z", "Y"), ("X", "Y")])


def _y_mech_transport(pa: dict[str, torch.Tensor], u: torch.Tensor) -> torch.Tensor:
    x, z = pa["X"], pa["Z"]
    p = 0.10 * (1 - x) * (1 - z) + 0.90 * x * (1 - z) + 0.70 * (1 - x) * z + 0.20 * x * z
    return (u < p).float()


def _build_transport_scm(p_z: float) -> StructuralCausalModel:
    mechanisms: dict[str, Mechanism] = {
        "Z": FunctionalMechanism([], lambda pa, u: u),
        "X": FunctionalMechanism(["Z"], lambda pa, u: (u < (0.3 + 0.4 * pa["Z"])).float()),
        "Y": FunctionalMechanism(["X", "Z"], _y_mech_transport),
    }
    exo: dict[str, Distribution] = {
        "Z": Bernoulli(p_z),
        "X": Uniform(0.0, 1.0),
        "Y": Uniform(0.0, 1.0),
    }
    return StructuralCausalModel(_GRAPH_TRANSPORT, mechanisms, exo)


def _scm_columns(scm: StructuralCausalModel, n: int, seed: int) -> dict[str, np.ndarray]:
    s = scm.see(n, seed=seed)
    return {v: s[v].long().numpy() for v in ("X", "Y", "Z")}


def _bootstrap_transport_ci(
    source_data: dict[str, np.ndarray],
    target_data: dict[str, np.ndarray],
    *,
    do: dict[str, int],
    n_boot: int,
    alpha: float,
    seed: int,
) -> tuple[float, float, float]:
    """``(point, ci_lo, ci_hi)`` for ``P(Y=1 | do(X = do["X"]))`` in the target via a paired
    percentile bootstrap that resamples source and target rows with replacement."""
    rng = np.random.default_rng(seed)
    nodes = list(_GRAPH_TRANSPORT.nodes)
    n_source = len(source_data[nodes[0]])
    n_target = len(target_data[nodes[0]])
    point = estimate_transported_effect(
        _GRAPH_TRANSPORT, ["X"], ["Y"], ["Z"], source_data, target_data, do=do
    )[(1,)]
    boots: list[float] = []
    for _ in range(n_boot):
        i = rng.integers(0, n_source, n_source)
        j = rng.integers(0, n_target, n_target)
        src_b = {v: source_data[v][i] for v in nodes}
        tgt_b = {v: target_data[v][j] for v in nodes}
        boots.append(
            estimate_transported_effect(_GRAPH_TRANSPORT, ["X"], ["Y"], ["Z"], src_b, tgt_b, do=do)[
                (1,)
            ]
        )
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return point, float(lo), float(hi)


def test_bootstrap_ci_for_general_transport() -> None:
    """Bootstrap CI around ``estimate_transported_effect`` covers truth where naive reuse misses."""
    source = _build_transport_scm(p_z=0.85)
    target = _build_transport_scm(p_z=0.15)
    source_data = _scm_columns(source, n=4000, seed=0)
    target_data = _scm_columns(target, n=4000, seed=1)

    do = {"X": 1}
    point, ci_lo, ci_hi = _bootstrap_transport_ci(
        source_data, target_data, do=do, n_boot=150, alpha=0.10, seed=42
    )
    truth = float(target.do({"X": 1.0}).see(20_000, seed=7)["Y"].float().mean().item())
    naive = estimate_effect(_GRAPH_TRANSPORT, ["X"], ["Y"], source_data, do=do)[(1,)]

    assert ci_hi > ci_lo + 0.01, f"CI degenerate: [{ci_lo:.3f}, {ci_hi:.3f}]"
    assert ci_lo <= truth <= ci_hi, (
        f"CI [{ci_lo:.3f}, {ci_hi:.3f}] does not cover truth {truth:.3f}"
    )
    assert abs(point - truth) < 0.05, f"transport point {point:.3f} far from truth {truth:.3f}"
    assert naive < ci_lo or naive > ci_hi, (
        f"naive source estimate {naive:.3f} fell inside the target CI [{ci_lo:.3f}, {ci_hi:.3f}]"
    )
