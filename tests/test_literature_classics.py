"""Classic causal-inference cases reproduced with causalrl.

Each test reproduces a textbook result using the library's own algorithms — a "reproducing the
literature" gallery that doubles as integration coverage. The cases: Simpson's paradox (kidney
stones), the front-door criterion (smoking → tar → cancer), Pearl's napkin, the instrumental
variable (point-unidentified but bounded), the bow arc, cross-domain transport (LA → NYC), and the
multi-armed bandit with unobserved confounders (MABUC).

It also collects difficult RL problems solved better by causal than associational RL — MABUC
(a confounding-aware bandit beats the naive one), the counterfactual "follow your intuition" bandit
(acting on the counterfactual beats any fixed interventional arm), and curriculum-driven hard
exploration (a causal prerequisite curriculum reaches a sparse goal flat Q-learning misses).
"""

from __future__ import annotations

import numpy as np
import pytest
from torch import Tensor
from torch.distributions import Bernoulli, Distribution, Uniform

from causalrl.agents.bandits import CausalThompsonSampling, NaiveThompsonSampling
from causalrl.curriculum import curriculum_q_learning
from causalrl.envs.suite.counterfactual_bandit import make_counterfactual_bandit_env
from causalrl.envs.suite.mabuc import MABUCEnv
from causalrl.identification.bounds import manski_bounds
from causalrl.identification.id_algorithm import (
    estimate_effect,
    identify_transport,
    is_identifiable_effect,
    is_transportable_effect,
)
from causalrl.scm.graph import CausalGraph
from causalrl.scm.mechanisms import FunctionalMechanism, Mechanism
from causalrl.scm.scm import StructuralCausalModel
from causalrl.shaping import TabularMDP, q_learning

_N = 40_000


def _flip(u: Tensor, p: float) -> Tensor:
    return (u < p).float()


def _cols(scm: StructuralCausalModel, keep: list[str], *, seed: int) -> dict[str, np.ndarray]:
    s = scm.see(_N, seed=seed)
    return {v: s[v].long().numpy() for v in keep}


def _true_do_y(scm: StructuralCausalModel, value: int) -> float:
    return float(scm.do({"X": float(value)}).see(_N, seed=7)["Y"].float().mean())


# --- Simpson's paradox (kidney stones): adjustment reverses the naive sign --------------------
def _kidney_stones_scm() -> StructuralCausalModel:
    # Z = severe case (confounder): severe cases get treatment X=1 more often and recover less; X
    # helps within every stratum, but the naive association favours X=0.
    graph = CausalGraph(directed_edges=[("Z", "X"), ("Z", "Y"), ("X", "Y")])
    mechanisms: dict[str, Mechanism] = {
        "Z": FunctionalMechanism([], lambda pa, u: u),
        "X": FunctionalMechanism(["Z"], lambda pa, u: (u < (0.2 + 0.6 * pa["Z"])).float()),
        "Y": FunctionalMechanism(
            ["X", "Z"],
            lambda pa, u: (u < (0.9 - 0.4 * pa["Z"] + pa["X"] * (0.05 + 0.15 * pa["Z"]))).float(),
        ),
    }
    exo: dict[str, Distribution] = {
        "Z": Bernoulli(0.5),
        "X": Uniform(0.0, 1.0),
        "Y": Uniform(0.0, 1.0),
    }
    return StructuralCausalModel(graph, mechanisms, exo)


def test_simpsons_paradox_adjustment_reverses_naive_sign() -> None:
    scm = _kidney_stones_scm()
    data = _cols(scm, ["Z", "X", "Y"], seed=0)
    graph = CausalGraph(directed_edges=[("Z", "X"), ("Z", "Y"), ("X", "Y")])
    y, x = data["Y"], data["X"]
    naive = y[x == 1].mean() - y[x == 0].mean()
    p1 = estimate_effect(graph, {"X"}, {"Y"}, data, do={"X": 1})[(1,)]
    p0 = estimate_effect(graph, {"X"}, {"Y"}, data, do={"X": 0})[(1,)]
    causal = p1 - p0
    assert naive < 0 < causal  # the paradox: naive prefers X=0, the causal effect prefers X=1
    assert p1 == pytest.approx(_true_do_y(scm, 1), abs=0.02)


# --- Front-door criterion (smoking -> tar -> cancer, with smoking <-> cancer confounding) -----
def _front_door_scm() -> StructuralCausalModel:
    graph = CausalGraph(directed_edges=[("U", "X"), ("U", "Y"), ("X", "M"), ("M", "Y")])
    mechanisms: dict[str, Mechanism] = {
        "U": FunctionalMechanism([], lambda pa, u: u),
        "X": FunctionalMechanism(["U"], lambda pa, u: (pa["U"] + _flip(u, 0.25)) % 2),
        "M": FunctionalMechanism(["X"], lambda pa, u: (pa["X"] + _flip(u, 0.1)) % 2),
        "Y": FunctionalMechanism(
            ["M", "U"], lambda pa, u: ((((pa["M"] + pa["U"]) > 0).float()) + _flip(u, 0.05)) % 2
        ),
    }
    exo: dict[str, Distribution] = {
        "U": Bernoulli(0.5),
        "X": Uniform(0.0, 1.0),
        "M": Uniform(0.0, 1.0),
        "Y": Uniform(0.0, 1.0),
    }
    return StructuralCausalModel(graph, mechanisms, exo)


def test_front_door_smoking_tar_cancer() -> None:
    scm = _front_door_scm()
    data = _cols(scm, ["X", "M", "Y"], seed=0)
    graph = CausalGraph(directed_edges=[("X", "M"), ("M", "Y")], bidirected_edges=[("X", "Y")])
    assert is_identifiable_effect(graph, {"X"}, {"Y"}) is True  # front-door makes it identifiable
    estimate = estimate_effect(graph, {"X"}, {"Y"}, data, do={"X": 1})[(1,)]
    assert estimate == pytest.approx(_true_do_y(scm, 1), abs=0.03)


# --- Pearl's napkin: identifiable despite latent confounding --------------------------------
def test_napkin_is_identifiable() -> None:
    graph = CausalGraph(
        directed_edges=[("R", "W"), ("W", "X"), ("X", "Y")],
        bidirected_edges=[("R", "X"), ("R", "Y")],
    )
    assert is_identifiable_effect(graph, {"X"}, {"Y"}) is True


# --- Instrumental variable: not point-identified, but Manski-bounded ------------------------
def _iv_scm() -> StructuralCausalModel:
    graph = CausalGraph(directed_edges=[("Z", "X"), ("U", "X"), ("U", "Y"), ("X", "Y")])
    mechanisms: dict[str, Mechanism] = {
        "Z": FunctionalMechanism([], lambda pa, u: u),
        "U": FunctionalMechanism([], lambda pa, u: u),
        "X": FunctionalMechanism(
            ["Z", "U"], lambda pa, u: ((((pa["Z"] + pa["U"]) > 0).float()) + _flip(u, 0.1)) % 2
        ),
        "Y": FunctionalMechanism(
            ["X", "U"], lambda pa, u: ((((pa["X"] + pa["U"]) > 0).float()) + _flip(u, 0.05)) % 2
        ),
    }
    exo: dict[str, Distribution] = {
        "Z": Bernoulli(0.5),
        "U": Bernoulli(0.5),
        "X": Uniform(0.0, 1.0),
        "Y": Uniform(0.0, 1.0),
    }
    return StructuralCausalModel(graph, mechanisms, exo)


def test_instrumental_variable_unidentified_but_bounded() -> None:
    admg = CausalGraph(directed_edges=[("Z", "X"), ("X", "Y")], bidirected_edges=[("X", "Y")])
    assert is_identifiable_effect(admg, {"X"}, {"Y"}) is False  # IV alone does not point-identify
    scm = _iv_scm()
    data = _cols(scm, ["X", "Y"], seed=0)
    lo, hi = manski_bounds(data, treatment="X", outcome="Y", action=1)
    assert lo <= _true_do_y(scm, 1) <= hi  # but the no-assumptions bounds still contain the truth


# --- The bow arc: the simplest non-identifiable confounded effect ---------------------------
def test_bow_arc_not_identifiable() -> None:
    graph = CausalGraph(directed_edges=[("X", "Y")], bidirected_edges=[("X", "Y")])
    assert is_identifiable_effect(graph, {"X"}, {"Y"}) is False


# --- Transportability: covariate shift LA -> NYC --------------------------------------------
def test_covariate_shift_transport_la_to_nyc() -> None:
    graph = CausalGraph(directed_edges=[("Z", "X"), ("Z", "Y"), ("X", "Y")])
    assert is_transportable_effect(graph, {"X"}, {"Y"}, ["Z"]) is True
    formula = identify_transport(graph, {"X"}, {"Y"}, ["Z"]).render()
    assert "do(" not in formula and "P_target(" in formula and "P_source(" in formula


# --- MABUC: a confounding-aware causal bandit beats the naive one ---------------------------
def test_mabuc_causal_agent_beats_naive() -> None:
    def run(agent_factory, *, steps: int = 6000) -> float:
        env = MABUCEnv(seed=1)
        agent = agent_factory()
        obs, _ = env.reset(seed=1)
        total = 0.0
        for _ in range(steps):
            action = agent.act(obs)
            _, reward, _, _, _ = env.step(action)
            agent.update(obs, action, reward)
            total += reward
            obs, _ = env.reset()
        return total / steps

    causal = run(lambda: CausalThompsonSampling(n_arms=2, n_contexts=2, seed=0))
    naive = run(lambda: NaiveThompsonSampling(n_arms=2, seed=0))
    assert causal > naive + 0.05  # conditioning on the "intuition" (context) wins under confounding


# --- Counterfactual decision-making: acting on intuition beats any fixed (do) arm -----------
def test_counterfactual_follow_intuition_beats_best_fixed_arm() -> None:
    # The "Greedy Casino": the counterfactual-optimal policy plays its intuition I and earns ~0.8,
    # while every fixed interventional arm do(X=a) earns only ~0.367 — an associational/optimal-arm
    # policy cannot match an agent that reasons counterfactually (Layer 3).
    env = make_counterfactual_bandit_env(seed=0)
    n = 4000

    def mean_reward(policy) -> float:
        total = 0.0
        for t in range(n):
            obs, _ = env.reset(seed=t)
            _, reward, _, _, _ = env.step(policy(obs))
            total += reward
        return total / n

    counterfactual = mean_reward(lambda obs: obs["intuition"])
    best_fixed = max(mean_reward(lambda obs, arm=arm: arm) for arm in (0, 1, 2))
    assert counterfactual > best_fixed + 0.2  # ~0.8 vs ~0.367


# --- Hard exploration: a causal prerequisite curriculum solves a sparse goal flat RL misses --
def test_curriculum_solves_sparse_goal_flat_rl_misses() -> None:
    length, goal = 12, 11

    def chain_task(target: int) -> TabularMDP:
        transitions: dict[tuple[int, int], int] = {}
        rewards: dict[tuple[int, int], float] = {}
        for s in range(length):
            left, right = max(s - 1, 0), min(s + 1, length - 1)
            transitions[(s, 0)], transitions[(s, 1)] = left, right
            rewards[(s, 0)] = 1.0 if left == target else 0.0
            rewards[(s, 1)] = 1.0 if right == target else 0.0
        return TabularMDP(length, 2, transitions, rewards, frozenset({target}), gamma=0.95)

    def reaches(policy: dict[int, int]) -> bool:
        s = 0
        for _ in range(4 * length):
            if s == goal:
                return True
            s = max(s - 1, 0) if policy[s] == 0 else min(s + 1, length - 1)
        return s == goal

    tasks = [chain_task(target) for target in range(1, length)]
    budget = 25 * len(tasks)
    curriculum = curriculum_q_learning(tasks, episodes_per_task=25, seed=0)
    flat = q_learning(chain_task(goal), episodes=budget, seed=0)
    assert reaches(curriculum) and not reaches(flat)
