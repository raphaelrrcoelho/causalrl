"""A *fitted* SCM used as a reinforcement-learning environment.

`fit_scm` returns an ordinary :class:`~causalrl.scm.scm.StructuralCausalModel`, and every
SCM-backed environment surface takes one (``envs/base.py:19``). These tests pin that the
composition actually executes: fit a world model from a confounded table, hand it to
:class:`~causalrl.envs.suite.scbandit.StructuralCausalBanditEnv`, wrap it in
:class:`~causalrl.envs.wrapper.CausalEnvWrapper`, and *act* in it -- the model-learning half of
model-based RL meeting the acting half.
"""

from __future__ import annotations

import numpy as np
import pytest

from causalrl.envs.suite.scbandit import StructuralCausalBanditEnv
from causalrl.envs.suite.simpson_bandit import SimpsonBandit
from causalrl.envs.wrapper import CausalEnvWrapper
from causalrl.scm.fit import fit_scm
from causalrl.scm.graph import CausalGraph
from causalrl.scm.scm import StructuralCausalModel

# Z confounds A and Y; action 1 is interventionally better (0.50 vs 0.40) while the naive
# marginal reverses. Ground truth lives on SimpsonBandit.true_action_value.
_GRAPH = CausalGraph(directed_edges=[("Z", "A"), ("Z", "Y"), ("A", "Y")])
_ARMS = [{}, {"A": 0}, {"A": 1}]


def _fitted_world_model(n: int = 20_000, seed: int = 0) -> StructuralCausalModel:
    data = SimpsonBandit(seed=seed).sample(n, seed=seed)
    return fit_scm(data, graph=_GRAPH, seed=seed)


def _fitted_env(n_mc: int = 20_000, seed: int = 0) -> CausalEnvWrapper:
    scm = _fitted_world_model(seed=seed)
    env = StructuralCausalBanditEnv(scm, _GRAPH, "Y", ["A"], {"A": [0, 1]}, n_mc=n_mc, seed=seed)
    return CausalEnvWrapper(env, reward_node="Y")


def test_fitted_scm_drives_the_causal_env_interface() -> None:
    """The causal handles come up on a learned model, and report the learned structure."""
    wrapped = _fitted_env(n_mc=200)
    assert wrapped.has_causal_interface
    scm = wrapped.scm
    assert scm is not None
    assert scm.provenance == "fitted"
    assert scm.fit_report is not None
    # The reward's parents are read off the *fitted* model -- they name the confounder.
    assert sorted(wrapped.reward_parents) == ["A", "Z"]
    # Mutilating a fitted model keeps it fitted, so the L3 guard still applies downstream.
    assert wrapped.do({"A": 1.0}).provenance == "fitted"


def test_acting_in_the_fitted_model_recovers_the_interventional_optimum() -> None:
    """Rollout rewards inside the learned model track the *true* interventional values.

    The logged data is confounded: E[Y|A=0] > E[Y|A=1] while E[Y|do(A=0)] < E[Y|do(A=1)].
    Stepping the learned model's do-arms must follow the interventional order, not the
    observational one -- otherwise the world model would be an RL environment that teaches
    the wrong action.
    """
    bandit = SimpsonBandit(seed=0)
    data = bandit.sample(20_000, seed=0)
    observational = [float(data["Y"][data["A"] == a].mean()) for a in (0, 1)]
    assert observational[0] > observational[1]  # the trap the logs set

    wrapped = _fitted_env(n_mc=20_000)
    env = wrapped.unwrapped
    assert isinstance(env, StructuralCausalBanditEnv)
    assert env.arms == _ARMS

    # The model's own arm values recover the oracle interventional values.
    for index, action in ((1, 0), (2, 1)):
        assert env.arm_values[index] == pytest.approx(bandit.true_action_value(action), abs=0.02)
    assert env.arms[int(np.argmax(env.arm_values))] == {"A": 1}

    # ...and so do sampled rollout rewards, which is what an agent actually consumes.
    # The env is seeded, so this rollout is deterministic rather than statistically flaky.
    rewards: dict[int, list[float]] = {1: [], 2: []}
    wrapped.reset(seed=0)
    for _ in range(700):
        for arm_index in (1, 2):
            _, reward, _terminated, _truncated, _info = wrapped.step(arm_index)
            rewards[arm_index].append(float(reward))
            wrapped.reset()
    assert float(np.mean(rewards[2])) > float(np.mean(rewards[1])) + 0.03


def test_persistent_intervention_changes_the_fitted_model_rollout() -> None:
    """``set_intervention`` reroutes sampling through the mutilated *fitted* model.

    The intervention is on ``Z`` -- the confounder, not the action -- which no bandit arm can
    reach. Pinning it is a capability the learned *model* has and a reward table does not.
    ``E[Y | do(Z=0)] = 0.62`` against the unintervened ``E[Y] = 0.45``.
    """
    wrapped = _fitted_env(n_mc=200)
    observational_arm = 0

    def mean_reward(steps: int = 500) -> float:
        total = 0.0
        wrapped.reset(seed=0)
        for _ in range(steps):
            _, reward, _terminated, _truncated, _info = wrapped.step(observational_arm)
            total += float(reward)
            wrapped.reset()
        return total / steps

    baseline = mean_reward()
    wrapped.set_intervention({"Z": 0.0})
    assert wrapped.active_interventions == {"Z": 0.0}
    intervened = mean_reward()
    wrapped.clear_intervention()
    assert wrapped.active_interventions is None
    restored = mean_reward()

    assert intervened > baseline + 0.08
    assert restored == pytest.approx(baseline, abs=1e-9)
