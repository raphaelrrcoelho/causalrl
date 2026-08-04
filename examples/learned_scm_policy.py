"""Learn the world model from logs, then act inside it -- a policy from a fitted SCM.

    uv run python examples/learned_scm_policy.py

`fit_scm` is the model-learning half of model-based RL. This example closes the loop on the
acting half that already exists in the library, with no new machinery:

    confounded logs --fit_scm--> StructuralCausalModel --StructuralCausalBanditEnv--> gym.Env
                    --Thompson sampling--> a policy --evaluated in the TRUE world--> regret

Two world models are fitted **from the same rows by the same fitter on the same split** (two
`fit_scm` calls sharing `seed=0`, so the train/holdout permutation is identical), differing only in
the causal structure they are given:

* **causal** -- ``Z -> A``, ``Z -> Y``, ``A -> Y``: the confounder is a parent of the reward.
* **confounder-blind** -- ``A -> Y``: ``Z`` is in the data but not in the model.

Both reproduce the *observational* conditional ``E[Y | A]`` to within Monte-Carlo error, so fit
quality is not what separates them; structure is. (The same isolation the `fit_scm` oracle gate
uses -- ``examples/learned_scm_oracle_gate.py``.)

**The regime this is about, stated up front.** A learned causal world model earns its keep when
the logs are *confounded* and *offline* -- when the action you would read off ``E[Y | A]`` is not
the action ``E[Y | do(A)]`` implies. That is a narrow claim, and the second regime below shows the
other side of it: on a randomized log the causal model buys exactly nothing, because there was
nothing to fix. This is not a general RL improvement, and it is not an accuracy claim -- the
project's real-data suite (``docs/causal_mbrl_agent/REAL_DATA.md``) already established that causal
*point estimates* do not reliably beat strong contenders. What the structure buys here is the
*decision*: which arm the agent converges on, and whether that arm is the right one.

The world is `SimpsonBandit` (`envs/suite/simpson_bandit.py`), which carries exact ground truth:
``E[Y|do(A=0)] = 0.40``, ``E[Y|do(A=1)] = 0.50``, while the naive marginal reverses to prefer
action 0. The policies are scored in that true world, never in the model that chose them.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import NamedTuple

import numpy as np

from causalrl.agents.bandits import NaiveThompsonSampling
from causalrl.envs.suite.scbandit import StructuralCausalBanditEnv
from causalrl.envs.suite.simpson_bandit import SimpsonBandit
from causalrl.envs.wrapper import CausalEnvWrapper
from causalrl.scm.fit import fit_scm
from causalrl.scm.graph import CausalGraph
from causalrl.scm.scm import StructuralCausalModel

CAUSAL_GRAPH = CausalGraph(directed_edges=[("Z", "A"), ("Z", "Y"), ("A", "Y")])
"""The confounder is a parent of the reward, so ``do(A)`` back-door-adjusts for it."""

BLIND_GRAPH = CausalGraph(directed_edges=[("A", "Y")])
"""``Z`` is logged but unmodelled, so ``do(A)`` collapses to the observational ``E[Y | A]``."""


class ModelResult(NamedTuple):
    """What one world model believed, what it chose, and what that choice was really worth."""

    name: str
    reward_parents: tuple[str, ...]
    implied_conditional: tuple[float, float]
    arm_values: tuple[float, ...]
    chosen_arms: tuple[str, ...]
    mean_true_value: float
    mean_regret: float

    def summary(self) -> str:
        conditional = ", ".join(f"{v:.3f}" for v in self.implied_conditional)
        values = ", ".join(f"{v:.3f}" for v in self.arm_values)
        return (
            f"  {self.name:<17} reward_parents={list(self.reward_parents)}\n"
            f"  {'':<17} implied E[Y|A=0,1]=({conditional})  in-model arm values=({values})\n"
            f"  {'':<17} chose {list(self.chosen_arms)}\n"
            f"  {'':<17} true value={self.mean_true_value:.3f}  regret={self.mean_regret:.3f}"
        )


class RegimeResult(NamedTuple):
    """Both world models' outcomes on one kind of log, plus the log's own naive contrast."""

    regime: str
    empirical_conditional: tuple[float, float]
    models: tuple[ModelResult, ...]

    def summary(self) -> str:
        conditional = ", ".join(f"{v:.3f}" for v in self.empirical_conditional)
        head = f"=== {self.regime} ===\n  logged E[Y|A=0,1]=({conditional})"
        return "\n".join([head, *(model.summary() for model in self.models)])


def randomized_log(bandit: SimpsonBandit, n: int, seed: int) -> dict[str, np.ndarray]:
    """An *unconfounded* log: half the rows under ``do(A=0)``, half under ``do(A=1)``.

    The control regime. ``A`` is assigned independently of ``Z``, so ``E[Y | A] = E[Y | do(A)]``
    and a confounder-blind world model is already right.
    """
    half = n // 2
    control = bandit.sample_do({"A": 0}, half, seed=seed)
    treated = bandit.sample_do({"A": 1}, n - half, seed=seed + 1)
    return {name: np.concatenate([control[name], treated[name]]) for name in control}


def _conditional(data: Mapping[str, np.ndarray]) -> tuple[float, float]:
    """``(E[Y | A=0], E[Y | A=1])`` -- the number a correlational learner would act on."""
    outcome, action = np.asarray(data["Y"], dtype=float), np.asarray(data["A"])
    return tuple(float(outcome[action == a].mean()) for a in (0, 1))  # type: ignore[return-value]


def world_model_env(
    data: Mapping[str, np.ndarray], graph: CausalGraph, *, seed: int, n_mc: int
) -> tuple[StructuralCausalModel, CausalEnvWrapper]:
    """Fit an SCM from ``data`` under ``graph`` and expose it as a Gymnasium bandit.

    Columns outside ``graph`` are ignored by ``fit_scm``, so the confounder-blind model is fitted
    on the identical rows -- it simply does not model ``Z``. The arms are the interventions
    ``{}`` (act as the logging policy did), ``do(A=0)`` and ``do(A=1)``.
    """
    scm = fit_scm(data, graph=graph, seed=seed)
    env = StructuralCausalBanditEnv(scm, graph, "Y", ["A"], {"A": [0, 1]}, n_mc=n_mc, seed=seed)
    return scm, CausalEnvWrapper(env, reward_node="Y")


def plan_in_model(wrapped: CausalEnvWrapper, *, steps: int, seed: int) -> int:
    """Run Thompson sampling *inside* the learned model; return the arm it converged on.

    This is the RL half: an agent choosing actions, receiving sampled rewards, and updating --
    except that every reward is imagined by the fitted SCM rather than paid for in the world.
    The shipped policy is the most-played arm.
    """
    env = wrapped.unwrapped
    assert isinstance(env, StructuralCausalBanditEnv)
    agent = NaiveThompsonSampling(len(env.arms), seed=seed)
    counts = np.zeros(len(env.arms), dtype=int)
    observation, _info = wrapped.reset(seed=seed)
    for _ in range(steps):
        action = agent.act(observation)
        observation, reward, _terminated, _truncated, _info = wrapped.step(action)
        agent.update(observation, action, float(reward))
        counts[action] += 1
        observation, _info = wrapped.reset()
    return int(np.argmax(counts))


def true_arm_value(bandit: SimpsonBandit, arm: Mapping[str, int], *, status_quo: float) -> float:
    """The arm's value in the REAL world: exact for ``do(A=a)``, ``E[Y]`` for the empty arm.

    ``status_quo`` is measured once, under the *confounded* logging policy, and reused as the empty
    arm's value in **both** regimes below. That reuse is sound only because of two `SimpsonBandit`
    facts: the treatment effect is constant across ``Z`` (0.1 in either stratum), and ``P(A=1)``
    averages to 0.5 under the confounded and the randomized policy alike -- so ``E[Y]`` is 0.45
    either way. On a world without both properties the "observe" arm would need its own per-regime
    measurement.
    """
    return status_quo if not arm else bandit.true_action_value(arm["A"])


def _evaluate(
    name: str,
    data: Mapping[str, np.ndarray],
    graph: CausalGraph,
    bandit: SimpsonBandit,
    *,
    status_quo: float,
    seeds: Sequence[int],
    steps: int,
    n_mc: int,
) -> ModelResult:
    scm, wrapped = world_model_env(data, graph, seed=0, n_mc=n_mc)
    env = wrapped.unwrapped
    assert isinstance(env, StructuralCausalBanditEnv)
    chosen: list[str] = []
    values: list[float] = []
    for seed in seeds:
        arm = env.arms[plan_in_model(wrapped, steps=steps, seed=seed)]
        chosen.append("observe" if not arm else f"do(A={arm['A']})")
        values.append(true_arm_value(bandit, arm, status_quo=status_quo))
    return ModelResult(
        name=name,
        reward_parents=tuple(wrapped.reward_parents),
        implied_conditional=_conditional(
            {key: value.numpy() for key, value in scm.see(40_000, seed=7).items()}
        ),
        arm_values=tuple(env.arm_values),
        chosen_arms=tuple(chosen),
        mean_true_value=float(np.mean(values)),
        mean_regret=float(bandit.optimal_value - np.mean(values)),
    )


def run_learned_scm_policy(
    *,
    n_logs: int = 20_000,
    steps: int = 2_000,
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
    n_mc: int = 20_000,
) -> tuple[RegimeResult, ...]:
    """Fit, plan and score both world models on a confounded log and a randomized one."""
    bandit = SimpsonBandit(seed=0)
    # E[Y] under the logging policy -- the value of the "observe" arm in the true world.
    status_quo = float(np.asarray(bandit.sample(200_000, seed=99)["Y"], dtype=float).mean())
    regimes = (
        ("confounded log", bandit.sample(n_logs, seed=0)),
        ("randomized log (control)", randomized_log(bandit, n_logs, seed=100)),
    )
    results: list[RegimeResult] = []
    for regime, data in regimes:
        models = tuple(
            _evaluate(
                name,
                data,
                graph,
                bandit,
                status_quo=status_quo,
                seeds=seeds,
                steps=steps,
                n_mc=n_mc,
            )
            for name, graph in (("causal", CAUSAL_GRAPH), ("confounder-blind", BLIND_GRAPH))
        )
        results.append(
            RegimeResult(regime=regime, empirical_conditional=_conditional(data), models=models)
        )
    return tuple(results)


def main(
    *,
    n_logs: int = 20_000,
    steps: int = 2_000,
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
    n_mc: int = 20_000,
) -> tuple[RegimeResult, ...]:
    results = run_learned_scm_policy(n_logs=n_logs, steps=steps, seeds=seeds, n_mc=n_mc)
    for result in results:
        print(result.summary())
        print()
    print(
        "Read it this way: on the confounded log both models fit E[Y|A] equally well, and only\n"
        "the one whose structure names Z as a parent of Y sends the agent to the right arm.\n"
        "On the randomized log they agree -- the causal structure buys nothing when the log was\n"
        "not confounded. The edge is the decision in the confounded regime, not accuracy in\n"
        "general, and this is a synthetic world with known ground truth, not a benchmark result."
    )
    return results


if __name__ == "__main__":
    main()
