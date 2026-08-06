"""CGFA-PPO example: a real K-head factored critic driving stable-baselines3 PPO.

This script runs Causal Graph-Factored Advantage PPO from:

    Cristiano da Costa Cunha, Ajmal Mian, Tim French, and Wei Liu (2026).
    "Causal Reinforcement Learning for Complex Card Games: A Magic: The Gathering
    Benchmark." arXiv:2605.06066.

on top of a stock stable-baselines3 PPO run.  The causal idea: the SCM says which variables
are direct parents of the reward node, so instead of one monolithic advantage the critic
keeps one value head **per causal parent**, trained against that parent's own return, and the
policy update rides a gated mixture of the scalar and the factor-aligned advantages.

What this example wires together
--------------------------------
1. :class:`~causalrl.agents.cgfa_critic.FactoredCritic` — ``K`` value heads on a shared trunk,
   learnable mixture logits ``beta``, and the state-conditional residual gate ``g(s)``.
2. :func:`~causalrl.agents.factored_advantage.factor_rewards` /
   :func:`~causalrl.agents.factored_advantage.factor_gae` — the per-factor reward
   ``phi_k(s') - phi_k(s)`` and the per-factor returns and advantages it accumulates into.
3. :func:`~causalrl.agents.factored_advantage.blend_advantages` — Eq. 11, the advantage that
   is written back into the SB3 rollout buffer.
4. The intervention-calibration loss, fed with ``eps_k``, the SCM's *predicted* per-factor
   change under ``do(arm)``, obtained by re-evaluating the structural equations rather than
   by simulating the environment (arXiv:2605.06066 §B).

Honest scope
------------
The environment is the library's one-step ``StructuralCausalBanditEnv`` (the confounded chain
X1->X2->X3->Y with X1<->Y).  Two consequences, stated plainly:

* Its observation is a constant, so there is no state for a state-conditional critic to
  condition on.  We feed the **chosen arm's one-hot** as the critic input instead, which
  makes ``V_k`` an arm-conditional per-factor value — informative here, but not the
  ``s_t``-conditional critic of the paper.
* Every episode is one step, so the per-factor return collapses to the per-factor reward and
  ``gamma`` does no work.  A multi-step environment is where the per-factor GAE earns its
  keep.

Requirements (optional extras, NOT in causalrl core deps):
    pip install "causalrl[examples]"
    # or: pip install stable-baselines3>=2.3 torch>=2.5

If stable-baselines3 is not installed the script exits with a clear message.
"""

from __future__ import annotations

import sys
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Optional SB3 import — degrade gracefully if not installed.
# ---------------------------------------------------------------------------
try:
    import stable_baselines3 as sb3
    from stable_baselines3.common.callbacks import BaseCallback
except ModuleNotFoundError:
    print(
        "stable-baselines3 is not installed.\n"
        "This example requires the 'examples' optional extra:\n"
        "    pip install 'causalrl[examples]'\n"
        "or:\n"
        "    pip install stable-baselines3>=2.3 torch>=2.5",
        file=sys.stderr,
    )
    sys.exit(0)

from causalrl.agents.cgfa_critic import CGFACriticConfig, FactoredCritic
from causalrl.agents.factored_advantage import FactoredAdvantageConfig
from causalrl.envs.suite.scbandit import StructuralCausalBanditEnv, make_confounded_chain_env
from causalrl.envs.wrapper import CausalEnvWrapper

# ---------------------------------------------------------------------------
# The CGFA wrapper's per-step quantities (arXiv:2605.06066 §E.1).
# ---------------------------------------------------------------------------


class FactorTracer:
    """Publishes ``phi(s)`` and the SCM-predicted change ``eps`` for each arm of the bandit.

    The paper's benchmark wrapper emits both per environment step.  Here the same two
    quantities are read straight off the SCM: ``phi`` by sampling the mutilated model (what
    actually happened), ``eps`` by averaging it (what the structural equations predict).
    The mutilated SCM for every arm is built once, up front.
    """

    def __init__(
        self, env: StructuralCausalBanditEnv, factor_nodes: list[str], *, n_mc: int = 64
    ) -> None:
        self.factor_nodes = factor_nodes
        self._models = [
            env.scm.do({k: float(v) for k, v in arm.items()}) if arm else env.scm
            for arm in env.arms
        ]
        # eps_k for arm a: E[phi_k | do(a)] - phi_k(s_t), with phi(s_t) = 0 for the one-step
        # bandit's featureless pre-action state.
        self._predicted = np.array(
            [
                [float(model.see(n_mc, seed=7 + i)[node].mean()) for node in factor_nodes]
                for i, model in enumerate(self._models)
            ]
        )

    def realised(self, actions: np.ndarray, *, seed: int) -> np.ndarray:
        """``phi(s_{t+1})`` for each step of the rollout, shape ``(T, K)``."""
        rng = np.random.default_rng(seed)
        rows = []
        for a in actions:
            draw = self._models[int(a)].see(1, seed=int(rng.integers(0, 2**31)))
            rows.append([float(draw[node].reshape(-1)[0]) for node in self.factor_nodes])
        return np.asarray(rows, dtype=np.float64)

    def predicted(self, actions: np.ndarray) -> np.ndarray:
        """``eps_{k,t}``, the SCM-predicted per-factor change, shape ``(T, K)``."""
        return self._predicted[np.asarray(actions, dtype=int)]


# ---------------------------------------------------------------------------
# Callback: train the K heads, then inject the blended advantage.
# ---------------------------------------------------------------------------


class CGFAAdvantageCallback(BaseCallback):
    """SB3 callback implementing Algorithm 1 lines 6-13 plus the critic-side update.

    On every ``collect_rollouts`` boundary it reads the rollout's actions, asks the tracer for
    ``phi`` and ``eps``, runs the ``K``-head critic to get per-factor returns and advantages,
    blends them into SB3's own advantage with the learned gate and mixture weights, writes the
    result back into the rollout buffer, and takes the per-factor / calibration optimiser
    steps.  SB3 then runs its clipped surrogate on the blended advantage.

    Parameters
    ----------
    critic:
        The :class:`~causalrl.agents.cgfa_critic.FactoredCritic` whose heads correspond, in
        order, to the SCM parents of the reward node.
    tracer:
        Supplies ``phi(s_{t+1})`` and the SCM-predicted ``eps`` per rollout step.
    n_arms:
        Size of the discrete action space; the critic input is the arm one-hot.
    epochs:
        Critic epochs per rollout.
    """

    def __init__(
        self,
        critic: FactoredCritic,
        tracer: FactorTracer,
        n_arms: int,
        *,
        epochs: int = 4,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        self.critic = critic
        self.tracer = tracer
        self.n_arms = n_arms
        self.epochs = epochs
        self._rollout = 0

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        assert self.model is not None
        buf: Any = self.model.rollout_buffer  # type: ignore[attr-defined]
        if not hasattr(buf, "advantages"):
            return

        n = buf.buffer_size if buf.full else buf.pos
        if n == 0:
            return
        self._rollout += 1

        actions = buf.actions.reshape(-1)[:n].astype(int)
        scalar_adv = buf.advantages.reshape(-1)[:n].astype(np.float64)

        # phi(s_t) = 0 for the featureless pre-action state, so the per-factor reward of
        # §E.1 is exactly the realised parent value under the chosen arm.
        factor_rewards = self.tracer.realised(actions, seed=self._rollout)
        scm_effects = self.tracer.predicted(actions)
        observations = np.eye(self.n_arms, dtype=np.float64)[actions]

        # Algorithm 1 lines 11-13: per-factor GAE, then the Eq. 11 residual blend. Every
        # episode terminates immediately, so nothing bootstraps across steps.
        bundle = self.critic.advantages(
            observations,
            factor_rewards,
            scalar_adv,
            gamma=0.99,
            dones=np.ones(n, dtype=bool),
        )
        buf.advantages.reshape(-1)[:n] = bundle.used.astype(np.float32)

        # Algorithm 1 lines 14-22, critic side: Eq. 9 on the per-factor returns and Eq. 12
        # against the SCM's predicted intervention effect.
        stats = self.critic.update(
            observations,
            bundle.returns,
            scm_effects=scm_effects,
            epochs=self.epochs,
        )
        if self.verbose >= 1:
            names = self.critic.factor_nodes
            print(
                f"[CGFA] rollout {self._rollout}: T={n} K={self.critic.n_factors} "
                f"L_factor={stats.factor:.4f} L_cal={stats.calibration:.4f} "
                f"gate={stats.gate_mean:.3f}"
            )
            for k, node in enumerate(names):
                print(
                    f"        {node:>4}: w={stats.mixture_weights[k]:.3f} "
                    f"corr(A_k, eps_k)={stats.factor_correlation[k]:+.3f} "
                    f"credit={stats.credit_share[k]:.3f}"
                )


# ---------------------------------------------------------------------------
# Main: wire everything together.
# ---------------------------------------------------------------------------


def main(total_timesteps: int = 100, eval_steps: int = 20, n_mc: int = 500) -> None:
    """Run the CGFA-PPO demonstration.

    Parameters
    ----------
    total_timesteps:
        Number of environment steps passed to ``model.learn``.  Set to a very small value
        (e.g. 32) for smoke tests; 100 is sufficient for illustration.
    eval_steps:
        Number of evaluation steps taken after training.
    n_mc:
        Monte-Carlo samples used to estimate arm values in the bandit env.  Reduce for
        faster construction in smoke tests.
    """
    print("=== CGFA-PPO example ===")
    print(f"stable-baselines3 version: {sb3.__version__}")

    # 1. Build the causal env wrapped with CausalEnvWrapper so we get reward_parents.
    inner_env = make_confounded_chain_env(n_mc=n_mc, seed=0)
    env = CausalEnvWrapper(inner_env, reward_node="Y")

    print(f"Reward node: {env.reward_node}")
    print(f"SCM reward parents (the K causal factors): {env.reward_parents}")

    # 2. One value head per SCM parent of the reward. FactoredAdvantageConfig carries the
    #    same node list for any framework-agnostic call site.
    factor_config = FactoredAdvantageConfig(factor_nodes=env.reward_parents)
    n_arms = int(inner_env.action_space.n)
    critic = FactoredCritic(
        obs_dim=n_arms,
        factor_nodes=env.reward_parents,
        config=CGFACriticConfig(hidden=(64, 64), learning_rate=1e-2),
        seed=0,
    )
    print(
        f"FactoredCritic: {critic.n_factors} heads over {factor_config.factor_nodes}, "
        f"initial mixture w={np.round(critic.mixture_weights(), 3)}, "
        f"initial gate g={critic.config.gate_init}"
    )

    tracer = FactorTracer(inner_env, env.reward_parents, n_mc=64)

    # 3. Train PPO with the CGFA callback (short run — for illustration only).
    callback = CGFAAdvantageCallback(critic, tracer, n_arms, verbose=1)
    n_steps = max(16, min(32, total_timesteps))
    batch_size = max(8, n_steps // 2)
    model = sb3.PPO(
        "MultiInputPolicy",
        env,
        verbose=0,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=2,
    )
    print(f"\nTraining PPO on CGFA-blended advantages ({total_timesteps} steps)...")
    model.learn(total_timesteps=total_timesteps, callback=callback)

    # 4. The heads are different functions: show what each learned per arm.
    print("\nPer-factor values V_k(arm) for the first few arms:")
    probes = np.eye(n_arms, dtype=np.float64)[: min(4, n_arms)]
    _, per_factor = critic.values(probes)
    for i, row in enumerate(per_factor):
        arm = inner_env.arms[i] or "observe"
        cells = ", ".join(f"{n}={v:+.3f}" for n, v in zip(critic.factor_nodes, row, strict=True))
        print(f"  arm {i} ({arm}): {cells}")

    # 5. Evaluate.
    obs, _ = env.reset()
    total_reward = 0.0
    for _ in range(eval_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(int(action))
        total_reward += float(reward)
        if terminated or truncated:
            obs, _ = env.reset()
    print(f"\nMean reward over {eval_steps} steps: {total_reward / eval_steps:.4f}")

    # 6. Show the do-intervention handle exposed by the wrapper.
    mutilated_scm = env.intervene("X1", 1.0)
    samples = mutilated_scm.see(200, seed=0)
    print(
        f"\nE[Y | do(X1=1)] via SCM query: {float(samples['Y'].mean()):.4f} "
        f"(ground-truth arm value ≈ {inner_env.arm_values[1]:.4f} for do(X1=1) arm)"
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
