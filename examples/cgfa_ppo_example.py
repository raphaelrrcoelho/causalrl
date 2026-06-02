"""CGFA-PPO example: wiring factored_advantage into stable-baselines3 PPO.

This script demonstrates how to integrate the
:func:`~causalrl.agents.factored_advantage.factored_advantage`
primitive from causalrl into a stock stable-baselines3 PPO run, reproducing the spirit of
CGFA-PPO (Causal Graph-Factored Advantage PPO) from:

    Cristiano da Costa Cunha, Ajmal Mian, Tim French, and Wei Liu (2026).
    "Causal Reinforcement Learning for Complex Card Games: A Magic: The Gathering
    Benchmark." arXiv:2605.06066.

The causal idea: the SCM defines which variables are direct parents of the reward node.
Instead of a single monolithic advantage ``A = V(s) - b``, we compute a per-factor
advantage for each SCM parent of the reward, then sum them.  This factored signal is a
strictly more informative credit-assignment target when the reward has multiple causal
parents.

In this minimal example we use the library's built-in StructuralCausalBanditEnv (the
confounded chain X1->X2->X3->Y with X1<->Y) wrapped in CausalEnvWrapper.  The one-step
bandit structure means the factored advantage reduces to the standard advantage here; the
value of this example is showing the wiring pattern, not demonstrating an empirical gain
(for that, use a multi-step environment where the causal graph has multiple reward-parent
factors contributing at different time scales).

Requirements (optional extras, NOT in causalrl core deps):
    pip install "causalrl[examples]"
    # or: pip install stable-baselines3>=2.3 torch>=2.5

If stable-baselines3 is not installed the script exits with a clear message.
"""

from __future__ import annotations

import sys

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

from causalrl.agents.factored_advantage import FactoredAdvantageConfig, factored_advantage
from causalrl.envs.suite.scbandit import make_confounded_chain_env
from causalrl.envs.wrapper import CausalEnvWrapper

# ---------------------------------------------------------------------------
# Callback: inject CGFA-corrected advantage into the rollout buffer.
# ---------------------------------------------------------------------------


class CGFAAdvantageCallback(BaseCallback):
    """SB3 callback that replaces the computed advantages with CGFA-factored advantages.

    This callback fires after ``_compute_returns_and_advantage`` has run (i.e. on every
    ``collect_rollouts`` call) and re-computes advantages using per-factor value estimates
    produced by the critic for each SCM parent of the reward.

    In this minimal example we do not train separate per-factor critics; instead we
    illustrate the API wiring by using the *same* value head for each factor (equivalent
    to the standard advantage scaled by the number of factors under ``aggregation="mean"``).
    A production implementation would maintain one critic head per factor node.

    Parameters
    ----------
    config:
        The :class:`~causalrl.agents.factored_advantage.FactoredAdvantageConfig` carrying
        the SCM reward-parent node names.
    """

    def __init__(self, config: FactoredAdvantageConfig, verbose: int = 0) -> None:
        super().__init__(verbose=verbose)
        self.config = config

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        """After each rollout: replace advantages with CGFA-factored advantages."""
        assert self.model is not None
        buf = self.model.rollout_buffer  # type: ignore[attr-defined]
        if not hasattr(buf, "advantages"):
            return

        # Retrieve the monolithic advantages and value estimates already computed by SB3.
        T = buf.pos if not buf.full else buf.buffer_size
        adv = buf.advantages.reshape(-1)[:T]  # (T,)
        # In a full CGFA-PPO, you'd have K value heads; here we use the same value for
        # each factor to show the wiring without requiring K separate networks.
        K = len(self.config.factor_nodes)
        # Reconstruct V from returns and advantages (V ≈ returns - advantages)
        ret = buf.returns.reshape(-1)[:T]
        V_mono = ret - adv  # scalar baseline per step (T,)

        # Build the per-factor value matrix (T, K).  In this demo all factors share the
        # same value estimate; swap each column for a dedicated critic head in production.
        factor_values = np.tile(V_mono.reshape(T, 1), (1, K))

        # Compute CGFA advantages using the library primitive.
        cgfa_adv = factored_advantage(factor_values, V_mono, config=self.config)

        # Write back (SB3 stores advantages as float32 numpy arrays).
        buf.advantages.reshape(-1)[:T] = cgfa_adv.astype(np.float32)

        if self.verbose >= 1:
            print(
                f"[CGFAAdvantageCallback] rollout_end: T={T}, K={K}, adv_mean={cgfa_adv.mean():.4f}"
            )


# ---------------------------------------------------------------------------
# Main: wire everything together.
# ---------------------------------------------------------------------------


def main() -> None:
    print("=== CGFA-PPO wiring example ===")
    print(f"stable-baselines3 version: {sb3.__version__}")

    # 1. Build the causal env wrapped with CausalEnvWrapper so we get reward_parents.
    inner_env = make_confounded_chain_env(n_mc=500, seed=0)
    env = CausalEnvWrapper(inner_env, reward_node="Y")

    print(f"Reward node: {env.reward_node}")
    print(f"SCM reward parents: {env.reward_parents}")

    # 2. Build the FactoredAdvantageConfig from the SCM parents of the reward.
    config = FactoredAdvantageConfig(
        factor_nodes=env.reward_parents,
        aggregation="sum",
    )
    print(f"FactoredAdvantageConfig: {config.factor_nodes}, aggregation={config.aggregation}")

    # 3. Demonstrate the factored_advantage primitive directly (the causal core).
    K = len(config.factor_nodes)
    T = 4
    rng = np.random.default_rng(42)
    dummy_factor_values = rng.standard_normal((T, K)).astype(np.float64)
    dummy_baselines = rng.standard_normal(T).astype(np.float64)
    adv = factored_advantage(dummy_factor_values, dummy_baselines, config=config)
    print(f"\nDummy factored_advantage demo (T={T}, K={K}):")
    print(f"  factor_values:\n{dummy_factor_values}")
    print(f"  baselines: {dummy_baselines}")
    print(f"  factored advantage: {adv}")

    # 4. Train PPO with the CGFA callback (short run — for illustration only).
    callback = CGFAAdvantageCallback(config, verbose=1)
    model = sb3.PPO("MultiInputPolicy", env, verbose=0, n_steps=32, batch_size=16, n_epochs=2)
    print("\nTraining PPO with CGFAAdvantageCallback (100 steps for illustration)...")
    model.learn(total_timesteps=100, callback=callback)

    # 5. Evaluate.
    obs, _ = env.reset()
    total_reward = 0.0
    for _ in range(20):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(int(action))
        total_reward += float(reward)
        if terminated or truncated:
            obs, _ = env.reset()
    print(f"\nMean reward over 20 steps (random policy warmup): {total_reward / 20:.4f}")

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
