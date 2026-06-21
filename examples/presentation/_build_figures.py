"""Run the three Pearl-hierarchy demos and render the presentation figures.

This is the offline figure builder behind ``causal_rl_presentation.ipynb``: it executes
exactly the same code the notebook runs live, so the rendered PNGs and the live cells can
never disagree. Run with::

    uv run python examples/presentation/_build_figures.py

All three demos are tabular and finish in a few seconds.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

FIG_DIR = Path(__file__).parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

BLUE, RED, GREY = "#1f77b4", "#d62728", "#999999"


# --------------------------------------------------------------------------------------
# Demo 1 — MABUC (L1 see vs L2 do): identical do() means, different counterfactual value
# --------------------------------------------------------------------------------------
def demo_mabuc() -> None:
    from causalrl.agents.bandits import CausalThompsonSampling, NaiveThompsonSampling
    from causalrl.envs.suite.mabuc import MABUCEnv, build_mabuc_scm

    scm = build_mabuc_scm()
    do0 = scm.do({"X": 0.0}).see(20000, seed=0)["Y"].mean().item()
    do1 = scm.do({"X": 1.0}).see(20000, seed=1)["Y"].mean().item()
    print(f"[MABUC] E[Y|do(X=0)]={do0:.3f}  E[Y|do(X=1)]={do1:.3f}  (indistinguishable)")

    def run(agent, n=8000, seed=1):
        env = MABUCEnv(seed=seed)
        obs, _ = env.reset(seed=seed)
        rewards = []
        for _ in range(n):
            a = agent.act(obs)
            _, r, _, _, _ = env.step(a)
            agent.update(obs, a, r)
            obs, _ = env.reset()
            rewards.append(r)
        return np.asarray(rewards, dtype=float)

    causal = run(CausalThompsonSampling(2, 2, seed=0))
    naive = run(NaiveThompsonSampling(2, seed=0))
    opt = 0.75
    print(f"[MABUC] causal avg={causal.mean():.3f}  naive avg={naive.mean():.3f}")

    def running_mean(x):
        return np.cumsum(x) / (np.arange(len(x)) + 1)

    def cum_regret(x):
        return np.cumsum(opt - x)

    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(15, 4.2))

    ax0.bar(["do(X=0)", "do(X=1)"], [do0, do1], color=[GREY, GREY], width=0.55)
    ax0.axhline(0.5, ls="--", c="k", lw=0.8)
    ax0.set_ylim(0, 1)
    ax0.set_ylabel("E[Y | do(X=a)]")
    ax0.set_title("L2 interventional means are identical\n(no do()-agent can tell the arms apart)")

    ax1.plot(running_mean(causal), color=BLUE, label="Causal TS (conditions on intuition)")
    ax1.plot(running_mean(naive), color=RED, label="Naive TS (ignores intuition)")
    ax1.axhline(opt, ls="--", c="k", lw=0.8, label="optimal = 0.75")
    ax1.set_ylim(0.45, 0.8)
    ax1.set_xlabel("step")
    ax1.set_ylabel("running avg reward")
    ax1.set_title("Reward per step")
    ax1.legend(loc="lower right", fontsize=8)

    ax2.plot(cum_regret(causal), color=BLUE, label=f"Causal (final {cum_regret(causal)[-1]:.0f})")
    ax2.plot(cum_regret(naive), color=RED, label=f"Naive (final {cum_regret(naive)[-1]:.0f})")
    ax2.set_xlabel("step")
    ax2.set_ylabel("cumulative regret")
    ax2.set_title("Cumulative regret vs the 0.75 oracle")
    ax2.legend(loc="upper left", fontsize=8)

    fig.suptitle(
        "Demo 1 — MABUC: same do()-means, but conditioning on the confounder proxy wins",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIG_DIR / "demo1_mabuc.png", dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------------------
# Demo 2 — POMIS (L2 "where to intervene?"): prune 27 candidate arms to 2 optimal sets
# --------------------------------------------------------------------------------------
def demo_pomis() -> None:
    from causalrl import pomis
    from causalrl.agents.scbandit import (
        BruteForceInterventionTS,
        FixedSetThompsonSampling,
        POMISThompsonSampling,
    )
    from causalrl.envs.suite.scbandit import make_confounded_chain_env

    env = make_confounded_chain_env(seed=1)
    sets = pomis(env.graph, "Y")
    n_arms = env.action_space.n
    print(f"[POMIS] candidate arms={n_arms}  POMIS sets={sets}  optimal={env.optimal_value:.3f}")

    def run(agent, steps=8000, seed=1):
        obs, _ = env.reset(seed=seed)
        rewards = []
        for _ in range(steps):
            a = agent.act(obs)
            nobs, r, _, _, _ = env.step(a)
            agent.update(obs, a, r)
            rewards.append(r)
            obs = nobs
        return np.asarray(rewards, dtype=float)

    pomis_r = run(
        POMISThompsonSampling(env.graph, env.reward, env.arms, seed=0, manipulable=env.manipulable)
    )
    brute_r = run(BruteForceInterventionTS(env.arms, seed=0), seed=2)
    naive_r = run(FixedSetThompsonSampling(env.arms, {"X3"}, seed=0), seed=3)

    def running_mean(x):
        return np.cumsum(x) / (np.arange(len(x)) + 1)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 4.4))

    ax0.bar(["brute force\n(all arms)", "POMIS\n(provably optimal)"], [n_arms, len(sets)],
            color=[GREY, BLUE], width=0.5)
    ax0.set_ylabel("# candidate interventions")
    ax0.set_title("POMIS prunes the search space")
    for i, v in enumerate([n_arms, len(sets)]):
        ax0.text(i, v + 0.4, str(v), ha="center", fontweight="bold")

    ax1.plot(running_mean(pomis_r), color=BLUE, label="POMIS TS (∅, {X3})")
    ax1.plot(running_mean(brute_r), color=GREY, label="Brute force (27 arms)")
    ax1.plot(running_mean(naive_r), color=RED, label="Naive do(X3) only")
    ax1.axhline(env.optimal_value, ls="--", c="k", lw=0.8, label=f"optimal={env.optimal_value:.2f}")
    ax1.set_xlabel("step")
    ax1.set_ylabel("running avg reward")
    ax1.set_title("Observing (∅) beats every fixed intervention")
    ax1.legend(loc="lower right", fontsize=8)

    fig.suptitle(
        "Demo 2 — POMIS: the graph tells you the few levers worth pulling (X1→X2→X3→Y, X1↔Y)",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIG_DIR / "demo2_pomis.png", dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------------------
# Demo 3 — Counterfactual policy (L3): act on E[Y_do(a) | intent] on a 3-arm bandit
# --------------------------------------------------------------------------------------
def demo_counterfactual() -> None:
    from causalrl import CounterfactualOptimalPolicy
    from causalrl.agents.bandits import NaiveThompsonSampling
    from causalrl.envs.suite.counterfactual_bandit import (
        build_counterfactual_scm,
        make_counterfactual_bandit_env,
    )

    scm = build_counterfactual_scm()
    agent = CounterfactualOptimalPolicy(
        scm, outcome="Y", action_node="X", intent_node="I",
        arms=[0, 1, 2], intents=[0, 1, 2], seed=0,
    )
    table = agent.decision_table  # {intent: {arm: E[Y_do(arm)|intent]}}
    M = np.array([[table[i][a] for a in [0, 1, 2]] for i in [0, 1, 2]])
    print(f"[CF] best fixed do(a) means ≈ {M.mean(axis=0).round(3)} (all ~0.37)")

    def run(agent, n=8000, seed=1):
        env = make_counterfactual_bandit_env(seed=seed)
        obs, _ = env.reset(seed=seed)
        rewards = []
        for _ in range(n):
            a = agent.act(obs)
            _, r, _, _, _ = env.step(a)
            agent.update(obs, a, r)
            obs, _ = env.reset()
            rewards.append(r)
        return np.asarray(rewards, dtype=float)

    cf = run(agent)
    naive = run(NaiveThompsonSampling(3, seed=0))
    print(f"[CF] counterfactual avg={cf.mean():.3f}  naive avg={naive.mean():.3f}")

    def running_mean(x):
        return np.cumsum(x) / (np.arange(len(x)) + 1)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 4.4))

    im = ax0.imshow(M, cmap="viridis", vmin=0.1, vmax=0.85)
    ax0.set_xticks([0, 1, 2], ["do(X=0)", "do(X=1)", "do(X=2)"])
    ax0.set_yticks([0, 1, 2], ["intent=0", "intent=1", "intent=2"])
    for i in range(3):
        for j in range(3):
            ax0.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                     color="white" if M[i, j] < 0.5 else "black", fontweight="bold")
    ax0.set_title("L3 decision table  E[Y_do(a) | intent]\n(diagonal = play your intuition)")
    fig.colorbar(im, ax=ax0, fraction=0.046, pad=0.04)

    ax1.plot(running_mean(cf), color=BLUE, label=f"Counterfactual policy ({cf.mean():.2f})")
    ax1.plot(running_mean(naive), color=RED, label=f"Naive TS / best fixed arm ({naive.mean():.2f})")
    ax1.axhline(0.367, ls="--", c=GREY, lw=0.8, label="best fixed do(a) ≈ 0.37")
    ax1.axhline(0.8, ls="--", c="k", lw=0.8, label="optimal = 0.80")
    ax1.set_ylim(0.2, 0.9)
    ax1.set_xlabel("step")
    ax1.set_ylabel("running avg reward")
    ax1.set_title("Conditioning on intent recovers the optimum")
    ax1.legend(loc="center right", fontsize=8)

    fig.suptitle(
        "Demo 3 — Counterfactual policy: every fixed arm averages 0.37, intent-conditioning gets 0.80",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIG_DIR / "demo3_counterfactual.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    demo_mabuc()
    demo_pomis()
    demo_counterfactual()
    print(f"\nFigures written to {FIG_DIR}/")
