"""Pure compute for the three see-and-touch demos — no plotting, no widgets.

Both the live notebook and the static HTML builder import this module, so the numbers behind
every figure come from one place and cannot drift. Everything here uses only the public
``causalrl`` surface plus numpy.
"""

from __future__ import annotations

import numpy as np

from causalrl import pomis
from causalrl.agents.bandits import CausalThompsonSampling, NaiveThompsonSampling
from causalrl.agents.scbandit import (
    BruteForceInterventionTS,
    FixedSetThompsonSampling,
    POMISThompsonSampling,
)
from causalrl.envs.suite.counterfactual_bandit import (
    build_counterfactual_scm,
    make_counterfactual_bandit_env,
)
from causalrl.envs.suite.mabuc import MABUCEnv
from causalrl.envs.suite.scbandit import make_confounded_chain_env

SNAP_STEPS = [0, 2, 5, 10, 25, 50, 100, 250, 500, 1000, 2000, 4000, 8000]


# --------------------------------------------------------------------------------------
# Shared bandit mechanics (used by the live "pull once" buttons too)
# --------------------------------------------------------------------------------------
def beta_posteriors(agent, ctx):
    """(alpha, beta) over arms for `agent` at context `ctx`.

    The causal agent keeps one row per (intuition, arm); the naive agent keeps a single row
    it reuses for every context. This is exactly the state `agent.act()` samples from.
    """
    if getattr(agent, "n_contexts", None) is not None:
        return agent._alpha[ctx].copy(), agent._beta[ctx].copy()
    return agent._alpha.copy(), agent._beta.copy()


def transparent_pull(agent, obs, env, rng):
    """One episode with the Thompson draw made visible (identical to `agent.act` internally)."""
    ctx = int(obs["intuition"])
    a_row, b_row = beta_posteriors(agent, ctx)
    thetas = rng.beta(a_row, b_row)
    action = int(np.argmax(thetas))
    _, r, _, _, _ = env.step(action)
    agent.update(obs, action, r)
    return ctx, thetas, action, float(r)


# --------------------------------------------------------------------------------------
# Demo 1 — MABUC belief snapshots
# --------------------------------------------------------------------------------------
def _train_snapshots(agent, seed=1, steps=8000, snap_at=SNAP_STEPS):
    env = MABUCEnv(seed=seed)
    obs, _ = env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    snaps, log = {}, []
    for t in range(steps + 1):
        if t in snap_at:
            snaps[t] = (np.array(agent._alpha, float), np.array(agent._beta, float))
        if t == steps:
            break
        ctx, thetas, action, r = transparent_pull(agent, obs, env, rng)
        if t < 6:
            log.append((t, ctx, thetas.round(2).tolist(), action, r))
        obs, _ = env.reset()
    return snaps, log


def mabuc_snapshots():
    """Return (snap_steps, causal_snaps, naive_snaps, first_decisions, agents)."""
    causal = CausalThompsonSampling(n_arms=2, n_contexts=2, seed=0)
    naive = NaiveThompsonSampling(n_arms=2, seed=0)
    causal_snaps, log = _train_snapshots(causal)
    naive_snaps, _ = _train_snapshots(naive)
    return SNAP_STEPS, causal_snaps, naive_snaps, log, (causal, naive)


# --------------------------------------------------------------------------------------
# Demo 2 — POMIS lever values + learning scoreboard
# --------------------------------------------------------------------------------------
def _arm_label(arm):
    return "∅ (just observe)" if not arm else "do(" + ", ".join(f"{k}={v}" for k, v in arm.items()) + ")"


def pomis_data():
    """Return a dict with every lever's label/value/POMIS-membership and the learning curves."""
    env = make_confounded_chain_env(seed=1)
    pomis_sets = pomis(env.graph, "Y")
    keys = {frozenset(s) for s in pomis_sets}
    labels = [_arm_label(a) for a in env.arms]
    values = [float(v) for v in env.arm_values]
    in_pomis = [frozenset(a.keys()) in keys for a in env.arms]

    def tail(agent, steps=8000, seed=1):
        obs, _ = env.reset(seed=seed)
        rewards = []
        for _ in range(steps):
            a = agent.act(obs)
            nobs, r, _, _, _ = env.step(a)
            agent.update(obs, a, r)
            rewards.append(r)
            obs = nobs
        x = np.asarray(rewards, float)
        return np.cumsum(x) / (np.arange(len(x)) + 1)

    curves = {
        "POMIS (2 arms)": tail(POMISThompsonSampling(env.graph, env.reward, env.arms,
                                                     seed=0, manipulable=env.manipulable)),
        "brute force (27 arms)": tail(BruteForceInterventionTS(env.arms, seed=0), seed=2),
        "naive do(X3)": tail(FixedSetThompsonSampling(env.arms, {"X3"}, seed=0), seed=3),
    }
    return {
        "labels": labels, "values": values, "in_pomis": in_pomis,
        "pomis_sets": pomis_sets, "optimal": float(env.optimal_value), "curves": curves,
        "n_arms": len(env.arms),
    }


# --------------------------------------------------------------------------------------
# Demo 3 — counterfactual decision table + a live round
# --------------------------------------------------------------------------------------
def counterfactual_agent_and_table():
    from causalrl import CounterfactualOptimalPolicy

    agent = CounterfactualOptimalPolicy(
        build_counterfactual_scm(), outcome="Y", action_node="X", intent_node="I",
        arms=[0, 1, 2], intents=[0, 1, 2], seed=0,
    )
    table = agent.decision_table
    M = np.array([[table[i][a] for a in (0, 1, 2)] for i in (0, 1, 2)])
    return agent, M


def make_cf_env(seed=3):
    return make_counterfactual_bandit_env(seed=seed)
