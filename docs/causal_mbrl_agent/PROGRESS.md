# Causal MBRL Agent — Progress / Resume

**Program:** a general, domain-neutral causal model-based RL agent (discover → robustly plan →
certify → transport). See `DESIGN.md` (M0–M3) and `plans/2026-07-19-m0-kill-gate.md`.

**Branch:** `causal-mbrl-agent`.

## Status (2026-07-19)

**M0 apparatus implemented (Tasks 1–4), verification pending CI.** Local runs on the `/mnt/c` WSL2
mount are impractically slow (multi-minute imports), so **CI is the verifier** — nothing is claimed
green until the GitHub Actions run passes.

| Task | Deliverable | State |
|---|---|---|
| T1 | `ConfoundedContextualBandit` oracle env (`envs/suite/confounded_context.py`) + tests | implemented |
| T2 | `CertifiedPolicyAgent` certify-gated agent (`agents/mbrl.py`) + tests | implemented |
| T3 | `run_m0_kill_gate` harness (`eval/mbrl_probe.py`) + tests | implemented |
| T4 | public-API exports + this doc | implemented |

## The M0 kill-gate verdict — NO-GO (2026-07-19, 10 seeds)

```
causal_source = 0.4500   naive_source = 0.4500
causal_shifted = 0.4500  naive_shifted = 0.4500   gap = 0.0000
```

The certify-gated agent did NOT beat naive — both land at 0.45.

**Diagnosis (two layers):**
1. *Env symmetry:* the reward is symmetric (q(0,0)=q(1,1)=0.55, q(0,1)=q(1,0)=0.35), so every
   constant/mixed policy averages 0.45; only the context-dependent optimal [0,1]=0.55 differs, and
   neither agent finds it.
2. *Deeper — the real lesson:* the **certify-gated planner is a safety mechanism whose ceiling is
   the behavior policy.** Under strong confounding it cannot certify the true optimum, so it abstains
   to a (noisy) behavior default rather than recovering it. Naive is fooled to [1,1]=0.45; abstention
   is also ≈0.45 → no gap. This agent avoids harm; it does not *perform* better — which is exactly
   the "only a certificate" limitation the whole program set out to move past.

**Fork:**
- (a) Recalibrate the env once so the confounder-fooled policy is strictly *worse* than behavior →
  causal-via-abstention beats naive, but only a modest "don't ship a confounded loser" win.
- (b) **Swap the planner** to one that actively *optimizes* under confounding (DOVI /
  `msm_policy_value_bounds` value-maximizer) — a functioning agent, not a safety gate. Aligns with
  the program's actual goal. **← recommended.**
- (c) Accept the honest negative: the certify-gated agent's ceiling is the behavior policy.

Reproduce:
```bash
python -c "from causalrl import run_m0_kill_gate; import json; \
r = run_m0_kill_gate(seeds=tuple(range(10))); \
print(json.dumps({k:{'mean':v.mean,'lo':v.ci95_low,'hi':v.ci95_high} for k,v in r.items()}, indent=2))"
```

## Resume here

1. Watch CI on `causal-mbrl-agent`; fix forward any red (likely candidates: the `gamma_max` values
   in `tests/test_mbrl_agents.py`, which encode a guess about when `certify_policy` certifies — flip
   between low/high if the certify/abstain assertions fail).
2. On green, run the verdict above; record the four means + CIs here.
3. Decide GO → M1 (full discover→plan→certify loop on the DTR/medicine instance) or iterate.

## Notes / decisions

- Discovery is deferred to M1 (the bandit's action×context interaction has no marginal main-effect
  for `discover_interventional`'s invariance test; structure is *given* in M0). See DESIGN.md §8.
- Causal agent is **certify-gated** (`certify_policy`), not Manski-greedy — Manski lower-bound greedy
  does not correct a backdoor `A ← U → Y`.
- Correlational/model-free baseline reuses `NaiveOffline` (one-step bandit collapses them); a distinct
  model-free/world-model baseline arrives with M1's multi-step env.
