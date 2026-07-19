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

## The M0 kill-gate verdict — PENDING

Run once CI is green:
```bash
python -c "from causalrl import run_m0_kill_gate; import json; \
r = run_m0_kill_gate(seeds=tuple(range(10))); \
print(json.dumps({k:{'mean':v.mean,'lo':v.ci95_low,'hi':v.ci95_high} for k,v in r.items()}, indent=2))"
```
Read: **GO** if `causal_shifted.mean > naive_shifted.mean` (CIs roughly non-overlapping) and
`causal_source ≥ naive_source` → proceed to M1. **NO-GO / iterate** otherwise: calibrate the knobs
(`gamma`, reward coefficients, `gamma_max`) once; if still no gap, swap the certify-gated rule for a
DOVI / `msm_policy_value_bounds` planner before declaring the thesis dead.

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
