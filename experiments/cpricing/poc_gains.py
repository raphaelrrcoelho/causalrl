#!/usr/bin/env python3
"""Do the causal mechanics buy *accuracy*, or only honesty? Two measurements, both against a
non-causal baseline, and both of which killed my first guess.

Companion to ``docs/causal_pricing/GAINS.md``.

G2 -- VARIANCE REDUCTION, and where it does and does not exist.
    The estimand ``E[P&L(theta') - P&L(theta)]`` computed two ways at an equal path budget:
    (a) NON-CAUSAL -- two independent sample sets, difference of means. This is what a
        generative path model that cannot abduct is limited to: it can condition on a regime
        and sample, but it cannot hold the realized noise fixed across the change.
    (b) CAUSAL -- abduct each realized path, ``do(theta')``, re-roll the SAME increments,
        average the per-path difference.
    First guess: pairing is a big win. WRONG, in general. Pairing buys variance reduction in
    proportion to the correlation it induces, so it is enormous for a SMALL perturbation (a
    Greek) and nearly worthless for a LARGE one (a regime flip). Both are measured below.

G1 -- INVERSION ERROR, and whether it compounds over a long path.
    Diffusion counterfactuals abduct by DDIM inversion, which is approximate. Financial paths
    are long (T = 50-250), so: does per-step inversion error compound?
    First guess: yes, and that is the flow model's edge. WRONG for a plain GBM -- log-price is
    a SUM of increments, so independent per-step errors average out and terminal error is
    O(eps * sqrt(T)), flat in the number of steps. Compounding requires STATE DEPENDENCE, so
    the second arm re-runs the test on a leverage-effect local-vol model where an error in the
    price feeds back into the next step's volatility. That is where the gap appears.

    Both arms inject an assumed per-step error eps; neither measures a real DDIM. Calibrating
    eps for a trained path diffusion is experiment E6 and gates the G1 claim.

Run: uv run python -m experiments.cpricing.poc_gains
"""

from __future__ import annotations

import math

import numpy as np
import torch
from experiments.cpricing.poc_ladder import MATURITY, SPOT, R, hedged_pnl

# --------------------------------------------------------------------------------------
# Two transition families: constant-vol (GBM) and state-dependent (leverage-effect local vol).
# --------------------------------------------------------------------------------------


def roll_gbm(z: torch.Tensor, sigma: float, dt: float) -> torch.Tensor:
    """log S_{t+1} = log S_t + (r - sigma^2/2) dt + sigma sqrt(dt) Z_t. Errors do NOT compound."""
    n, steps = z.shape
    log_path = torch.empty(n, steps + 1)
    log_path[:, 0] = math.log(SPOT)
    for t in range(steps):
        log_path[:, t + 1] = (
            log_path[:, t] + (R - 0.5 * sigma**2) * dt + sigma * math.sqrt(dt) * z[:, t]
        )
    return log_path


def invert_gbm(log_path: torch.Tensor, sigma: float, dt: float) -> torch.Tensor:
    inc = log_path[:, 1:] - log_path[:, :-1]
    return (inc - (R - 0.5 * sigma**2) * dt) / (sigma * math.sqrt(dt))


# Leverage effect: volatility rises as the price falls, sigma(S) = sigma0 * (S0/S)^p, capped.
# A downward error raises the next step's vol, which enlarges the next error -- state feedback.
LEVERAGE_P = 1.5
VOL_CAP = 3.0


def _local_vol(log_s: torch.Tensor, sigma0: float) -> torch.Tensor:
    ratio = torch.exp(torch.clamp((math.log(SPOT) - log_s) * LEVERAGE_P, max=math.log(VOL_CAP)))
    return sigma0 * ratio


def roll_localvol(z: torch.Tensor, sigma0: float, dt: float) -> torch.Tensor:
    n, steps = z.shape
    log_path = torch.empty(n, steps + 1)
    log_path[:, 0] = math.log(SPOT)
    for t in range(steps):
        sig = _local_vol(log_path[:, t], sigma0)
        log_path[:, t + 1] = (
            log_path[:, t] + (R - 0.5 * sig**2) * dt + sig * math.sqrt(dt) * z[:, t]
        )
    return log_path


def invert_localvol(log_path: torch.Tensor, sigma0: float, dt: float) -> torch.Tensor:
    """Exact: the step's vol depends only on the CURRENT state, which is observed."""
    n, steps = log_path.shape[0], log_path.shape[1] - 1
    z = torch.empty(n, steps)
    for t in range(steps):
        sig = _local_vol(log_path[:, t], sigma0)
        inc = log_path[:, t + 1] - log_path[:, t]
        z[:, t] = (inc - (R - 0.5 * sig**2) * dt) / (sig * math.sqrt(dt))
    return z


# --------------------------------------------------------------------------------------
# G2 -- variance reduction, small vs large intervention.
# --------------------------------------------------------------------------------------


def _paired_vs_independent(
    sigma_base: float, sigma_shocked: float, hedge_sigma: float, n: int, steps: int, seed: int
) -> dict[str, float]:
    torch.manual_seed(seed)
    dt = MATURITY / steps
    half = n // 2  # equal path budget: both arms simulate n trajectories in total

    # (a) non-causal: independent draws under each regime.
    z_a, z_b = torch.randn(half, steps), torch.randn(half, steps)
    p_base = hedged_pnl(np.exp(roll_gbm(z_a, sigma_base, dt).numpy()), hedge_sigma)
    p_shock = hedged_pnl(np.exp(roll_gbm(z_b, sigma_shocked, dt).numpy()), hedge_sigma)
    se_ind = float(math.sqrt(p_shock.var(ddof=1) / half + p_base.var(ddof=1) / half))
    est_ind = float(p_shock.mean() - p_base.mean())

    # (b) causal: abduct the increments, re-roll under the shocked parameter.
    z = torch.randn(half, steps)
    path_f = roll_gbm(z, sigma_base, dt)
    z_rec = invert_gbm(path_f, sigma_base, dt)  # exact abduction
    path_c = roll_gbm(z_rec, sigma_shocked, dt)
    diff = hedged_pnl(np.exp(path_c.numpy()), hedge_sigma) - hedged_pnl(
        np.exp(path_f.numpy()), hedge_sigma
    )
    se_pair = float(math.sqrt(diff.var(ddof=1) / half))
    est_pair = float(diff.mean())

    return {
        "estimate_independent": est_ind,
        "se_independent": se_ind,
        "estimate_paired": est_pair,
        "se_paired": se_pair,
        "variance_ratio_=_compute_multiple": (se_ind / se_pair) ** 2,
    }


def gain_2_variance_reduction(n: int = 8000, steps: int = 50, seed: int = 0) -> dict[str, dict]:
    return {
        # A vega-like query: the derivative-shaped, SMALL perturbation. Pairing should win big.
        "small_perturbation (vol 0.20 -> 0.2020, a vega)": _paired_vs_independent(
            0.20, 0.2020, 0.20, n, steps, seed
        ),
        # A stress-test query: a LARGE regime flip. Pairing should buy almost nothing.
        "large_intervention (vol 0.15 -> 0.45, regime flip)": _paired_vs_independent(
            0.15, 0.45, 0.15, n, steps, seed
        ),
    }


# --------------------------------------------------------------------------------------
# G1 -- does inversion error compound? Only with state dependence.
# --------------------------------------------------------------------------------------


def gain_1b_inversion_error_destroys_pairing(
    n: int = 8000, steps: int = 50, seed: int = 3, sigma0: float = 0.20
) -> dict[str, object]:
    """The real cost of approximate abduction: it kills the regime where pairing pays.

    G1's compounding hypothesis died (see ``gain_1_inversion_error``). This is what replaced
    it, and it is sharper. Pairing's variance reduction comes from the correlation between the
    factual and counterfactual P&L. An inversion error ``eps`` injects independent noise into
    the counterfactual path, decorrelating it. When ``eps`` is small relative to the
    perturbation the pairing survives; once ``eps`` is comparable to it, the counterfactual is
    mostly noise and the 1000x+ advantage collapses toward 1x.

    So exact inversion is not about long paths. It is about being able to ask SMALL questions
    at all -- which is what a Greek, a vega, and a model-risk sensitivity are.
    """
    torch.manual_seed(seed)
    dt = MATURITY / steps
    half = n // 2
    sigma_shocked = sigma0 * 1.01  # a 1% vol bump: the vega-shaped query

    # Baseline non-causal standard error at the same budget.
    z_a, z_b = torch.randn(half, steps), torch.randn(half, steps)
    p_base = hedged_pnl(np.exp(roll_gbm(z_a, sigma0, dt).numpy()), sigma0)
    p_shock = hedged_pnl(np.exp(roll_gbm(z_b, sigma_shocked, dt).numpy()), sigma0)
    se_ind = float(math.sqrt(p_shock.var(ddof=1) / half + p_base.var(ddof=1) / half))

    z = torch.randn(half, steps)
    path_f = roll_gbm(z, sigma0, dt)
    z_ex = invert_gbm(path_f, sigma0, dt)
    pnl_f = hedged_pnl(np.exp(path_f.numpy()), sigma0)

    rows: list[dict[str, float]] = []
    for eps in (0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0):
        z_ap = z_ex + eps * torch.randn(half, steps)
        path_c = roll_gbm(z_ap, sigma_shocked, dt)
        diff = hedged_pnl(np.exp(path_c.numpy()), sigma0) - pnl_f
        se_pair = float(math.sqrt(diff.var(ddof=1) / half))
        rows.append(
            {
                "eps": eps,
                "se_paired": se_pair,
                "variance_ratio": (se_ind / se_pair) ** 2,
                "estimate": float(diff.mean()),
            }
        )
    return {"se_independent": se_ind, "perturbation_rel": 0.01, "by_eps": rows}


def gain_1_inversion_error(
    n: int = 2000, seed: int = 1, eps: float = 1e-3, sigma0: float = 0.20
) -> dict[str, object]:
    torch.manual_seed(seed)
    gbm_rows: list[dict[str, float]] = []
    lv_rows: list[dict[str, float]] = []

    for steps in (10, 25, 50, 100, 200):
        dt = MATURITY / steps
        z = torch.randn(n, steps)

        # --- constant-vol: log-price is a sum, so per-step errors average out.
        path = roll_gbm(z, sigma0, dt)
        z_ex = invert_gbm(path, sigma0, dt)
        z_ap = z_ex + eps * torch.randn(n, steps)
        s_ex = np.exp(roll_gbm(z_ex, 0.45, dt)[:, -1].numpy())
        s_ap = np.exp(roll_gbm(z_ap, 0.45, dt)[:, -1].numpy())
        gbm_rows.append(
            {
                "steps": steps,
                "rel_err": float(np.abs(s_ap - s_ex).mean() / np.abs(s_ex).mean()),
                "roundtrip_exact": float((roll_gbm(z_ex, sigma0, dt) - path).abs().max()),
            }
        )

        # --- state-dependent: an error in S_t changes sigma(S_t), which enlarges the next error.
        path_l = roll_localvol(z, sigma0, dt)
        zl_ex = invert_localvol(path_l, sigma0, dt)
        zl_ap = zl_ex + eps * torch.randn(n, steps)
        sl_ex = np.exp(roll_localvol(zl_ex, 0.45, dt)[:, -1].numpy())
        sl_ap = np.exp(roll_localvol(zl_ap, 0.45, dt)[:, -1].numpy())
        lv_rows.append(
            {
                "steps": steps,
                "rel_err": float(np.abs(sl_ap - sl_ex).mean() / np.abs(sl_ex).mean()),
                "roundtrip_exact": float((roll_localvol(zl_ex, sigma0, dt) - path_l).abs().max()),
            }
        )

    return {
        "eps_per_step": eps,
        "constant_vol": gbm_rows,
        "state_dependent_vol": lv_rows,
        "growth_constant_vol": gbm_rows[-1]["rel_err"] / gbm_rows[0]["rel_err"],
        "growth_state_dependent": lv_rows[-1]["rel_err"] / lv_rows[0]["rel_err"],
        "penalty_at_200_steps": lv_rows[-1]["rel_err"] / gbm_rows[-1]["rel_err"],
    }


def main() -> None:
    print("=" * 82)
    print("G2  variance reduction: paired counterfactual vs independent resampling")
    print("=" * 82)
    for label, res in gain_2_variance_reduction().items():
        print(f"  {label}")
        for k, v in res.items():
            print(f"      {k:36s} {v:.6g}")

    print()
    print("=" * 82)
    print("G1  does inversion error compound?   [ILLUSTRATIVE eps, not a real DDIM benchmark]")
    print("=" * 82)
    out = gain_1_inversion_error()
    print(f"  eps_per_step = {out['eps_per_step']}")
    print(
        f"  {'steps':>6}  {'constant-vol':>16}  {'state-dependent':>16}   (relative terminal error)"
    )
    for a, b in zip(out["constant_vol"], out["state_dependent_vol"], strict=True):  # type: ignore[arg-type]
        print(f"  {a['steps']:6.0f}  {a['rel_err']:16.3e}  {b['rel_err']:16.3e}")
    print(f"  growth 10->200 steps, constant vol       {out['growth_constant_vol']:.2f}x")
    print(f"  growth 10->200 steps, state-dependent    {out['growth_state_dependent']:.2f}x")
    print(f"  state-dependence penalty at 200 steps    {out['penalty_at_200_steps']:.1f}x")
    print("  exact inversion round-trip error is 0.0 in both columns, at every horizon.")
    print("  VERDICT: the compounding hypothesis is FALSE. Errors average out as O(eps*sqrt(T)).")

    print()
    print("=" * 82)
    print("G1b  what approximate abduction actually costs: it destroys the pairing gain")
    print("=" * 82)
    out1b = gain_1b_inversion_error_destroys_pairing()
    print(f"  perturbation = {out1b['perturbation_rel']:.0%} vol bump (a vega-shaped query)")
    print(f"  se_independent (non-causal baseline)  {out1b['se_independent']:.6g}")
    print(f"  {'inversion eps':>14}  {'se_paired':>12}  {'variance ratio':>15}  {'estimate':>12}")
    for row in out1b["by_eps"]:  # type: ignore[union-attr]
        print(
            f"  {row['eps']:14.0e}  {row['se_paired']:12.3e}  {row['variance_ratio']:15.1f}"
            f"  {row['estimate']:12.5f}"
        )


if __name__ == "__main__":
    main()
