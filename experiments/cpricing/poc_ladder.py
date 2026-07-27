#!/usr/bin/env python3
"""Feasibility probe for the causal-derivative-pricing proposal: does the shipped stack
actually support the three identification rungs the proposal claims?

Three claims, three assertions. Nothing here is a research result -- this is the
"can the library carry the weight" check that gates the proposal.

C1 (IDENTIFIED). A Euler-discretised diffusion built with ``build_unrolled_scm`` admits
    *exact* path abduction: invert the location-scale transition to recover every Brownian
    increment, pin them, re-roll. The round-trip reproduces the factual path to float
    precision, and ``do(regime := flipped)`` re-rolls the SAME noise under a different
    volatility regime -- the pathwise counterfactual.

C2 (BOUNDED). The shipped Tan marginal-sensitivity-model kernel
    (``ipw_sensitivity_bounds``) solves *exactly* the bounded-pricing-kernel program that
    defines a gain-loss / no-good-deal price bound. Verified against an independent
    brute-force solution of the bounded-likelihood-ratio program.

C3 (EMPIRICAL / hedge). On an infinite-variance P&L sample, ``certify_mean`` refuses the
    mean and downgrades to a median certificate instead of reporting a fragile number.

Run: uv run python experiments/cpricing/poc_ladder.py
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch.distributions import Bernoulli, Normal

from causalrl.bounds.continuous import certify_mean, tail_index_hill
from causalrl.identification.bounds import ipw_sensitivity_bounds

# --------------------------------------------------------------------------------------
# A discretised two-regime diffusion, expressed as a time-unrolled SCM.
#
#   d log S = (r - sigma(V)^2 / 2) dt + sigma(V) dW,      V in {0, 1} a shared latent regime
#
# Euler-Maruyama on a grid of T steps is exactly the unrolled SCM the library builds; the
# causal license for treating it as the diffusion's interventional semantics is Hansen &
# Sokol (EJP 2014), whose post-intervention SDE is the limit of post-intervention Euler SEMs.
# --------------------------------------------------------------------------------------

R = 0.0  # risk-free rate, in the numeraire
SIGMA = (0.15, 0.45)  # sigma(V=0), sigma(V=1) -- the "calm" and "stressed" regimes
T_STEPS = 50
MATURITY = 1.0
DT = MATURITY / T_STEPS
STRIKE = 100.0
SPOT = 100.0


def _sigma_of(v: torch.Tensor) -> torch.Tensor:
    """sigma(V) for a per-sample regime indicator tensor."""
    return torch.where(v > 0.5, torch.full_like(v, SIGMA[1]), torch.full_like(v, SIGMA[0]))


def transition(
    state: torch.Tensor,
    action: object,
    latents: dict[str, torch.Tensor],
    noise: torch.Tensor,
) -> torch.Tensor:
    """log S_{t+1} = log S_t + (r - sigma^2/2) dt + sigma sqrt(dt) Z_t. Location-scale in Z."""
    sig = _sigma_of(latents["V"])
    return state + (R - 0.5 * sig**2) * DT + sig * math.sqrt(DT) * noise


def invert_path(log_path: torch.Tensor, v: torch.Tensor) -> dict[str, torch.Tensor]:
    """Recover every Brownian increment from an observed log-price path (the abduction step).

    ``log_path`` is ``(n, T+1)``. Location-scale mechanisms are invertible in closed form, so
    this is exact -- the property that licenses ``kind=IDENTIFIED`` for the counterfactual.
    """
    sig = _sigma_of(v)
    drift = (R - 0.5 * sig**2) * DT
    increments = log_path[:, 1:] - log_path[:, :-1]
    z = (increments - drift[:, None]) / (sig[:, None] * math.sqrt(DT))
    return {f"state_{t + 1}": z[:, t] for t in range(z.shape[1])}


def _bs_call(spot: np.ndarray, sigma: float, tau: float) -> np.ndarray:
    """Black-Scholes call price (r = R, strike = STRIKE), vectorised over spot."""
    if tau <= 0:
        return np.maximum(spot - STRIKE, 0.0)
    vs = sigma * math.sqrt(tau)
    d1 = (np.log(spot / STRIKE) + (R + 0.5 * sigma**2) * tau) / vs
    d2 = d1 - vs
    ncdf = lambda x: 0.5 * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))  # noqa: E731
    return spot * ncdf(d1) - STRIKE * math.exp(-R * tau) * ncdf(d2)


def _bs_delta(spot: np.ndarray, sigma: float, tau: float) -> np.ndarray:
    if tau <= 0:
        return (spot > STRIKE).astype(float)
    vs = sigma * math.sqrt(tau)
    d1 = (np.log(spot / STRIKE) + (R + 0.5 * sigma**2) * tau) / vs
    ncdf = lambda x: 0.5 * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))  # noqa: E731
    return ncdf(d1)


def hedged_pnl(price_paths: np.ndarray, hedge_sigma: float) -> np.ndarray:
    """P&L of a discretely rebalanced short-call delta hedge run at ``hedge_sigma``.

    Sell the call at its ``hedge_sigma`` price, delta-hedge on the same grid, hold to expiry.
    Under continuous rebalancing with the *correct* sigma this is ~0; the residual is the
    quantity every model-risk question is actually about.
    """
    n, steps = price_paths.shape[0], price_paths.shape[1] - 1
    cash = _bs_call(price_paths[:, 0], hedge_sigma, MATURITY)  # premium received
    shares = np.zeros(n)
    for t in range(steps):
        tau = MATURITY - t * DT
        delta = _bs_delta(price_paths[:, t], hedge_sigma, tau)
        cash -= (delta - shares) * price_paths[:, t]  # buy the delta increment
        shares = delta
    cash += shares * price_paths[:, -1]  # liquidate
    cash -= np.maximum(price_paths[:, -1] - STRIKE, 0.0)  # pay the claim
    return cash


# --------------------------------------------------------------------------------------
# C1 -- exact path abduction and the pathwise regime counterfactual.
# --------------------------------------------------------------------------------------


def claim_1_pathwise_counterfactual(n: int = 4000, seed: int = 0) -> dict[str, float]:
    from causalrl.scm.unrolled import build_unrolled_scm

    torch.manual_seed(seed)
    scm = build_unrolled_scm(
        transition,
        horizon=T_STEPS,
        state0_dist=Normal(math.log(SPOT), 0.0001),
        latents={"V": Bernoulli(0.5)},
        process_noise_dist=Normal(0.0, 1.0),
    )

    # --- factual world: everyone starts calm (V = 0), draw a path per sample.
    v_fact = torch.zeros(n)
    z = torch.randn(n, T_STEPS)
    log_path = torch.empty(n, T_STEPS + 1)
    log_path[:, 0] = math.log(SPOT)
    for t in range(T_STEPS):
        log_path[:, t + 1] = transition(log_path[:, t], None, {"V": v_fact}, z[:, t])

    # --- abduction: recover the increments from the observed path alone.
    recovered = invert_path(log_path, v_fact)
    recovery_err = max(
        float((recovered[f"state_{t + 1}"] - z[:, t]).abs().max()) for t in range(T_STEPS)
    )

    known = {"state_0": log_path[:, 0], "V": v_fact, **recovered}
    post = scm.abduct(known=known, n=n)

    # --- round-trip: re-roll with no intervention must reproduce the factual path exactly.
    factual = post.predict()
    roundtrip_err = max(
        float((factual[f"state_{t}"] - log_path[:, t]).abs().max()) for t in range(T_STEPS + 1)
    )

    # --- the counterfactual: SAME Brownian path, stressed regime.
    cf = post.predict(do={"V": 1.0})
    s_fact = np.exp(log_path[:, -1].numpy())
    s_cf = np.exp(cf[f"state_{T_STEPS}"].numpy())

    # Counterfactual hedging P&L: what this book would have earned on this very path had the
    # regime been stressed, hedging at the calm vol throughout.
    paths_f = np.exp(np.stack([factual[f"state_{t}"].numpy() for t in range(T_STEPS + 1)], 1))
    paths_c = np.exp(np.stack([cf[f"state_{t}"].numpy() for t in range(T_STEPS + 1)], 1))
    pnl_f = hedged_pnl(paths_f, SIGMA[0])
    pnl_c = hedged_pnl(paths_c, SIGMA[0])

    assert recovery_err < 1e-4, f"increment recovery not exact: {recovery_err}"
    assert roundtrip_err < 1e-4, f"abduct->predict round-trip not exact: {roundtrip_err}"

    return {
        "recovery_err": recovery_err,
        "roundtrip_err": roundtrip_err,
        "mean_S_T_factual": float(s_fact.mean()),
        "mean_S_T_counterfactual": float(s_cf.mean()),
        "mean_pnl_factual": float(pnl_f.mean()),
        "mean_pnl_counterfactual": float(pnl_c.mean()),
        # The object no L2 (interventional-mean) analysis gives you: the *paired, per-path*
        # P&L difference. Its spread is model risk at the path level, not on average.
        "pathwise_pnl_delta_mean": float((pnl_c - pnl_f).mean()),
        "pathwise_pnl_delta_p05": float(np.quantile(pnl_c - pnl_f, 0.05)),
        "pathwise_pnl_delta_p95": float(np.quantile(pnl_c - pnl_f, 0.95)),
    }


# --------------------------------------------------------------------------------------
# C2 -- the MSM kernel IS a bounded-pricing-kernel (gain-loss) bound.
# --------------------------------------------------------------------------------------


def _brute_force_kernel_bound(payoff: np.ndarray, ratio_cap: float, maximize: bool) -> float:
    """sup/inf of E_Q[payoff] over kernels with dQ/dP in a box of width ``ratio_cap``.

    Independent reference implementation: the self-normalised optimum of a fractional-linear
    program over a box is attained at a threshold rule -- weight ``ratio_cap`` on payoffs above
    the optimal threshold and 1 below (or the mirror image) -- so sweeping every threshold and
    taking the best is exact, and shares no code with the library kernel.
    """
    order = np.argsort(payoff)
    y = payoff[order]
    n = y.size
    best = -np.inf if maximize else np.inf
    for k in range(n + 1):  # first k get the low weight, the rest the high weight
        w = np.empty(n)
        w[:k] = 1.0 if maximize else ratio_cap
        w[k:] = ratio_cap if maximize else 1.0
        val = float((w * y).sum() / w.sum())
        best = max(best, val) if maximize else min(best, val)
    return best


def claim_2_good_deal_equivalence(n: int = 3000, seed: int = 1) -> dict[str, float]:
    rng = np.random.default_rng(seed)

    # A payoff sample under the physical measure P: a call on a lognormal terminal price.
    vol = 0.2
    shock = vol * math.sqrt(MATURITY) * rng.normal(size=n)
    s_t = SPOT * np.exp((R - 0.5 * vol**2) * MATURITY + shock)
    payoff = np.maximum(s_t - STRIKE, 0.0)

    gamma = 2.0
    e = 0.5  # uniform nominal propensity -> a single shared weight box
    odds = (1.0 - e) / e
    # The MSM weight box is [1 + odds/gamma, 1 + odds*gamma]; only its RATIO matters to a
    # self-normalised functional, and that ratio is exactly the gain-loss / kernel-ratio cap.
    ratio_cap = (1.0 + odds * gamma) / (1.0 + odds / gamma)

    lo, hi = ipw_sensitivity_bounds(
        payoff.tolist(), np.full(n, e).tolist(), gamma=gamma, return_certificate=False
    )
    ref_lo = _brute_force_kernel_bound(payoff, ratio_cap, maximize=False)
    ref_hi = _brute_force_kernel_bound(payoff, ratio_cap, maximize=True)

    err = max(abs(lo - ref_lo), abs(hi - ref_hi))
    assert err < 1e-8, f"MSM kernel does not match the bounded-kernel program: {err}"

    # The limiting map, for the record: as the nominal propensity goes to 0 the achievable
    # kernel-ratio cap tends to gamma^2, so gamma = sqrt(L) recovers a gain-loss bound of L.
    limit_caps = {
        f"ratio_cap@e={ee}": (1.0 + ((1 - ee) / ee) * gamma) / (1.0 + ((1 - ee) / ee) / gamma)
        for ee in (0.5, 0.1, 0.01, 0.001)
    }

    return {
        "msm_lower": float(lo),
        "msm_upper": float(hi),
        "bruteforce_lower": float(ref_lo),
        "bruteforce_upper": float(ref_hi),
        "max_abs_err": float(err),
        "ratio_cap": float(ratio_cap),
        "gamma_squared": float(gamma**2),
        "unbounded_mean": float(payoff.mean()),
        **limit_caps,
    }


# --------------------------------------------------------------------------------------
# C3 -- heavy-tailed P&L: the library refuses the mean.
# --------------------------------------------------------------------------------------


def claim_3_heavy_tail_downgrade(n: int = 5000, seed: int = 2) -> dict[str, object]:
    rng = np.random.default_rng(seed)

    # Short-vol P&L with a jump component: small positive carry most days, rare large losses
    # with a Pareto tail of index 1.3 < 2 -- infinite variance, and here infinite mean is close.
    carry = rng.normal(0.05, 0.02, size=n)
    jump_hit = rng.random(n) < 0.03
    jump_loss = (rng.pareto(1.3, size=n) + 1.0) * 2.0
    pnl = carry - jump_hit * jump_loss

    alpha = tail_index_hill(pnl)
    cert = certify_mean(pnl.tolist())

    assert alpha < 2.0, f"expected an infinite-variance tail, got alpha={alpha}"
    assert cert.hedge is not None, "expected certify_mean to hedge on an infinite-variance sample"

    return {
        "hill_alpha": float(alpha),
        "certificate_kind": cert.kind.value,
        "estimand_target": cert.estimand.target,
        "hedge_reason": cert.hedge.reason,
        "downgraded_from": cert.hedge.downgraded_from,
        "certified_value": str(cert.value),
        "certified_ci": str(cert.ci),
        "naive_sample_mean": float(pnl.mean()),
        "certificate": str(cert),
    }


def main() -> None:
    print("=" * 78)
    print("C1  pathwise counterfactual via exact path abduction        [rung: IDENTIFIED]")
    print("=" * 78)
    for k, v in claim_1_pathwise_counterfactual().items():
        print(f"  {k:32s} {v}")

    print()
    print("=" * 78)
    print("C2  MSM kernel == bounded-pricing-kernel bound              [rung: BOUNDED]")
    print("=" * 78)
    for k, v in claim_2_good_deal_equivalence().items():
        print(f"  {k:32s} {v}")

    print()
    print("=" * 78)
    print("C3  heavy-tailed P&L: mean refused, median certified        [rung: hedge]")
    print("=" * 78)
    for k, v in claim_3_heavy_tail_downgrade().items():
        print(f"  {k:32s} {v}")

    print()
    print("all three feasibility claims hold.")


if __name__ == "__main__":
    main()
