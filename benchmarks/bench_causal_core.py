"""Micro-benchmarks + regression guards for the causal-core fast paths.

Run: PYTHONPATH=src .venv/bin/python benchmarks/bench_causal_core.py

Guards two performance claims behind the 2026-05-30 API/perf pass:
  (1) Known-noise abduction gives an EXACT continuous counterfactual in O(n), where the old
      rejection path needs ~0 acceptance for continuous evidence (effectively unusable).
  (2) The MSM closed form (`ipw_sensitivity_bounds`, sorted prefix-sums) matches a scipy
      linear-program reference to within 1e-6 and is materially faster.

Asserts correctness + a conservative speedup floor so a regression fails the bench.
"""
from __future__ import annotations

import time

import numpy as np
import torch
from torch.distributions import Normal

from causalrl.identification.bounds import ipw_sensitivity_bounds
from causalrl.scm.graph import CausalGraph
from causalrl.scm.mechanisms import LinearGaussianMechanism
from causalrl.scm.scm import StructuralCausalModel


def _timed(fn, repeat: int = 5) -> tuple[float, object]:
    best = float("inf")
    out = None
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t0)
    return best, out


def _lin_scm() -> StructuralCausalModel:
    g = CausalGraph(directed_edges=[("X", "Y")])
    mechs = {
        "X": LinearGaussianMechanism([], {}, bias=0.0),
        "Y": LinearGaussianMechanism(["X"], {"X": 2.0}, bias=0.0),
    }
    exo = {"X": Normal(0.0, 1.0), "Y": Normal(0.0, 1.0)}
    return StructuralCausalModel(g, mechs, exo)


def bench_continuous_counterfactual() -> None:
    """Exact known-noise abduction vs the (continuous-evidence) rejection path."""
    scm = _lin_scm()
    n = 20_000

    def exact() -> float:
        # abduct-once (pin U_x, U_y), predict under do(X=3): Y = 2*3 + U_y exactly.
        post = scm.abduct(known={"X": 0.5, "Y": 0.1}, n=n)
        return float(post.predict(do={"X": 3.0})["Y"].mean())

    t_exact, val = _timed(exact)
    assert abs(val - 6.1) < 1e-6, f"exact CF wrong: {val}"

    # Rejection path on continuous evidence: near-zero acceptance. Measure acceptance rate
    # at the same n to show why it is unusable (don't assert timing — it mostly fails).
    noise = scm._sample_exogenous(n, seed=0)
    factual = scm._evaluate(noise)
    atol = 1e-6
    accepted = int(((factual["Y"] - 1.1).abs() <= atol).sum())
    print(
        f"[counterfactual] exact known-noise: Y={val:.6f} in {t_exact * 1e3:.2f} ms (n={n}); "
        f"rejection acceptance at atol={atol}: {accepted}/{n} draws "
        f"({'UNUSABLE' if accepted == 0 else f'{accepted} kept'})"
    )
    assert accepted == 0, "continuous rejection unexpectedly accepted — bench premise off"


def _linprog_msm_reference(y: np.ndarray, e: np.ndarray, gamma: float) -> tuple[float, float]:
    """Reference MSM bound via LP: optimize self-normalized Σw_i y_i / Σw_i over the Tan box.
    The self-normalized fractional program's extremum is attained at a threshold rule, but we
    solve it honestly with scipy as an independent cross-check of the closed form."""
    from scipy.optimize import linprog  # type: ignore[import-untyped]

    odds = (1.0 - e) / e
    lo = 1.0 + odds / gamma
    hi = 1.0 + odds * gamma
    n = len(y)

    def solve(maximize: bool) -> float:
        # Charnes-Cooper: w_i = lo_i*t + s_i, s_i in [0, (hi_i-lo_i)*t], Σ w_i = 1.
        # Variables: [s_0..s_{n-1}, t]. Maximize Σ s_i y_i + t Σ lo_i y_i.
        c_y = y.copy()
        c = np.concatenate([(-c_y if maximize else c_y), [(-(lo * y).sum() if maximize else (lo * y).sum())]])
        # Σ w_i = 1  ->  Σ s_i + t Σ lo_i = 1
        a_eq = np.concatenate([np.ones(n), [lo.sum()]]).reshape(1, -1)
        b_eq = np.array([1.0])
        # s_i <= (hi_i - lo_i) * t  ->  s_i - (hi_i-lo_i) t <= 0
        a_ub = np.zeros((n, n + 1))
        for i in range(n):
            a_ub[i, i] = 1.0
            a_ub[i, n] = -(hi[i] - lo[i])
        b_ub = np.zeros(n)
        bounds = [(0, None)] * n + [(1e-12, None)]
        res = linprog(c, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq, bounds=bounds, method="highs")
        val = float((y * (res.x[:n] + lo * res.x[n])).sum())
        return val

    return solve(maximize=False), solve(maximize=True)


def bench_msm_closed_form() -> None:
    """Closed-form MSM (prefix sums) vs scipy LP reference: agreement + speedup."""
    rng = np.random.default_rng(0)
    n = 400  # LP is O(n vars); keep modest so the reference is tractable
    y = rng.uniform(0, 1, size=n)
    e = rng.uniform(0.2, 0.8, size=n)
    gamma = 2.0

    t_cf, iv = _timed(lambda: ipw_sensitivity_bounds(y.tolist(), e.tolist(), gamma=gamma))
    t_lp, (lp_lo, lp_hi) = _timed(lambda: _linprog_msm_reference(y, e, gamma), repeat=2)

    assert abs(iv.lower - lp_lo) < 1e-6, f"lower mismatch: closed={iv.lower} lp={lp_lo}"
    assert abs(iv.upper - lp_hi) < 1e-6, f"upper mismatch: closed={iv.upper} lp={lp_hi}"
    speedup = t_lp / t_cf
    print(
        f"[msm bounds] closed-form [{iv.lower:.6f}, {iv.upper:.6f}] in {t_cf * 1e3:.3f} ms "
        f"vs scipy-LP in {t_lp * 1e3:.2f} ms  (speedup {speedup:.0f}x, agree <1e-6, n={n})"
    )
    assert speedup >= 5.0, f"closed-form expected >=5x faster than LP, got {speedup:.1f}x"


def main() -> None:
    print("causal-core micro-benchmarks (2026-05-30 API/perf pass)")
    bench_continuous_counterfactual()
    bench_msm_closed_form()
    print("OK — all bench assertions passed")


if __name__ == "__main__":
    main()
