#!/usr/bin/env python3
"""Can we learn the market's SCM with the shipped discovery tools?

Probe for the "learn the SCM from the market" question. It tests the specific failure mode
that makes naive causal discovery on market data untrustworthy -- and then tests whether the
library's interventional discovery actually fixes it. (It does not, quite. That is the
finding, and it names a concrete library gap.)

THE FAILURE MODE. Constraint-based discovery (PC/FCI) assumes *faithfulness*: no conditional
independence in the data that is not implied by the graph. Efficient markets violate this
by construction, because arbitrage is a mechanism whose entire purpose is to cancel
exploitable dependence. Model it directly:

    N  -- a scheduled macro surprise (FOMC/CPI): exogenous, timing known in advance
    F  -- dealer / arbitrageur hedging flow, which responds to N
    V  -- an implied-vol signal, pushed UP by N directly and DOWN by the flow F

When the arbitrage is effective the two paths N -> V and N -> F -> V cancel, so N and V are
marginally independent *even though N causes V*. PC sees independence and deletes a real
causal edge. This is not a contrived corner: cancellation is what arbitrage does for a living.

THREE TESTS.
  1. Observational PC on this system -- does it delete the real N -> V edge?
  2. ``discover_interventional`` with do(F) -- does experimental data repair the deletion?
  3. A conditional-independence test run *inside* the do(F) regime -- is the dependence
     visible there, i.e. is the information actually present in interventional data?

Run: uv run python -m experiments.cpricing.poc_discovery
"""

from __future__ import annotations

import numpy as np

from causalrl.discovery import conditional_mutual_information, discover, discover_interventional

N_LEVELS = 3  # macro surprise: {0, 1, 2} = {dovish, neutral, hawkish}


def simulate(
    n: int,
    seed: int,
    *,
    do_flow: int | None = None,
    arb_strength: float = 1.0,
) -> dict[str, np.ndarray]:
    """Sample the N -> F -> V / N -> V system.

    ``do_flow`` pins the flow F to a constant -- a perfect intervention that severs the
    arbitrage channel. ``arb_strength`` scales how completely the flow offsets the direct
    effect; 1.0 is exact cancellation (a perfectly efficient market).
    """
    rng = np.random.default_rng(seed)

    news = rng.integers(0, N_LEVELS, size=n)

    # Dealers observe the surprise and trade against it, imperfectly. No clipping: a boundary
    # would leak residual News-dependence into IV and blunt the very cancellation being tested.
    jitter = rng.choice([-1, 0, 1], size=n, p=[0.25, 0.50, 0.25])
    flow_raw = news + jitter if do_flow is None else np.full(n, do_flow)

    # IV is pushed up by the news (coefficient 2) and down by the flow. At arb_strength = 1 the
    # flow coefficient is also 2, so 2*News - 2*(News + jitter) = -2*jitter: the News term cancels
    # EXACTLY and IV is marginally independent of News despite News being one of its parents.
    noise = rng.choice([-1, 0, 1], size=n, p=[0.15, 0.70, 0.15])
    iv_raw = 2 * news - round(2 * arb_strength) * flow_raw + noise

    # Shift both columns to non-negative integers for the discrete CMI estimator.
    return {"News": news, "Flow": flow_raw + 1, "IV": iv_raw + 6}


def _edges(cpdag) -> set[tuple[str, str]]:
    out = {tuple(e) for e in cpdag.directed_edges}
    out |= {tuple(sorted(e)) for e in cpdag.undirected_edges}
    return out  # type: ignore[return-value]


def _adjacent(cpdag, a: str, b: str) -> bool:
    for x, y in cpdag.directed_edges:
        if {x, y} == {a, b}:
            return True
    return any(set(e) == {a, b} for e in cpdag.undirected_edges)


def main() -> None:
    n, seed = 20_000, 0

    # ------------------------------------------------------------------ test 1: observational
    print("=" * 82)
    print("TEST 1  observational PC on an efficient (fully arbitraged) market")
    print("=" * 82)
    obs = simulate(n, seed, arb_strength=1.0)

    cmi_nv = conditional_mutual_information(obs, "News", "IV", [])
    cmi_nv_given_f = conditional_mutual_information(obs, "News", "IV", ["Flow"])
    print(f"  CMI(News ; IV)          = {cmi_nv:.5f}   <- marginal, ~0 because the paths cancel")
    print(f"  CMI(News ; IV | Flow)   = {cmi_nv_given_f:.5f}   <- reappears once Flow is blocked")

    cpdag = discover(obs, ["News", "Flow", "IV"])
    found = _adjacent(cpdag, "News", "IV")
    print(f"  discovered edges        : {sorted(_edges(cpdag))}")
    print(f"  News--IV edge recovered : {found}   (ground truth: the edge EXISTS)")
    print(f"  VERDICT: {'no failure' if found else 'PC DELETED A REAL CAUSAL EDGE'}")

    # For contrast: an inefficient market, where the arbitrage is only partial.
    obs_ineff = simulate(n, seed, arb_strength=0.5)
    cpdag_ineff = discover(obs_ineff, ["News", "Flow", "IV"])
    print(
        f"  same system, HALF-arbitraged (arb_strength=0.5): News--IV recovered = "
        f"{_adjacent(cpdag_ineff, 'News', 'IV')}"
    )
    print("  -> the better the market works, the worse discovery works. That is the problem.")

    # --------------------------------------------------- test 2: does interventional data fix it?
    print()
    print("=" * 82)
    print("TEST 2  discover_interventional with do(Flow) -- a scheduled event severs the channel")
    print("=" * 82)
    idata = {"Flow": simulate(n, seed + 1, do_flow=1, arb_strength=1.0)}
    icpdag = discover_interventional(obs, idata, ["News", "Flow", "IV"])
    ifound = _adjacent(icpdag, "News", "IV")
    print(f"  discovered edges        : {sorted(_edges(icpdag))}")
    print(f"  News--IV edge recovered : {ifound}")
    print(
        "  VERDICT: "
        + (
            "repaired"
            if ifound
            else "STILL MISSING -- interventional discovery ORIENTS existing edges,\n"
            "           it does not re-test the skeleton, so a faithfulness deletion survives."
        )
    )

    # ------------------------------------------- test 3: is the information there to be had?
    print()
    print("=" * 82)
    print("TEST 3  is the dependence visible INSIDE the do(Flow) regime?")
    print("=" * 82)
    idf = idata["Flow"]
    cmi_int = conditional_mutual_information(idf, "News", "IV", [])
    print(f"  CMI(News ; IV) under do(Flow=1) = {cmi_int:.5f}")
    print(f"  CMI(News ; IV) observational    = {cmi_nv:.5f}")
    print(
        "  VERDICT: "
        + (
            "the information IS present in interventional data -- the skeleton phase simply\n"
            "           never looks at it. Concrete library gap: let `discover` pool or test\n"
            "           independence per-regime, not just orient afterwards."
            if cmi_int > 10 * max(cmi_nv, 1e-9)
            else "not recovered even interventionally; the design needs rethinking."
        )
    )


if __name__ == "__main__":
    main()
