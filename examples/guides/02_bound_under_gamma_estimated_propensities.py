"""Task guide 2: bound an effect under Gamma with estimated propensities.

When you only have an *estimated* propensity model, the marginal sensitivity model asks: if the true
propensity is within an odds-ratio Gamma of your estimate, how far could the treated-counterfactual
mean move? `ipw_sensitivity_bounds` returns a BOUNDED certificate at each Gamma (2.0 default); the
interval widens monotonically, and `tipping_gamma` is the smallest Gamma at which the bound first
crosses a reference (here 0).

Run: python examples/guides/02_bound_under_gamma_estimated_propensities.py
"""

from __future__ import annotations

import numpy as np

from causalrl import ipw_sensitivity_bounds, tipping_gamma
from causalrl.identification.bounds import Interval


def main() -> None:
    rng = np.random.default_rng(0)
    n = 2_000
    # Estimated propensities for the treated units and their outcomes.
    e_hat = rng.uniform(0.2, 0.8, size=n)
    y = (rng.standard_normal(n) + 0.3).tolist()

    for gamma in (1.0, 1.5, 2.0, 3.0):
        cert = ipw_sensitivity_bounds(y, e_hat.tolist(), gamma=gamma)  # BOUNDED Certificate (2.0)
        assert cert.value is not None
        lo, hi = cert.value.lower, cert.value.upper
        print(f"Gamma={gamma:>3}: E[Y(1)] in [{lo:+.4f}, {hi:+.4f}]  ({cert.kind.name})")

    # The Gamma at which the treated-mean bound first admits 0.
    legacy: Interval = ipw_sensitivity_bounds(
        y, e_hat.tolist(), gamma=1.0, return_certificate=False
    )
    print(f"point estimate (Gamma=1): {legacy.lower:.4f}")
    g_star = tipping_gamma(
        lambda g: ipw_sensitivity_bounds(y, e_hat.tolist(), gamma=g, return_certificate=False),
        reference=0.0,
        gamma_max=10.0,
    )
    print(f"tipping Gamma vs 0      : {g_star}")
    print("OK — sharp estimated-propensity bounds, honest about the sensitivity budget")


if __name__ == "__main__":
    main()
