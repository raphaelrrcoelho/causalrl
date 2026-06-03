"""Partial-identification bounds + the sensitivity tipping point (``tipping_gamma``).

Off-policy / confounded contributions are rarely point-identified. causalrl gives you

1. marginal-sensitivity-model (MSM) **bounds** that widen with the assumed unobserved-confounding
   strength ``gamma`` (Tan 2006; Kallus-Zhou 2020), and
2. :func:`causalrl.tipping_gamma` — the smallest ``gamma`` at which a conclusion (here, "this
   contribution is positive") can no longer be defended. It is the odds-ratio-scale analog of the
   E-value (VanderWeele & Ding, 2017): a larger value ⇒ a more robust conclusion.

Run:  python examples/sensitivity_bounds.py
"""

from __future__ import annotations

import numpy as np

from causalrl import msm_contribution_bounds, tipping_gamma


def main() -> None:
    rng = np.random.default_rng(0)
    n = 2000
    # A logged contribution comparison: outcomes Y in [0,1], a binary factor F whose true effect on
    # Y is +0.12, and the Z-ignoring nominal propensity e0 at the logged action. The one-hot arms
    # 1{F=1} / 1{F=0} partition the log, so the MSM contribution interval is sharp.
    f = rng.integers(0, 2, n).astype(float)
    e0 = np.where(f == 1, 0.6, 0.4)
    y = np.clip(0.5 + 0.12 * f + rng.normal(0, 0.1, n), 0.0, 1.0)
    on, off = (f == 1).astype(float).tolist(), (f == 0).astype(float).tolist()
    yl, el = y.tolist(), e0.tolist()

    print("MSM contribution bound  V(do F=1) - V(do F=0)  vs the assumed confounding level gamma:")
    for g in (1.0, 1.5, 2.0, 3.0):
        iv = msm_contribution_bounds(yl, el, on, off, gamma=g)
        w = iv.upper - iv.lower
        flag = "  <- sign uncertain" if iv.lower <= 0 <= iv.upper else ""
        print(f"  gamma={g:<4} [{iv.lower:+.3f}, {iv.upper:+.3f}]  width={w:.3f}{flag}")

    # How strong would unobserved confounding have to be to overturn "the contribution is positive"?
    g_star = tipping_gamma(
        lambda g: msm_contribution_bounds(yl, el, on, off, gamma=g),
        reference=0.0,
        gamma_max=10.0,
    )
    if g_star is None:
        print("\ntipping_gamma = None -> positive conclusion robust to confounding up to gamma=10.")
    else:
        print(
            f"\ntipping_gamma = {g_star:.2f} -> a confounder with treatment odds-ratio >= "
            f"{g_star:.2f} could explain away the positive contribution; below it the sign holds."
        )


if __name__ == "__main__":
    main()
