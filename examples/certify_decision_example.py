"""certify_decision — one call: is a decision from confounded logs robust to hidden confounding?

A worked "ship it?" decision. The logs say the new policy (the *treated* arm) looks better, but
assignment was confounded by a hidden severity variable Z (healthier units were preferentially
given the new policy). :func:`causalrl.certify_decision` composes the decision stack and returns a
human-readable verdict:

1. with a *measured* Z (post-hoc audit), the sign-robustness certificate refuses to certify when
   the omitted-variable bias could exceed the logged contrast; and
2. with only the nominal *propensities*, the marginal-sensitivity-model tipping point reports how
   strong unobserved confounding would have to be (a Tan odds-ratio) to overturn the call.

Run:  python examples/certify_decision_example.py
"""

from __future__ import annotations

import numpy as np

from causalrl import certify_decision


def main() -> None:
    rng = np.random.default_rng(0)
    n = 20000
    z = rng.integers(0, 2, size=n)  # hidden severity (0 = healthy, 1 = sick)
    # The logger preferentially gives the new policy to healthy units...
    f = (rng.random(n) < 0.3 + 0.4 * (z - 0.5)).astype(int)
    # ...and sicker units score higher here, while the new policy is in TRUTH slightly worse.
    y = 1.0 * z - 0.1 * f + rng.normal(0, 0.1, size=n)

    print("A confounded 'ship the new policy?' decision\n" + "-" * 44)

    # (1) Post-hoc audit recovers the severity bins -> sign-robustness certificate.
    audited = certify_decision(y, f, confounder_bins=z)
    print("with a measured confounder (audit):")
    print(" ", audited.summary)

    # (2) All we kept were the logging propensities -> MSM sensitivity tipping point.
    e0 = np.clip(np.where(f == 1, 0.3 + 0.4 * (z - 0.5), 1.0 - (0.3 + 0.4 * (z - 0.5))), 0.05, 0.95)
    sens = certify_decision(y, f, propensities=e0, gamma_max=10.0)
    print("with only logging propensities (sensitivity):")
    print(" ", sens.summary)

    # Contrast: a decision with a genuine effect and a thin confounding channel DOES certify.
    y_clean = 0.4 * f + 0.05 * z + rng.normal(0, 0.1, size=n)
    clean = certify_decision(y_clean, f, confounder_bins=z)
    print("a genuinely-better arm over a thin channel:")
    print(" ", clean.summary)


if __name__ == "__main__":
    main()
