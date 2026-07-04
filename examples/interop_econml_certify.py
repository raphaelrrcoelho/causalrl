"""Certify the value of an EconML CATE-induced policy against hidden confounding.

EconML learns the heterogeneous effect; causalrl bounds the *decision* of acting on the induced
policy (treat iff tau_hat(x) > 0) under Tan's marginal sensitivity model. Run:

    uv run --extra interop python examples/interop_econml_certify.py

Prints a skip message and exits 0 if EconML is not installed, so it is safe in any environment.
"""

from __future__ import annotations

import numpy as np


def main() -> None:
    try:
        from econml.dml import LinearDML
    except Exception as exc:  # optional interop stack absent or broken -> skip cleanly
        print(f"[skip] EconML unavailable ({type(exc).__name__}); pip install causalrl[interop].")
        return

    from causalrl import certify_estimate
    from causalrl.interop.econml import from_econml_cate

    rng = np.random.default_rng(0)
    n = 4000
    x = rng.normal(size=(n, 3))
    f = (rng.random(n) < 0.5).astype(int)
    tau = 0.5 + 0.5 * x[:, 0]  # heterogeneous treatment effect
    y = tau * f + x[:, 1] + rng.normal(0, 0.1, size=n)
    e0 = np.full(n, 0.5)  # uniform logging propensities

    cate = LinearDML(discrete_treatment=True)
    cate.fit(y, f, X=x)

    contrast = from_econml_cate(cate, x, outcomes=y, treated=f, logging_propensities=e0)
    cert = certify_estimate(contrast)
    print("causalrl decision certificate for the EconML CATE-induced policy:")
    print(cert)
    print(f"recommendation: {cert.recommendation}")


if __name__ == "__main__":
    main()
