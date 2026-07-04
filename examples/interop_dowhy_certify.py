"""Certify a DoWhy propensity-weighting estimate against hidden confounding.

DoWhy estimates the effect; causalrl bounds the *decision* — does the ship/abstain call survive
hidden confounding? Run with the optional interop dependencies:

    uv run --extra interop python examples/interop_dowhy_certify.py

Prints a skip message and exits 0 if DoWhy is not installed, so it is safe in any environment.
"""

from __future__ import annotations

import numpy as np


def main() -> None:
    try:
        import pandas as pd
        from dowhy import CausalModel
    except Exception as exc:  # optional interop stack absent or broken -> skip cleanly
        print(f"[skip] DoWhy unavailable ({type(exc).__name__}); pip install causalrl[interop].")
        return

    from causalrl import certify_estimate
    from causalrl.interop.dowhy import from_dowhy_estimate

    rng = np.random.default_rng(0)
    n = 4000
    z = rng.integers(0, 2, size=n)
    f = (rng.random(n) < 0.3 + 0.3 * z).astype(int)  # Z confounds treatment assignment
    y = 0.5 * f + 1.0 * z + rng.normal(0, 0.1, size=n)
    df = pd.DataFrame({"f": f, "y": y, "z": z})

    model = CausalModel(data=df, treatment="f", outcome="y", common_causes=["z"])
    estimand = model.identify_effect(proceed_when_unidentifiable=True)
    estimate = model.estimate_effect(estimand, method_name="backdoor.propensity_score_weighting")
    print(f"DoWhy point estimate (ATE): {estimate.value:+.3f}")

    try:
        contrast = from_dowhy_estimate(estimate, outcomes=y, treated=f, confounder_bins=z)
        source = "from the DoWhy estimate"
    except TypeError:
        # This DoWhy build does not surface propensity_scores on the estimate; fit them explicitly.
        from sklearn.linear_model import LogisticRegression

        from causalrl import PolicyValueContrast

        zz = z.reshape(-1, 1)
        e0 = LogisticRegression().fit(zz, f).predict_proba(zz)[:, 1]
        contrast = PolicyValueContrast.from_binary(y, f, propensities=e0, confounder_bins=z)
        source = "propensities fit explicitly with sklearn"

    cert = certify_estimate(contrast, labels=("treated", "control"))
    print(f"causalrl decision certificate ({source}):")
    print(cert)
    print(f"recommendation: {cert.recommendation}")


if __name__ == "__main__":
    main()
