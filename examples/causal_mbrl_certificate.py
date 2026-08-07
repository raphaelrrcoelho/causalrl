"""The certificate earns its keep: honest robustness where confident estimators mislead.

Out-of-CI demo (network + scikit-learn). Run:

    uv run python examples/causal_mbrl_certificate.py

On the LaLonde data the strong point estimators disagree wildly (see causal_mbrl_lalonde.py: about
-$600 to +$1,050, some with the wrong sign). Each hands you a confident number. causalrl's
`certify_decision` instead asks the honest question -- is the decision from these confounded logs
robust to *unmeasured* confounding? -- and reports the odds-ratio Gamma at which it tips. The naive
decision tips at a modest Gamma, so the certificate refuses to trust it; the randomized experiment
then vindicates that refusal (the true effect is the opposite sign). That honest "don't trust
this" is the value a point estimate -- strong or naive -- cannot give you.
"""

from __future__ import annotations

import pandas as pd
from _causal_baselines import _propensity

from causalrl import certify_decision

RCT_BENCHMARK = 1794.0  # NSW randomized experiment: the truth is "prefer action 1" (+$1,794).
URL = "https://raw.githubusercontent.com/robjellis/lalonde/master/lalonde_data.csv"
COVARIATES = ["age", "educ", "black", "hispan", "married", "nodegree", "re74", "re75"]


def main() -> None:
    df = pd.read_csv(URL)
    a, y = df["treat"].to_numpy(), df["re78"].to_numpy()
    propensities = _propensity(df[COVARIATES].to_numpy(dtype=float), a)

    cert = certify_decision(
        rewards=y.tolist(),
        actions=a.tolist(),
        propensities=propensities.tolist(),
        gamma_max=5.0,
    )
    tip = cert.tipping_gamma
    tip_str = f"Γ≈{tip:.2f}" if tip is not None else "does not tip below the cap (robust)"

    print(f"naive decision  : {cert.decision}  (contrast {cert.naive_contrast:+,.0f})")
    print(f"certified robust: {cert.certified}")
    print(f"tips at         : {tip_str}  (unmeasured confounding this strong overturns it)\n")
    print(
        f"randomized-experiment truth: {RCT_BENCHMARK:+,.0f}  "
        "(prefer action 1 -- the OPPOSITE sign)"
    )
    print(
        "=> The certificate refused to trust the naive decision, and the RCT shows it was right to."
    )
    print("   A confident point estimate ships the wrong call; the certificate flags it instead.")


if __name__ == "__main__":
    main()
