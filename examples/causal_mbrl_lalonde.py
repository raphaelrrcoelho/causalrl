"""CausalMBRLAgent vs STRONG contenders on the LaLonde job-training program.

Out-of-CI demo (needs network for the dataset; scikit-learn for the contenders). Run it:

    uv run python examples/causal_mbrl_lalonde.py

The observational LaLonde data (Dehejia & Wahba) pairs NSW trainees with non-experimental PSID
controls; the randomized NSW experiment says the truth is about +$1,794. We benchmark the causal
agent against REAL contenders -- IPW, doubly-robust AIPW, propensity stratification -- not just a
naive difference-in-means.

Honest read: this is a famously pathological dataset. The naive strawman gets the sign wrong, and so
do some STRONG methods (propensity stratification lands negative here); estimates scatter from about
-$600 to +$1,050. No single point estimate is trustworthy -- which is the argument for a
confounding certificate over a confident number, not for trusting any one estimator (see
causal_mbrl_certificate.py).
"""

from __future__ import annotations

import pandas as pd
from _causal_baselines import aipw_ate, ipw_ate, propensity_strata_ate

from causalrl import CausalMBRLAgent

RCT_BENCHMARK = 1794.0  # NSW randomized-experiment ATE on 1978 earnings (Dehejia & Wahba).
URL = "https://raw.githubusercontent.com/robjellis/lalonde/master/lalonde_data.csv"
COVARIATES = ["age", "educ", "black", "hispan", "married", "nodegree", "re74", "re75"]


def main() -> None:
    df = pd.read_csv(URL)
    a, y = df["treat"].to_numpy(), df["re78"].to_numpy()
    x = df[COVARIATES].to_numpy(dtype=float)
    data: dict[str, object] = {"A": a, "Y": y}
    for column in COVARIATES:
        data[column] = df[column].to_numpy()

    gformula = CausalMBRLAgent(2, covariates=COVARIATES).fit(data).planner.contrast
    rows = [
        ("randomized-experiment truth", RCT_BENCHMARK, "target"),
        ("naive diff-in-means", float(y[a == 1].mean() - y[a == 0].mean()), "strawman"),
        ("ours: g-formula (linear)", gformula, "ours"),
        ("strong: IPW", ipw_ate(x, a, y), "contender"),
        ("strong: AIPW (doubly-robust)", aipw_ate(x, a, y), "contender"),
        ("strong: propensity strata", propensity_strata_ate(x, a, y), "contender"),
    ]

    print(f"n = {len(df)}  ({int(a.sum())} trained, {int((1 - a).sum())} PSID controls)\n")
    for label, value, kind in rows:
        gap = "" if kind == "target" else f"   gap {value - RCT_BENCHMARK:+,.0f}"
        decision = "assign" if value > 0 else "KILL  "
        print(f"{label:32s} {value:+9,.0f}  [{decision}]{gap}")
    print(
        "\nHonest read: point estimates scatter and even a strong method (propensity strata) gets"
    )
    print("the sign wrong. No single number is trustworthy here -- the case for a certificate.")


if __name__ == "__main__":
    main()
