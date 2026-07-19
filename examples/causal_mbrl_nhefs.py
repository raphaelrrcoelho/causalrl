"""CausalMBRLAgent vs STRONG contenders on NHEFS (quitting smoking -> weight change).

Out-of-CI demo. Needs the `causaldata` package for the dataset and scikit-learn for the contenders:

    uv run --with causaldata python examples/causal_mbrl_nhefs.py

NHEFS is the canonical causal-inference teaching dataset (Hernan & Robins, "What If"); the
established adjusted answer is about +3.4 to +3.5 kg. We benchmark the causal agent against REAL
contenders -- IPW, doubly-robust AIPW, propensity stratification -- not just a naive strawman.

Honest read: on this well-behaved dataset the causal agent AGREES with the strong contenders (all
land near +3.3 kg), and all of them clear the confounded naive comparison. This is parity with the
serious methods, not a strawman win -- the honest baseline before the certificate layer (which the
point estimators lack) adds value on harder, fragile problems.
"""

from __future__ import annotations

from _causal_baselines import aipw_ate, ipw_ate, propensity_strata_ate
from causaldata import nhefs

from causalrl import CausalMBRLAgent

CONFOUNDERS = [
    "sex",
    "age",
    "race",
    "education",
    "smokeintensity",
    "smokeyrs",
    "exercise",
    "active",
    "wt71",
]


def main() -> None:
    df = nhefs.load_pandas().data.dropna(subset=["wt82_71", "qsmk", *CONFOUNDERS])
    a, y = df["qsmk"].to_numpy(), df["wt82_71"].to_numpy()
    x = df[CONFOUNDERS].to_numpy(dtype=float)
    data: dict[str, object] = {"A": a, "Y": y}
    for column in CONFOUNDERS:
        data[column] = df[column].to_numpy(dtype=float)

    gformula = CausalMBRLAgent(2, covariates=CONFOUNDERS).fit(data).planner.contrast
    rows = [
        ("naive diff-in-means [strawman]", float(y[a == 1].mean() - y[a == 0].mean())),
        ("ours: g-formula (linear)", gformula),
        ("strong: IPW", ipw_ate(x, a, y)),
        ("strong: AIPW (doubly-robust)", aipw_ate(x, a, y)),
        ("strong: propensity strata", propensity_strata_ate(x, a, y)),
    ]

    print(f"n = {len(df)}  ({int(a.sum())} quitters)")
    print("established adjusted effect (Hernan-Robins): +3.4 to +3.5 kg\n")
    for label, value in rows:
        print(f"{label:32s} {value:+.2f} kg")
    print("\nHonest read: the causal agent agrees with the strong contenders (all ~+3.3 kg), all")
    print(
        "above the confounded naive +2.54 -- parity with the serious methods, not a strawman win."
    )


if __name__ == "__main__":
    main()
