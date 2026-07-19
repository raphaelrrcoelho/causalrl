"""Put CausalMBRLAgent on a real confounded decision: the LaLonde job-training program.

Out-of-CI demo (needs network for the dataset; scikit-learn is optional). Run it:

    uv run python examples/causal_mbrl_lalonde.py

The observational LaLonde data (Dehejia & Wahba) pairs NSW *trainees* with non-experimental PSID
controls, so the naive treated-minus-control earnings comparison is negative -- training looks
harmful. The randomized NSW experiment says the truth is about +$1,794. A correlational agent kills
the program; the causal agent back-door-adjusts (g-formula standardization over the observed
covariates) and recovers the decision -- the confounded/offline regime where causal beats naive.

Note (honest): on this famously hard dataset the estimate is model-sensitive -- the linear g-formula
recovers the right sign near the experimental truth, while a flexible gradient-boosted model does
not. Both are printed; the sign-flip to the correct decision is the robust claim, not the exact kg.
"""

from __future__ import annotations

import pandas as pd

from causalrl import CausalMBRLAgent

RCT_BENCHMARK = 1794.0  # NSW randomized-experiment ATE on 1978 earnings (Dehejia & Wahba).
URL = "https://raw.githubusercontent.com/robjellis/lalonde/master/lalonde_data.csv"
COVARIATES = ["age", "educ", "black", "hispan", "married", "nodegree", "re74", "re75"]


def _contrast(data: dict[str, object], outcome_model: object = None) -> float:
    agent = CausalMBRLAgent(2, covariates=COVARIATES, outcome_model=outcome_model).fit(data)
    return agent.planner.contrast


def main() -> None:
    df = pd.read_csv(URL)
    data: dict[str, object] = {"A": df["treat"].to_numpy(), "Y": df["re78"].to_numpy()}
    for column in COVARIATES:
        data[column] = df[column].to_numpy()

    y, a = df["re78"].to_numpy(), df["treat"].to_numpy()
    rows = [
        ("randomized-experiment truth", RCT_BENCHMARK),
        ("naive correlational (confounded)", float(y[a == 1].mean() - y[a == 0].mean())),
        ("causal g-formula (linear)", _contrast(data)),
    ]
    try:
        from sklearn.ensemble import GradientBoostingRegressor

        rows.append(
            ("causal g-formula (boosted)", _contrast(data, lambda: GradientBoostingRegressor()))
        )
    except ImportError:
        pass

    print(f"n = {len(df)}  ({int(a.sum())} trained, {int((1 - a).sum())} PSID controls)\n")
    for label, value in rows:
        gap = "" if label.endswith("truth") else f"   gap {value - RCT_BENCHMARK:+,.0f}"
        decision = "ASSIGN training" if value > 0 else "kill the program"
        print(f"{label:34s} {value:+9,.0f}   [{decision}]{gap}")


if __name__ == "__main__":
    main()
