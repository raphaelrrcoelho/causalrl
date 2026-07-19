"""Put CausalMBRLAgent on a real medical dataset: NHEFS (quitting smoking -> weight change).

Out-of-CI demo. Needs the `causaldata` package for the dataset. Run it:

    uv run --with causaldata python examples/causal_mbrl_nhefs.py

NHEFS is the canonical causal-inference teaching dataset (Hernan & Robins, "What If"). People who
quit smoking differ systematically from those who don't, so the crude quit-vs-not weight comparison
is confounded. The established adjusted answer is about +3.4 to +3.5 kg. The agent's g-formula
standardization over the standard confounder set recovers it on real medical data, where the crude
estimate is biased low -- the confounded regime where causal structure beats correlation.
"""

from __future__ import annotations

from causaldata import nhefs

from causalrl.agents.mbrl import GFormulaBackdoorAgent

TEXTBOOK = "+3.4 to +3.5 kg (Hernan-Robins IP-weighting / standardization)"
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
    data: dict[str, object] = {"A": df["qsmk"].to_numpy(), "Y": df["wt82_71"].to_numpy()}
    for column in CONFOUNDERS:
        data[column] = df[column].to_numpy(dtype=float)

    y, a = df["wt82_71"].to_numpy(), df["qsmk"].to_numpy()
    crude = float(y[a == 1].mean() - y[a == 0].mean())
    agent = GFormulaBackdoorAgent(2, covariates=CONFOUNDERS).fit(data)

    print(f"n = {len(df)}  ({int(a.sum())} quitters)\n")
    print(f"established adjusted effect:   {TEXTBOOK}")
    print(f"crude (naive), confounded:    {crude:+.2f} kg")
    print(f"causal g-formula (linear):    {agent.contrast:+.2f} kg")


if __name__ == "__main__":
    main()
