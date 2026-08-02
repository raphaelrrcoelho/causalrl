"""fit_scm on NHEFS: does a learned SCM agree with the targeted g-formula estimate?

Out-of-CI demo. Needs the `causaldata` package:

    uv run --with causaldata python examples/learned_scm_nhefs.py

NHEFS (Hernan & Robins, "What If"): quitting smoking -> weight change; the established adjusted
answer is about +3.4 to +3.5 kg. We fit a whole SCM over {confounders, A, Y} and read the effect
off the model by intervening, then compare against the targeted g-formula estimator the real-data
suite already validated.

The honest expectation is AGREEMENT, not a win. A fitted SCM spends its capacity on the whole joint
distribution while g-formula spends all of it on one contrast, so parity is the good outcome. What
the SCM adds is everything below the comparison: other interventions, and a counterfactual interval,
from the same object -- queries a single-estimand estimator structurally cannot answer.
"""

from __future__ import annotations

from causaldata import nhefs

from causalrl import CausalMBRLAgent, LinearGaussianFit, fit_scm
from causalrl.scm.graph import CausalGraph

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
    data = {"A": df["qsmk"].to_numpy(dtype=float), "Y": df["wt82_71"].to_numpy(dtype=float)}
    for column in CONFOUNDERS:
        data[column] = df[column].to_numpy(dtype=float)

    graph = CausalGraph(
        directed_edges=[(c, "A") for c in CONFOUNDERS]
        + [(c, "Y") for c in CONFOUNDERS]
        + [("A", "Y")]
    )
    families = {node: LinearGaussianFit() for node in [*CONFOUNDERS, "A", "Y"]}
    scm = fit_scm(data, graph=graph, families=families)

    n = 40_000
    treated = float(scm.do({"A": 1.0}).see(n, seed=0)["Y"].mean())
    control = float(scm.do({"A": 0.0}).see(n, seed=0)["Y"].mean())
    learned_ate = treated - control
    gformula = CausalMBRLAgent(2, covariates=CONFOUNDERS).fit(data).planner.contrast

    print(f"learned SCM  E[Y|do(A=1)] - E[Y|do(A=0)] = {learned_ate:+.3f} kg")
    print(f"targeted g-formula (existing suite)      = {gformula:+.3f} kg")
    print(f"agreement gap                            = {abs(learned_ate - gformula):.3f} kg")
    print("\nliterature reference: +3.4 to +3.5 kg")
    print("\nWhat only the fitted SCM can answer -- a dose the data never assigned:")
    print(scm.fit_report.summary())


if __name__ == "__main__":
    main()
