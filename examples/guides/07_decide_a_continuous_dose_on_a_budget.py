"""Task guide 7: choose a continuous action under a wall-clock budget, and know what you got.

The first six guides all decide between arms. This one decides a *dose* -- a real number, from a
continuous range -- which is the ordinary shape of a treatment decision in medicine, pricing,
budgeting and control, and the shape causalrl could not express until intervention domains stopped
being finite value tuples.

Three things have to hold before a number from a model is worth acting on, and this guide runs each
of them in turn:

1. **Does the model describe the regime you are asking about?** `certify_fitted_query` runs the
   outcome's own fitted mechanism against the factual rows in that regime. A model that mispredicts
   them is not to be read as a point estimate there, whatever its overall holdout score says.
2. **What is the best dose, given a budget?** A continuous domain has no arm list, so the decision
   is a search. `AnytimeInterventionSearch` keeps a usable incumbent at every instant and returns it
   when the `Deadline` expires.
3. **Was the search complete?** A search cut off by its clock examined a subset of what it was asked
   to. The certificate says so, rather than presenting a partial sweep as an exhaustive one.

Run: python examples/guides/07_decide_a_continuous_dose_on_a_budget.py
"""

from __future__ import annotations

import numpy as np

from causalrl import (
    CausalGraph,
    Continuous,
    Deadline,
    InterventionSpace,
    certify_fitted_query,
    fit_scm,
)
from causalrl.agents.anytime import AnytimeInterventionSearch

N_ROWS = 4000
SEED = 0
# The true dose-response is single-peaked: too little does nothing, too much harms.
BEST_DOSE = 1.4


def simulate(n: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Logs from a clinic that mostly gives low doses; severity drives both dose and outcome."""
    severity = rng.normal(size=n)
    dose = np.clip(0.8 + 0.4 * severity + rng.normal(scale=0.3, size=n), 0.0, 3.0)
    response = 4.0 - (dose - BEST_DOSE) ** 2 - 0.8 * severity + rng.normal(scale=0.2, size=n)
    return {"severity": severity, "dose": dose, "response": response}


def main() -> None:
    rng = np.random.default_rng(SEED)
    data = simulate(N_ROWS, rng)
    graph = CausalGraph(
        directed_edges=[("severity", "dose"), ("severity", "response"), ("dose", "response")],
        nodes=["severity", "dose", "response"],
    )
    model = fit_scm(data, graph=graph)

    print("1. Is the model trustworthy in the regime we are about to query?")
    for dose in (1.0, 2.9):
        # 2.9 is far out in the tail of what this clinic ever prescribed.
        nearest = float(data["dose"][np.argmin(np.abs(data["dose"] - dose))])
        cert = certify_fitted_query(
            model, data, intervention={"dose": nearest}, outcome="response", atol=0.05
        )
        verdict = "TRUSTED" if cert.hedge is None else f"HEDGED ({cert.hedge.reason.split(':')[0]})"
        print(f"   dose~{dose:.1f}: {verdict}")
    print()

    print("2. Search the continuous dose range for the best response.")
    space = InterventionSpace.create({"dose": Continuous(0.0, 3.0)})

    def value(_observation: dict[str, float], intervention: dict[str, float]) -> float:
        """The model's answer for this dose. A search, because there are no arms to enumerate."""
        drawn = model.do({"dose": float(intervention["dose"])}).see(256, seed=SEED)
        return float(np.asarray(drawn["response"]).mean())

    generous = AnytimeInterventionSearch(value, rounds=6, candidates_per_round=12, seed=SEED)
    chosen = generous.act({}, space=space, deadline=Deadline.after(30.0))
    report = generous.last_search
    print(f"   chose dose={float(chosen['dose']):.3f}  (true optimum {BEST_DOSE})")
    print(f"   {report.rounds}/{report.rounds_planned} rounds, {report.candidates} candidates")
    print(f"   hedge: {report.certificate().hedge}")
    print()

    print("3. The same search on a budget it cannot meet.")
    rushed = AnytimeInterventionSearch(value, rounds=64, candidates_per_round=64, seed=SEED)
    incumbent = rushed.act({}, space=space, deadline=Deadline.after(0.05))
    truncated = rushed.last_search
    hedge = truncated.certificate().hedge
    print(f"   still returned a usable answer: dose={float(incumbent['dose']):.3f}")
    print(
        f"   {truncated.rounds}/{truncated.rounds_planned} rounds -- "
        f"exhausted={truncated.exhausted}"
    )
    assert hedge is not None, "a truncated search must not present itself as exhaustive"
    print(f"   hedge: {hedge.reason[:72]}...")
    print(f"   downgraded_from: {hedge.downgraded_from}")


if __name__ == "__main__":
    main()
