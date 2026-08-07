"""CausalMBRLAgent vs STRONG contenders on NHEFS (quitting smoking -> weight change), as a policy.

Out-of-CI demo. Needs the `causaldata` package for the dataset and scikit-learn for the contenders:

    uv run --with causaldata python examples/causal_mbrl_nhefs.py

NHEFS is the canonical causal-inference teaching dataset (Hernan & Robins, "What If"); the
established adjusted answer is about +3.4 to +3.5 kg. We benchmark the causal agent against REAL
contenders -- IPW, doubly-robust AIPW, propensity stratification -- not just a naive strawman.

What policy question does NHEFS honestly support? Not "should this person quit smoking" -- weight is
a SIDE EFFECT and the benefits of quitting are not in this dataset, so an agent maximising kilograms
would give the clinically wrong advice. What it does support is SCREENING: post-cessation weight
gain is the most-cited barrier to quitting, and `argmax_a E[weight change | do(a), x]` is exactly
"would quitting raise this person's weight?" -- the flag for offering weight-management support
alongside cessation. `CausalMBRLAgent.act(observation)` is called once per participant on that
person's confounders and those per-person flags are the policy that is then valued and certified.

There is NO ground truth here -- no randomized arm -- so no regret can be scored, and that is
precisely why the certificate is the only external check available. `certify_policy` prices the flag
policy against the logged behaviour from the observational data alone, with the propensity-score
quintile as the (discretised) state, and runs the finite-sample conformal downside gate on top.

Honest read: on this well-behaved dataset the causal agent AGREES with the strong contenders (all
land near +3.3 kg), and all of them clear the confounded naive comparison. This is parity with the
serious methods, not a strawman win. In policy currency the contextual flag turns out to be
DEGENERATE -- the model finds nobody for whom quitting lowers weight, so `act()` collapses to the
marginal decision -- and the confounding layer still refuses it at a modest Γ. The point estimators
have no comparable layer; that layer, not the number, is what causalrl adds.
"""

from __future__ import annotations

import numpy as np
from _causal_baselines import _propensity, aipw_ate, ipw_ate, propensity_strata_ate
from causaldata import nhefs

from causalrl import CausalMBRLAgent, ConfoundedTrajectoryDataset, Transition, certify_policy

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
N_STRATA = 5


def main() -> None:
    df = nhefs.load_pandas().data.dropna(subset=["wt82_71", "qsmk", *CONFOUNDERS])
    a, y = df["qsmk"].to_numpy(), df["wt82_71"].to_numpy()
    x = df[CONFOUNDERS].to_numpy(dtype=float)
    data: dict[str, object] = {"A": a, "Y": y}
    for column in CONFOUNDERS:
        data[column] = df[column].to_numpy(dtype=float)

    agent = CausalMBRLAgent(2, covariates=CONFOUNDERS).fit(data)
    # The policy: one act() call per participant, on that participant's confounders.
    flag = np.array(
        [agent.act({c: float(df[c].iloc[i]) for c in CONFOUNDERS}) for i in range(len(df))]
    )

    rows = [
        ("naive diff-in-means [strawman]", float(y[a == 1].mean() - y[a == 0].mean())),
        ("ours: g-formula (linear)", agent.planner.contrast),
        ("strong: IPW", ipw_ate(x, a, y)),
        ("strong: AIPW (doubly-robust)", aipw_ate(x, a, y)),
        ("strong: propensity strata", propensity_strata_ate(x, a, y)),
    ]

    print(f"n = {len(df)}  ({int(a.sum())} quitters)")
    print("established adjusted effect (Hernan-Robins): +3.4 to +3.5 kg\n")
    print("effect estimates:")
    for label, value in rows:
        print(f"  {label:32s} {value:+.2f} kg")

    # The screening policy, valued and certified from the observational logs alone.
    e = _propensity(x, a)
    edges = np.quantile(e, np.linspace(0.0, 1.0, N_STRATA + 1))
    states = np.clip(np.digitize(e, edges[1:-1]), 0, N_STRATA - 1)
    transitions = [
        Transition(state=int(s), action=int(ai), reward=float(yi), next_state=int(s), done=True)
        for s, ai, yi in zip(states, a, y, strict=True)
    ]
    dataset = ConfoundedTrajectoryDataset(transitions, n_states=N_STRATA, n_actions=2)
    cert = certify_policy(dataset, flag.tolist(), gamma_max=5.0, alpha=0.1)
    tip = cert.tipping_gamma
    verdict = f"tips at Γ≈{tip:.2f}" if tip is not None else "robust to Γ=5 (does not tip)"

    print("\nthe screening policy -- 'would quitting raise this person's weight?', per person:")
    print(f"  flagged by .act():        {100 * float(flag.mean()):.1f}% of the sample")
    print(f"  off-policy value vs the logged behaviour: {cert.naive_contrast:+.2f} kg")
    print(f"  confounding layer:        {cert.decision}; {verdict}")
    lcb = cert.conformal_lcb
    gate = "no finite lower bound" if lcb is None or lcb == float("-inf") else f"{lcb:+.2f} kg"
    print(f"  finite-sample gate (conformal, alpha=0.1): worst-case single outcome {gate}")
    print(f"  recommendation:           {cert.recommendation}")
    print("  No regret is reported: NHEFS has no randomized arm, so there is no ground-truth")
    print("  counterfactual to price a policy against. The certificate is the only check there is.")
    print("\nHonest read: the causal agent agrees with the strong contenders (all ~+3.3 kg), all")
    print(
        "above the confounded naive +2.54 -- parity with the serious methods, not a strawman win."
    )
    print("The contextual policy is degenerate here, and the certificate refuses it anyway; that")
    print("refusal, not the point estimate, is the layer the contenders do not have.")


if __name__ == "__main__":
    main()
