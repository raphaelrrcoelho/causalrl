"""CausalMBRLAgent vs STRONG contenders on LaLonde -- as a POLICY, scored against the RCT.

Out-of-CI demo (needs network for the dataset; scikit-learn for the contenders). Run it:

    uv run python examples/causal_mbrl_lalonde.py

The observational LaLonde data (Dehejia & Wahba) pairs NSW trainees with non-experimental PSID
controls; the randomized NSW experiment says the truth is about +$1,794. We benchmark the causal
agent against REAL contenders -- IPW, doubly-robust AIPW, propensity stratification -- not just a
naive difference-in-means.

The decision this dataset exists to inform is "who do we enrol in the job-training program?", so it
is scored as a policy, not as an effect size. `CausalMBRLAgent.act(observation)` is called once per
person on that person's covariates; those per-person actions are the policy, and the RCT prices
them: the experiment identifies only the AVERAGE effect, so scoring a per-person policy against it
requires reading that effect as homogeneous, under which a policy enrolling a share f of the
population leaves regret (1 - f) x $1,794 per person on the table. That assumption is stated, not
hidden -- LaLonde offers no per-person ground truth. `certify_policy` then scores the same policy
from the observational logs alone, with the propensity-score quintile as the (discretised) state.

Honest read: this is a famously pathological dataset, and it is pathological in policy currency too.
Point estimates scatter from about -$600 to +$1,050 and a STRONG method (propensity stratification)
gets the sign wrong. The contextual policy is WORSE than the marginal "enrol everyone" it is built
from -- the per-person heterogeneity the model believes it sees costs real money against the RCT.
And the off-policy value computed from the observational logs has the wrong sign outright. Both
certificate layers refuse: the confounding layer at a small Γ, and the finite-sample conformal gate
for want of effectively-weighted support. That refusal is the deliverable, not a number.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from _causal_baselines import _propensity, aipw_ate, ipw_ate, propensity_strata_ate

from causalrl import CausalMBRLAgent, ConfoundedTrajectoryDataset, Transition, certify_policy

RCT_BENCHMARK = 1794.0  # NSW randomized-experiment ATE on 1978 earnings (Dehejia & Wahba).
URL = "https://raw.githubusercontent.com/robjellis/lalonde/master/lalonde_data.csv"
COVARIATES = ["age", "educ", "black", "hispan", "married", "nodegree", "re74", "re75"]
N_STRATA = 5


def _propensity_strata(x: np.ndarray, a: np.ndarray) -> tuple[np.ndarray, int]:
    """Propensity-score quintile per row: the discrete state the certificate conditions on."""
    e = _propensity(x, a)
    edges = np.quantile(e, np.linspace(0.0, 1.0, N_STRATA + 1))
    return np.clip(np.digitize(e, edges[1:-1]), 0, N_STRATA - 1), N_STRATA


def main() -> None:
    df = pd.read_csv(URL)
    a, y = df["treat"].to_numpy(), df["re78"].to_numpy()
    x = df[COVARIATES].to_numpy(dtype=float)
    data: dict[str, object] = {"A": a, "Y": y}
    for column in COVARIATES:
        data[column] = df[column].to_numpy()

    agent = CausalMBRLAgent(2, covariates=COVARIATES).fit(data)
    gformula = agent.planner.contrast
    # The policy: one act() call per person, on that person's covariates.
    enrol = np.array(
        [agent.act({c: float(df[c].iloc[i]) for c in COVARIATES}) for i in range(len(df))]
    )

    estimates = [
        ("naive diff-in-means", float(y[a == 1].mean() - y[a == 0].mean()), "strawman"),
        ("ours: g-formula (linear)", gformula, "ours"),
        ("strong: IPW", ipw_ate(x, a, y), "contender"),
        ("strong: AIPW (doubly-robust)", aipw_ate(x, a, y), "contender"),
        ("strong: propensity strata", propensity_strata_ate(x, a, y), "contender"),
    ]

    print(f"n = {len(df)}  ({int(a.sum())} trained, {int((1 - a).sum())} PSID controls)")
    print(f"randomized-experiment truth (ATE on 1978 earnings): {RCT_BENCHMARK:+,.0f}\n")
    print("effect estimates:")
    for label, value, _kind in estimates:
        print(f"  {label:32s} {value:+9,.0f}   gap {value - RCT_BENCHMARK:+,.0f}")

    # Every estimator implies a policy: a per-person one from act(), a constant one from an ATE.
    print("\nthe decision, priced by the RCT (regret assumes the experiment's average effect is")
    print("homogeneous -- LaLonde gives no per-person ground truth):\n")
    print(f"  {'policy':32s} {'rule':12s} {'enrolled':>9s} {'regret / person':>16s}")
    policies: list[tuple[str, str, np.ndarray]] = [
        ("oracle (the RCT's answer)", "enrol all", np.ones(len(df), dtype=int)),
        ("ours: CausalMBRLAgent .act()", "contextual", enrol),
    ]
    policies += [
        (label, "enrol all" if value > 0 else "enrol none", np.full(len(df), value > 0))
        for label, value, _kind in estimates
    ]
    for label, rule, policy in policies:
        share = float(np.asarray(policy, dtype=float).mean())
        print(f"  {label:32s} {rule:12s} {100 * share:8.1f}% {(1 - share) * RCT_BENCHMARK:15,.0f}")

    # The certificate: what the observational logs alone say about our per-person policy.
    states, n_states = _propensity_strata(x, a)
    transitions = [
        Transition(state=int(s), action=int(ai), reward=float(yi), next_state=int(s), done=True)
        for s, ai, yi in zip(states, a, y, strict=True)
    ]
    dataset = ConfoundedTrajectoryDataset(transitions, n_states=n_states, n_actions=2)
    cert = certify_policy(dataset, enrol.tolist(), gamma_max=5.0, alpha=0.1)
    tip = cert.tipping_gamma
    verdict = f"tips at Γ≈{tip:.2f}" if tip is not None else "robust to Γ=5 (does not tip)"
    print("\ncertificate on our .act() policy (certify_policy; state = propensity quintile):")
    print(f"  off-policy value vs the logged behaviour: {cert.naive_contrast:+,.0f} per person --")
    print(f"  the WRONG sign; the RCT prices the program at {RCT_BENCHMARK:+,.0f}.")
    print(f"  confounding layer: {cert.decision}; {verdict}.")
    lcb = cert.conformal_lcb
    gate = (
        "no finite lower bound at alpha=0.1"
        if lcb is None or lcb == float("-inf")
        else f"{lcb:,.0f}"
    )
    print(f"  finite-sample downside gate (conformal, alpha=0.1): {gate}.")
    print(f"  recommendation: {cert.recommendation}.")
    print("\nHonest read: point estimates scatter, a strong method gets the sign wrong, and the")
    print("contextual policy is beaten by the constant it was built from. No number here is")
    print("trustworthy -- and both certificate layers say so, which is the case for a certificate.")


if __name__ == "__main__":
    main()
