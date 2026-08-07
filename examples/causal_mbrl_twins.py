"""CausalMBRLAgent vs the top ITE meta-learners on Twins, scored as a POLICY (bandit regret).

Out-of-CI demo (network + scikit-learn). Run:

    uv run python examples/causal_mbrl_twins.py

Twins (Louizos et al.) pairs same-sex twins; treatment = being the heavier twin, outcome = 1-year
mortality. Because BOTH twins are observed we know both potential outcomes -- the exact individual
outcome under either arm (ground truth). We observe one twin per pair, assigned by a fair coin, so
this is literally a two-armed contextual bandit with known propensity 1/2 -- and the question an RL
practitioner actually asks is not "how accurate is your CATE" but "how good is the POLICY your CATE
induces". So the headline here is off-policy value and regret, not PEHE.

The outcome is recoded to SURVIVAL (1 - mortality) so that higher is better and the agent's
`argmax_a E[Y | do(a), x]` points the clinically right way. `GFormulaBackdoorAgent.act(observation)`
is called once per pair on that pair's covariates; those per-unit actions ARE the policy, and are
what both the ground-truth value and `certify_policy` are computed on. The certificate's state is a
DISCRETISATION: gestation decile (`gestat10`), the one already-discrete covariate in the data.

Honest read: individual counterfactuals are something DRL structurally cannot do -- but on real,
sparse, binary mortality the induced policies are near-worthless. Every learned policy (ours and the
top meta-learners) is beaten or matched by the trivial constant "always the heavier twin", and ours
is the worst of them. The certificate is the honest part: the off-policy IPS contrast OVERSTATES the
improvement roughly twofold, and `certify_policy` refuses the policy at a very small Γ. An honest
null in policy currency, benchmarked against the top techniques (not a strawman).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from _causal_baselines import slearner_cate, xlearner_cate

from causalrl import (
    ConfoundedTrajectoryDataset,
    GFormulaBackdoorAgent,
    Transition,
    certify_policy,
)

BASE = "https://raw.githubusercontent.com/AMLab-Amsterdam/CEVAE/master/datasets/TWINS/"


def main() -> None:
    t = pd.read_csv(BASE + "twin_pairs_T_3years_samesex.csv", index_col=0)
    xdf = pd.read_csv(BASE + "twin_pairs_X_3years_samesex.csv", index_col=0)
    ydf = pd.read_csv(BASE + "twin_pairs_Y_3years_samesex.csv", index_col=0)

    w0, w1 = t["dbirwt_0"].to_numpy(), t["dbirwt_1"].to_numpy()
    y0, y1 = ydf["mort_0"].to_numpy(), ydf["mort_1"].to_numpy()
    keep = (w0 < 2000) & (w1 < 2000)  # low-birth-weight pairs: a meaningful mortality signal
    heavier_is_1 = w1 >= w0
    # survival, so that "higher is better" and the agent's argmax is the right direction.
    survival_heavier = 1.0 - np.where(heavier_is_1, y1, y0)[keep]
    survival_lighter = 1.0 - np.where(heavier_is_1, y0, y1)[keep]
    true_ite = survival_heavier - survival_lighter  # exact per-pair effect on survival

    numeric = xdf.select_dtypes(include="number").apply(pd.to_numeric, errors="coerce")
    filled = numeric.fillna(numeric.median()).fillna(0.0)
    x = filled.to_numpy()[keep]
    cols = [f"x{i}" for i in range(x.shape[1])]
    gestation = filled["gestat10"].to_numpy()[keep].astype(int)  # the discrete state

    rng = np.random.default_rng(0)
    observed_heavier = (rng.random(len(true_ite)) < 0.5).astype(
        int
    )  # random -> unconfounded, e=1/2
    y_obs = np.where(observed_heavier == 1, survival_heavier, survival_lighter)

    data: dict[str, object] = {"A": observed_heavier, "Y": y_obs}
    for i, column in enumerate(cols):
        data[column] = x[:, i]
    agent = GFormulaBackdoorAgent(2, covariates=cols).fit(data)

    # The policy: one act() call per pair, on that pair's covariates. These actions are the policy.
    ours = np.array(
        [agent.act({c: float(x[i, j]) for j, c in enumerate(cols)}) for i in range(len(y_obs))]
    )

    def value(policy: np.ndarray) -> float:
        """True 1-year survival under ``policy``, using both known potential outcomes."""
        return float(np.mean(np.where(policy == 1, survival_heavier, survival_lighter)))

    def pehe(estimate: np.ndarray) -> float:
        return float(np.sqrt(np.mean((estimate - true_ite) ** 2)))

    n = len(true_ite)
    oracle = np.where(true_ite > 0, 1, 0)
    s_cate, x_cate = (
        slearner_cate(x, observed_heavier, y_obs),
        xlearner_cate(x, observed_heavier, y_obs),
    )
    policies = [
        ("oracle (best arm per pair)", oracle, None),
        ("strong: X-learner sign policy", (x_cate > 0).astype(int), pehe(x_cate)),
        ("strong: S-learner sign policy", (s_cate > 0).astype(int), pehe(s_cate)),
        ("constant: always heavier twin", np.ones(n, dtype=int), pehe(np.full(n, agent.contrast))),
        ("ours: GFormula .act() per pair", ours, pehe(agent.cate(data))),
        ("logged behaviour (coin flip)", observed_heavier, None),
        ("constant: always lighter twin", np.zeros(n, dtype=int), None),
    ]

    print(f"n = {n} low-birth-weight twin pairs (a 2-armed bandit, known propensity 1/2)")
    print(f"true ATE on SURVIVAL (heavier twin): {true_ite.mean():+.4f}  (exact ground truth)\n")
    print("policy value = true 1-year survival under the induced policy (higher is better);")
    print("regret = oracle - value; PEHE on the per-pair effect is shown where defined.")
    print("(the oracle's low '% heavier' is tie-breaking: both twins survive in most pairs.)\n")
    print(f"  {'policy':32s} {'value':>7s} {'regret':>8s} {'% heavier':>10s} {'PEHE':>7s}")
    best = value(oracle)
    for label, policy, score in policies:
        share = f"{100 * float(policy.mean()):.1f}%"
        pehe_cell = "     --" if score is None else f"{score:7.4f}"
        print(
            f"  {label:32s} {value(policy):7.4f} {best - value(policy):8.4f} "
            f"{share:>10s} {pehe_cell}"
        )

    # The certificate on OUR policy, from the logs alone -- what you could know without the oracle.
    bins = sorted(set(int(g) for g in gestation))
    index = {b: i for i, b in enumerate(bins)}
    transitions = [
        Transition(
            state=index[int(g)], action=int(a), reward=float(r), next_state=index[int(g)], done=True
        )
        for g, a, r in zip(gestation, observed_heavier, y_obs, strict=True)
    ]
    dataset = ConfoundedTrajectoryDataset(transitions, n_states=len(bins), n_actions=2)
    cert = certify_policy(dataset, ours.tolist(), gamma_max=5.0)
    tip = cert.tipping_gamma
    verdict = f"tips at Γ≈{tip:.2f}" if tip is not None else "robust to Γ=5 (does not tip)"
    print("\ncertificate on our .act() policy (certify_policy; state = gestation decile):")
    print(f"  off-policy contrast vs the logged behaviour {cert.naive_contrast:+.4f} survival,")
    print(f"  but the TRUE improvement is only {value(ours) - value(observed_heavier):+.4f}.")
    print(f"  {cert.decision}; {verdict}; recommendation: {cert.recommendation}.")
    print("\nHonest read: in policy currency the null is sharper than in PEHE -- every learned")
    print("policy is matched or beaten by the trivial 'always the heavier twin', and ours is the")
    print("worst of them. The off-policy estimate overstates our improvement about twofold; the")
    print(
        "certificate refuses it. The population ATE is recoverable, the per-unit decision is not."
    )


if __name__ == "__main__":
    main()
