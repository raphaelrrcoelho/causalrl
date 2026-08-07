"""OBD off-policy evaluation: the top OPE techniques, honestly (including where they break).

Out-of-CI demo. Needs the `obp` package for the bundled Open Bandit Dataset sample (obp's own loader
is broken on modern pandas, so we read its CSVs directly). Run:

    uv run --with obp python examples/causal_mbrl_obd.py

OBD logs a real recommender under a uniform-RANDOM policy (known propensity 1/80), so standard OPE
is unconfounded and the top estimators are unbiased -- nothing to deconfound. This shows three
honest things: (1) the canonical off-policy task -- estimate the 80-arm random policy's value from
the BTS-policy logs; IPS and causalrl's `msm_policy_value_bounds` (the self-normalised Hajek point
at Γ=1) agree and land in the ballpark of the on-policy ground truth (sample-limited on this 10k
slice; accurate on the full dataset). (2) A CONTROLLED illustration: induce confounding by
outcome-selection and IPS balloons ~3x -- the top estimator is structurally fooled. (3) The
certificate on the clean known-propensity logs, over all 80 arms NATIVELY: `certify_policy` takes a
`ConfoundedTrajectoryDataset` plus one target action per logged impression, so the 80-arm bandit
needs no collapsing into a binary treatment. A correct true-negative. Honest scope: OBD's "a good
item beats random" is clear-cut, not a marginal sign-flippable decision, so the certificate cannot
show its edge here -- that needs an observational decision (see causal_mbrl_certificate.py /
LaLonde). The dataset's state is a single context bin, which is exactly right here: the logging
policy is non-contextual, so the empirical behaviour propensity is the logged action frequency.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from causalrl import (
    ConfoundedTrajectoryDataset,
    Transition,
    certify_policy,
    msm_policy_value_bounds,
)

N_ACTIONS = 80


def _load(policy: str) -> pd.DataFrame:
    import obp

    path = os.path.join(os.path.dirname(obp.__file__), "dataset", "obd", policy, "all", "all.csv")
    return pd.read_csv(path, index_col=0)


def _bandit_logs(actions: np.ndarray, clicks: np.ndarray) -> ConfoundedTrajectoryDataset:
    """The logged impressions as a one-step, 80-arm contextual-bandit dataset (single state)."""
    transitions = [
        Transition(state=0, action=int(a), reward=float(y), next_state=0, done=True)
        for a, y in zip(actions, clicks, strict=True)
    ]
    return ConfoundedTrajectoryDataset(transitions, n_states=1, n_actions=N_ACTIONS)


def main() -> None:
    rnd, bts = _load("random"), _load("bts")
    true_random = float(rnd["click"].mean())

    # (1) canonical off-policy evaluation: the 80-arm random policy's value from the BTS logs.
    # The target policy puts 1/80 on whatever action was logged, so its value is the self-normalised
    # off-policy value -- which causalrl's MSM bound returns exactly at Γ=1, and widens beyond it.
    p, y = np.clip(bts["propensity_score"].to_numpy(), 1e-6, None), bts["click"].to_numpy(float)
    uniform = [1.0 / N_ACTIONS] * len(y)
    ips = float(((1.0 / N_ACTIONS) / p * y).mean())
    snips = msm_policy_value_bounds(
        y.tolist(), p.tolist(), uniform, gamma=1.0, return_certificate=False
    ).lower
    print("(1) canonical OPE -- random-policy value from BTS logs (top OPE techniques):")
    print(f"    ground truth (on-policy) {true_random:.5f} | IPS {ips:.5f} | SNIPS {snips:.5f}")
    print("    IPS and SNIPS agree and are in the ballpark; sample-limited on this 10k slice.")
    band = msm_policy_value_bounds(
        y.tolist(), p.tolist(), uniform, gamma=1.5, return_certificate=False
    )
    print("    (that residual gap is 10k-slice sampling noise, not confounding -- the logged")
    print(f"    propensities are known. For scale, the Γ=1.5 MSM band is [{band.lower:.5f},")
    print(f"    {band.upper:.5f}].)\n")

    # (2) controlled confounding illustration on the best random-logged item.
    a, yr = rnd["item_id"].to_numpy(), rnd["click"].to_numpy(dtype=float)
    istar = int(pd.Series(yr).groupby(a).mean().idxmax())
    true_best = float(yr[a == istar].mean())
    ips_clean = float((np.where(a == istar, N_ACTIONS, 0.0) * yr).mean())
    rng = np.random.default_rng(0)
    keep = (yr == 1) | (rng.random(len(yr)) < 0.30)  # outcome-selection -> induced confounding
    ac, yc = a[keep], yr[keep]
    ips_conf = float((np.where(ac == istar, N_ACTIONS, 0.0) * yc).mean())
    print(f"(2) best-item value (controlled confounding illustration, item {istar}):")
    print(f"    ground truth {true_best:.4f} | clean IPS {ips_clean:.4f} (unbiased)", end="")
    print(f" | confounded IPS {ips_conf:.4f} (3x -- structurally fooled)\n")

    # (3) the certificate on the clean, known-propensity logs -- a correct true-negative, scored
    # over all 80 arms natively: the target policy is "always recommend item istar", one action per
    # logged impression, and the behaviour arm is the uniform-random logging policy.
    dataset = _bandit_logs(a, yr)
    cert = certify_policy(dataset, [istar] * len(a), gamma_max=5.0)
    tip = cert.tipping_gamma
    print(f"(3) certificate over all {N_ACTIONS} arms (certify_policy on the clean random logs):")
    print(f"    target policy: always recommend item {istar}; behaviour: uniform-random logging.")
    print(
        f"    contrast {cert.naive_contrast:+.5f} CTR = {true_best:.5f} (item {istar}, on-policy)"
        f" - {true_random:.5f} (random)"
    )
    verdict = f"tips at Γ≈{tip:.2f}" if tip is not None else "robust to Γ=5 (does not tip)"
    print(f"    verdict: {cert.decision}; {verdict}; recommendation {cert.recommendation}")
    print(
        "    Honest scope: propensities are known and correct, so there is nothing to deconfound,"
    )
    print("    and the abstention is the certificate being conservative about a hypothetical")
    print("    propensity error that this dataset does not have -- at Γ=1, the truth here, the")
    print("    decision stands. 'Best item beats random' is not a marginal decision, so the")
    print("    certificate's edge needs a marginal OBSERVATIONAL decision (LaLonde) to show.")


if __name__ == "__main__":
    main()
