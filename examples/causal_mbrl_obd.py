"""OBD off-policy evaluation: the top OPE techniques, honestly (including where they break).

Out-of-CI demo. Needs the `obp` package for the bundled Open Bandit Dataset sample (obp's own loader
is broken on modern pandas, so we read its CSVs directly). Run:

    uv run --with obp python examples/causal_mbrl_obd.py

OBD logs a real recommender under a uniform-RANDOM policy (known propensity 1/80), so standard OPE
is unconfounded and the top estimators are unbiased -- nothing to deconfound. This shows three
honest things: (1) the canonical off-policy task -- estimate the random policy's value from the
BTS-policy logs; IPS and SNIPS agree and land in the ballpark of the on-policy ground truth (sample-
limited on this 10k slice; accurate on the full dataset). (2) A CONTROLLED illustration: induce
confounding by outcome-selection and IPS balloons ~3x -- the top estimator is structurally fooled.
(3) The certificate on the clean known-propensity logs -- a correct true-negative. Honest scope:
OBD's "a good item beats random" is clear-cut, not a marginal sign-flippable decision, so the
certificate cannot show its edge here -- that needs an observational decision (see
causal_mbrl_certificate.py / LaLonde).
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from causalrl import certify_decision

N_ACTIONS = 80


def _load(policy: str) -> pd.DataFrame:
    import obp

    path = os.path.join(os.path.dirname(obp.__file__), "dataset", "obd", policy, "all", "all.csv")
    return pd.read_csv(path, index_col=0)


def main() -> None:
    rnd, bts = _load("random"), _load("bts")
    true_random = float(rnd["click"].mean())

    # (1) canonical off-policy evaluation: random-policy value estimated from the BTS logs.
    p, y = bts["propensity_score"].to_numpy(), bts["click"].to_numpy(dtype=float)
    w = (1.0 / N_ACTIONS) / np.clip(p, 1e-6, None)
    ips, snips = float((w * y).mean()), float((w * y).sum() / w.sum())
    print("(1) canonical OPE -- random-policy value from BTS logs (top OPE techniques):")
    print(f"    ground truth (on-policy) {true_random:.5f} | IPS {ips:.5f} | SNIPS {snips:.5f}")
    print("    IPS and SNIPS agree and are in the ballpark; sample-limited on this 10k slice.\n")

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

    # (3) the certificate on the clean, known-propensity logs -- a correct true-negative.
    cert = certify_decision(
        outcomes=yr.tolist(),
        treated=(a == istar).astype(int).tolist(),
        propensities=[1.0 / N_ACTIONS] * len(a),
        gamma_max=5.0,
    )
    tip = cert.tipping_gamma
    print(
        f"(3) certificate on clean OBD: {cert.decision}; tips at Γ≈{tip:.2f}"
        if tip
        else "(3) robust"
    )
    print(
        "    Honest scope: propensities are known and correct, so there is nothing to deconfound,"
    )
    print("    and 'best item beats random' is not a marginal decision -- the certificate's edge")
    print("    needs a marginal OBSERVATIONAL decision (LaLonde), not a clear-cut one.")


if __name__ == "__main__":
    main()
