"""MNAR recommendation debiasing vs the top techniques, on Coat (selection bias).

Out-of-CI demo (network only). Run:

    uv run python examples/causal_mbrl_coat.py

In recommendation you only see ratings for items users *chose* to engage with, and that choice
is not random -- people rate things they like. So the observed ratings are Missing Not At Random,
and a naive average is biased high. Coat (Schnabel et al., "Recommendations as Treatments") ships a
biased self-selected training set, an UNBIASED randomly-assigned test set (ground truth), and the
observation propensities. We benchmark against the top MNAR techniques -- IPS, SNIPS, doubly-robust.

Read this as off-policy evaluation, because that is what it is: the *behaviour policy* is
self-selection (each user-item pair is exposed with the logged propensity), the *target policy* is
uniform MCAR exposure, and the estimand is the target policy's value -- the average rating you would
see if exposure were random. The MCAR test set measures that value directly, so we have ground
truth. causalrl computes both the point estimate and its sensitivity band with the same kernel:
`msm_policy_value_bounds` is the self-normalised (Hajek) off-policy value at Γ=1 and Tan's marginal
sensitivity model above it, and `msm_contribution_bounds` bounds the debiasing CORRECTION
V(MCAR) - V(self-selected), whose truth the test set also pins down.

Honest read: the debiasing genuinely works -- IPS / SNIPS / DR cut most of the selection bias and
land far closer to the truth than the naive average. A real win over naive, and parity with the top
methods. BUT it hinges on the propensity (selection) model being right: the sensitivity band shows
how far the answer moves once the propensities are allowed to be wrong by an odds-ratio Γ, and how
large a Γ it takes before the band even covers the measured truth. The honest deliverable is the
point estimate PLUS that band -- how much your answer depends on a selection model you cannot fully
verify.
"""

from __future__ import annotations

import io
import ssl
import urllib.request
import zipfile

import numpy as np

from causalrl import msm_contribution_bounds, msm_policy_value_bounds

URL = "https://www.cs.cornell.edu/~schnabts/mnar/coat.zip"
GAMMAS = (1.3, 1.6, 2.0)


def _load() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ctx = ssl.create_default_context()
    req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60, context=ctx) as response:
        archive = zipfile.ZipFile(io.BytesIO(response.read()))

    def read(name: str) -> np.ndarray:
        return np.loadtxt(io.BytesIO(archive.read(name)))

    return read("coat/train.ascii"), read("coat/test.ascii"), read("coat/propensities.ascii")


def _bias(matrix: np.ndarray, observed: np.ndarray, grand: float, axis: int) -> np.ndarray:
    """Observed mean along ``axis`` minus the grand mean (an additive bias term)."""
    counts = observed.sum(axis=axis)
    sums = np.where(observed, matrix, 0.0).sum(axis=axis)
    means = np.divide(sums, counts, out=np.full(counts.shape, grand), where=counts > 0)
    return means - grand


def _covering_gamma(
    ratings: list[float], logging: list[float], target: list[float], truth: float
) -> float | None:
    """Smallest Γ on a fine grid whose MSM band covers the measured MCAR truth."""
    for gamma in np.arange(1.0, 3.01, 0.01):
        band = msm_policy_value_bounds(
            ratings, logging, target, gamma=float(gamma), return_certificate=False
        )
        if band.lower <= truth <= band.upper:
            return float(gamma)
    return None


def main() -> None:
    train, test, propensity = _load()
    observed = train > 0
    n_total = train.size
    true_avg = float(test[test > 0].mean())

    # The logged bandit: one unit per observed (user, item) pair, exposed with propensity e0.
    ratings = train[observed].tolist()
    logging_propensities = propensity[observed].tolist()
    mcar_target = [1.0] * len(ratings)  # uniform exposure: the same weight on every logged pair

    w = train[observed] / propensity[observed]
    naive = float(train[observed].mean())
    ips = float(w.sum() / n_total)
    # SNIPS is causalrl's off-policy value of the MCAR target policy -- the Γ=1 collapse of the MSM.
    snips = msm_policy_value_bounds(
        ratings, logging_propensities, mcar_target, gamma=1.0, return_certificate=False
    ).lower

    grand = float(train[observed].mean())
    user_bias = _bias(train, observed, grand, axis=1)
    item_bias = _bias(train, observed, grand, axis=0)
    rhat = np.clip(grand + user_bias[:, None] + item_bias[None, :], 1.0, 5.0)
    dr = float(rhat.mean() + ((train - rhat)[observed] / propensity[observed]).sum() / n_total)

    print(f"Coat {train.shape[0]}x{train.shape[1]}  ({100 * observed.mean():.0f}% self-selected)\n")
    print("off-policy value of MCAR (uniform) exposure -- avg rating, 1-5:")
    for label, value in [
        ("TRUE avg rating (MCAR test)", true_avg),
        ("naive observed (biased)", naive),
        ("ours/strong: IPS", ips),
        ("ours/strong: SNIPS (causalrl)", snips),
        ("ours/strong: doubly-robust", dr),
    ]:
        gap = "" if label.startswith("TRUE") else f"   err {value - true_avg:+.3f}"
        print(f"  {label:30s} {value:.3f}{gap}")

    print("\ncertificate -- if the propensities (selection model) are wrong by odds-ratio Γ:")
    print("  Γ    MSM band on V(MCAR)      band on the correction V(MCAR) - V(self-selected)")
    for gamma in GAMMAS:
        level = msm_policy_value_bounds(
            ratings, logging_propensities, mcar_target, gamma=gamma, return_certificate=False
        )
        correction = msm_contribution_bounds(
            ratings, logging_propensities, mcar_target, logging_propensities, gamma=gamma
        )
        print(
            f"  {gamma:<4} [{level.lower:.2f}, {level.upper:.2f}]"
            f"             [{correction.lower:+.2f}, {correction.upper:+.2f}]"
        )
    print(f"  (the measured correction is {true_avg - naive:+.3f}: {true_avg:.3f} - {naive:.3f}.)")

    covering = _covering_gamma(ratings, logging_propensities, mcar_target, true_avg)
    if covering is None:
        print("  No Γ up to 3.0 produces a band covering the measured truth.")
    else:
        print(
            f"  The band first covers the measured truth {true_avg:.3f} at Γ≈{covering:.2f} --"
            " i.e. a selection"
        )
        print(
            f"  model wrong by that odds-ratio fully explains SNIPS' residual"
            f" {snips - true_avg:+.3f} error."
        )
    print("\nHonest read: IPS/SNIPS/DR cut most of the bias (win over naive, parity with the top")
    print("methods) -- but the answer moves substantially once the selection model is allowed to")
    print("be wrong. The point estimate plus that band is the honest deliverable, not the number")
    print("alone. causalrl's contribution here is the band and the Γ, not a better point estimate.")


if __name__ == "__main__":
    main()
