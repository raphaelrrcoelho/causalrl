"""Partial-identification bounds for confounded causal effects (Manski; sensitivity models)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import NamedTuple

import numpy as np

from causalrl.data.dataset import ConfoundedTrajectoryDataset
from causalrl.exceptions import NotIdentifiableError


class Interval(NamedTuple):
    """A partial-identification interval ``[lower, upper]``.

    Tuple-compatible: ``lo, hi = interval`` and ``interval[0]`` work, so existing
    tuple-consuming callers are unaffected; ``.lower`` / ``.upper`` read more clearly.
    """

    lower: float
    upper: float


def causal_q_bounds(
    dataset: ConfoundedTrajectoryDataset,
    state: int,
    action: int,
    *,
    require_identified: bool = False,
) -> Interval:
    """Manski natural bounds on E[return | do(action), state] from confounded logs.

    For a return in [0, 1] with empirical mean m = E[R|s,a] and propensity p = P(a|s):
        lower = m * p,  upper = m * p + (1 - p).
    A never-logged action (p = 0) yields the vacuous [0, 1] — not identifiable from the
    logs alone. With `require_identified=True`, a vacuous bound raises NotIdentifiableError
    carrying (state, action) as the witness.
    """
    p = dataset.behavior_propensity(state, action)
    m = dataset.mean_reward(state, action)
    lower = m * p
    upper = m * p + (1.0 - p)
    if require_identified and p == 0.0:
        raise NotIdentifiableError(
            f"E[R|do(a={action}), s={state}] is not identifiable: action never logged "
            f"in this state (vacuous bound [0, 1])",
            witness=(state, action),
        )
    return Interval(lower, upper)


def manski_bounds(
    data: Mapping[str, Sequence[float]],
    *,
    treatment: str,
    outcome: str,
    action: int,
    outcome_range: tuple[float, float] = (0.0, 1.0),
) -> Interval:
    """Sharp no-assumptions bounds on ``E[outcome | do(treatment = action)]`` (Manski 1990).

    From observational ``data`` (integer ``treatment`` column, numeric ``outcome`` in
    ``outcome_range``): the units that took ``action`` contribute their observed mean, while
    the rest are bounded only by the outcome range. With ``p = P(treatment = action)`` and
    ``m = E[outcome | treatment = action]`` the bounds are
    ``[m*p + y_min*(1-p), m*p + y_max*(1-p)]`` — sharp, collapsing to a point when every unit took
    ``action``. The observational counterpart of :func:`causal_q_bounds`.
    """
    x = np.asarray(data[treatment])
    y = np.asarray(data[outcome], dtype=float)
    y_min, y_max = outcome_range
    mask = x == action
    p = float(mask.mean())
    observed = float(y[mask].mean()) if bool(mask.any()) else 0.0
    return Interval(observed * p + y_min * (1.0 - p), observed * p + y_max * (1.0 - p))


def _fractional_extreme(y: np.ndarray, lo: np.ndarray, hi: np.ndarray, *, maximize: bool) -> float:
    """Extreme value of ``(Σ a_i y_i)/(Σ a_i)`` with ``a_i ∈ [lo_i, hi_i]``.

    The optimum is a threshold rule on ``y`` (the upper weight goes to the extreme-``y`` end); scan
    the sorted threshold with prefix sums in linear time.
    """
    order = np.argsort(-y if maximize else y)
    y, lo, hi = y[order], lo[order], hi[order]
    zero = np.zeros(1)
    cum_hi_y = np.concatenate([zero, np.cumsum(hi * y)])
    cum_hi = np.concatenate([zero, np.cumsum(hi)])
    cum_lo_y = np.concatenate([zero, np.cumsum(lo * y)])
    cum_lo = np.concatenate([zero, np.cumsum(lo)])
    tot_lo_y, tot_lo = cum_lo_y[-1], cum_lo[-1]
    numer = cum_hi_y + (tot_lo_y - cum_lo_y)  # top-k get hi, the rest get lo
    denom = cum_hi + (tot_lo - cum_lo)
    ratios = numer[denom > 0] / denom[denom > 0]
    return float(ratios.max() if maximize else ratios.min())


def ipw_sensitivity_bounds(
    outcomes: Sequence[float], propensities: Sequence[float], *, gamma: float
) -> Interval:
    """Marginal-sensitivity-model bounds on the treated counterfactual mean ``E[Y(1)]``.

    ``outcomes`` and ``propensities`` are the treated units' outcomes ``Y_i`` and *nominal*
    propensities ``e(Z_i) = P(treated | Z_i)`` (what an unconfounded model fits). Under Tan's
    marginal sensitivity model the true inverse-propensity weight lies within an odds-ratio factor
    ``gamma >= 1`` of the nominal, giving ``a_i in [1 + (1/g)(1/e_i - 1), 1 + g(1/e_i - 1)]``; the
    bounds are the extreme stabilized (Hájek) weighted means over that range. At ``gamma = 1`` the
    interval collapses to the IPW point estimate; it widens monotonically with ``gamma`` and
    contains ``E[Y(1)]`` whenever the true confounding odds ratio is at most ``gamma``.

    Faithful to Z. Tan, *A Distributional Approach for Causal Inference Using Propensity Scores*
    (JASA 2006) and Q. Zhao, D. Small, B. Bhattacharya, *Sensitivity Analysis for Inverse
    Probability Weighting Estimators via the Percentile Bootstrap* (JRSS-B 2019). No code is ported.
    """
    if gamma < 1.0:
        raise ValueError("gamma must be >= 1")
    y = np.asarray(outcomes, dtype=float)
    e = np.asarray(propensities, dtype=float)
    odds = (1.0 - e) / e
    lo_w = 1.0 + odds / gamma
    hi_w = 1.0 + odds * gamma
    return Interval(
        _fractional_extreme(y, lo_w, hi_w, maximize=False),
        _fractional_extreme(y, lo_w, hi_w, maximize=True),
    )


def msm_per_step_bounds(
    rewards_by_step: Sequence[Sequence[float]],
    propensities_by_step: Sequence[Sequence[float]],
    *,
    gamma: float,
) -> Interval:
    """Per-step marginal-sensitivity-model bounds on a cumulative (summed) reward.

    Each element of ``rewards_by_step`` / ``propensities_by_step`` is one time step's
    per-unit rewards ``r_t`` and nominal propensities ``e_t``; the cumulative-reward MSM
    bound is the sum over steps of the per-step :func:`ipw_sensitivity_bounds`. This is the
    additive (per-step) cumulative-reward MSM: each step is bounded independently under the
    sensitivity model and the bounds add, which is tight for sparse / per-step rewards.

    Reusable kernel of the per-step cumulative-reward MSM used for confounded multi-step OPE
    (Bennett & Kallus, *Efficient and Sharp OPE in Robust MDPs*, NeurIPS 2024). The
    experiment supplies the per-step nominal propensities; no code is ported.
    """
    if gamma < 1.0:
        raise ValueError("gamma must be >= 1")
    if len(rewards_by_step) != len(propensities_by_step):
        raise ValueError("rewards_by_step and propensities_by_step must have equal length")
    lower = upper = 0.0
    for r_t, e_t in zip(rewards_by_step, propensities_by_step, strict=True):
        iv = ipw_sensitivity_bounds(r_t, e_t, gamma=gamma)
        lower += iv.lower
        upper += iv.upper
    return Interval(lower, upper)


def msm_stratified_bounds(
    values: Sequence[float],
    propensities: Sequence[float],
    strata: Sequence[int],
    target_weights: Mapping[int, float],
    *,
    gamma: float,
) -> Interval:
    """Stratified marginal-sensitivity-model bounds: ``Σ_s w_s · MSM_s``.

    Compute the MSM bound within each stratum (units sharing a ``strata`` label) and combine
    them with ``target_weights`` (the target stratum marginal ``w_s``, e.g. a uniform initial
    state distribution). Strata absent from the data contribute nothing. Because conditioning
    removes between-stratum weight variation, the stratified bound is never wider than the
    pooled :func:`ipw_sensitivity_bounds` and never under-covers (THEORY.md, Prop 1).

    The reusable kernel of the stratified cumulative-reward MSM; the experiment supplies the
    stratum labels (e.g. initial state) and nominal propensities.
    """
    if gamma < 1.0:
        raise ValueError("gamma must be >= 1")
    v = np.asarray(values, dtype=float)
    e = np.asarray(propensities, dtype=float)
    s = np.asarray(strata)
    lower = upper = 0.0
    for label, w in target_weights.items():
        mask = s == label
        if not bool(mask.any()):
            continue
        iv = ipw_sensitivity_bounds(v[mask].tolist(), e[mask].tolist(), gamma=gamma)
        lower += w * iv.lower
        upper += w * iv.upper
    return Interval(lower, upper)
