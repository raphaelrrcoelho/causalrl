"""Off-policy-evaluation bounds: what a policy could be worth when the logs are confounded.

Partial-identification and sensitivity kernels for policy *value*, all returning an
:class:`~causalrl.identification.bounds.Interval` (or, for the two ``return_certificate`` routines,
a ``BOUNDED`` :class:`~causalrl.certify.certificate.Certificate`):

* :func:`causal_q_bounds` — Manski natural bounds on ``E[return | do(a), s]`` from confounded RL
  logs, the no-assumptions floor under any amount of hidden confounding.
* :func:`ipw_sensitivity_bounds` — Tan's marginal sensitivity model on a treated counterfactual
  mean, and :func:`msm_policy_value_bounds` its off-policy generalisation ``V(pi_t)``.
* :func:`msm_contribution_bounds` — the MSM band on a *difference* ``V(pi_on) - V(pi_off)``, the
  band the decision layer (:func:`causalrl.certify_decision`) tips against.
* :func:`msm_per_step_bounds` / :func:`msm_stratified_bounds` — the cumulative-reward and
  stratified refinements of the same model.

The graph-side identification and sign-robustness kernels (``manski_bounds``,
``pivotality_certificate``, ``tipping_gamma``, …) stay in
:mod:`causalrl.identification.bounds`; these are the ones an RL caller reaches for, so they live
under :mod:`causalrl.ope` with the rest of off-policy evaluation. Every name here is additionally
re-exported at the top level (``from causalrl import msm_policy_value_bounds``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Literal, overload

import numpy as np

from causalrl.data.dataset import ConfoundedTrajectoryDataset
from causalrl.exceptions import NotIdentifiableError
from causalrl.identification.bounds import Interval

if TYPE_CHECKING:
    from causalrl.certify.certificate import Certificate

__all__ = [
    "causal_q_bounds",
    "ipw_sensitivity_bounds",
    "msm_contribution_bounds",
    "msm_per_step_bounds",
    "msm_policy_value_bounds",
    "msm_stratified_bounds",
]


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


@overload
def ipw_sensitivity_bounds(
    outcomes: Sequence[float],
    propensities: Sequence[float],
    *,
    gamma: float,
    return_certificate: Literal[True] | None = ...,
) -> Certificate: ...
@overload
def ipw_sensitivity_bounds(
    outcomes: Sequence[float],
    propensities: Sequence[float],
    *,
    gamma: float,
    return_certificate: Literal[False],
) -> Interval: ...
def ipw_sensitivity_bounds(
    outcomes: Sequence[float],
    propensities: Sequence[float],
    *,
    gamma: float,
    return_certificate: bool | None = None,
) -> Interval | Certificate:
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

    ``return_certificate`` (2.0 default): returns a ``BOUNDED`` :class:`Certificate`; pass ``False``
    for the legacy :class:`Interval`, or ``True`` (equivalently, the ``*_certified`` variant) to be
    explicit.
    """
    if return_certificate is None:
        return_certificate = True  # causalrl 2.0: return a Certificate by default
    if return_certificate:
        from causalrl.certify.routines import ipw_sensitivity_bounds_certified

        return ipw_sensitivity_bounds_certified(outcomes, propensities, gamma=gamma)
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


@overload
def msm_policy_value_bounds(
    outcomes: Sequence[float],
    logging_propensities: Sequence[float],
    target_propensities: Sequence[float],
    *,
    gamma: float,
    return_certificate: Literal[True] | None = ...,
) -> Certificate: ...
@overload
def msm_policy_value_bounds(
    outcomes: Sequence[float],
    logging_propensities: Sequence[float],
    target_propensities: Sequence[float],
    *,
    gamma: float,
    return_certificate: Literal[False],
) -> Interval: ...
def msm_policy_value_bounds(
    outcomes: Sequence[float],
    logging_propensities: Sequence[float],
    target_propensities: Sequence[float],
    *,
    gamma: float,
    return_certificate: bool | None = None,
) -> Interval | Certificate:
    """Marginal-sensitivity-model bounds on an off-policy value ``V(pi_t) = E[(pi_t/e0) Y]``.

    Self-normalised (Hájek) off-policy value of a target policy ``pi_t`` estimated from logs of a
    logging policy with *nominal* propensities ``e0(a|x) = P(a | x)`` (a valid probability in
    ``(0, 1]``). ``outcomes`` are the logged rewards ``Y_i``; ``target_propensities`` are
    ``pi_t(a_i | x_i)`` at the logged action. Under Tan's marginal sensitivity model the true
    logging propensity deviates from nominal by an odds-ratio at most ``gamma >= 1``, so the true
    inverse weight ``1/ẽ0`` lies in ``[1 + odds/gamma, 1 + odds*gamma]`` with ``odds = (1-e0)/e0``;
    each unit's contribution weight is ``pi_t(a_i|x_i) * (1/ẽ0)``. The bounds are the extreme
    stabilised weighted means of ``Y`` over those per-unit weight ranges.

    Reduces to :func:`ipw_sensitivity_bounds` when ``pi_t`` is constant across the logged actions
    (the treated / uniform-target mean — the constant cancels in the self-normalised ratio), and
    collapses to the self-normalised IPS point at ``gamma = 1``. The off-policy generalisation of
    Tan's MSM in the spirit of N. Kallus & A. Zhou, *Confounding-Robust Policy Evaluation in
    Infinite-Horizon Reinforcement Learning* (NeurIPS 2020). The caller supplies ``pi_t`` and the
    nominal ``e0``; no code is ported.

    ``return_certificate`` (2.0 default): returns a ``BOUNDED`` :class:`Certificate`; pass ``False``
    for the legacy :class:`Interval`, or ``True`` (equivalently, the ``*_certified`` variant) to be
    explicit.
    """
    if return_certificate is None:
        return_certificate = True  # causalrl 2.0: return a Certificate by default
    if return_certificate:
        from causalrl.certify.routines import msm_policy_value_bounds_certified

        return msm_policy_value_bounds_certified(
            outcomes, logging_propensities, target_propensities, gamma=gamma
        )
    if gamma < 1.0:
        raise ValueError("gamma must be >= 1")
    y = np.asarray(outcomes, dtype=float)
    e0 = np.asarray(logging_propensities, dtype=float)
    pt = np.asarray(target_propensities, dtype=float)
    odds = (1.0 - e0) / e0
    lo_w = pt * (1.0 + odds / gamma)
    hi_w = pt * (1.0 + odds * gamma)
    return Interval(
        _fractional_extreme(y, lo_w, hi_w, maximize=False),
        _fractional_extreme(y, lo_w, hi_w, maximize=True),
    )


def msm_contribution_bounds(
    outcomes: Sequence[float],
    logging_propensities: Sequence[float],
    target_propensities_on: Sequence[float],
    target_propensities_off: Sequence[float],
    *,
    gamma: float,
) -> Interval:
    """Marginal-sensitivity-model bounds on a *contribution* ``V(pi_on) - V(pi_off)``.

    The off-policy value DIFFERENCE between two target rules, estimated from confounded logs
    under Tan's marginal sensitivity model — e.g. a per-agent credit or per-factor contribution
    ``E[Y_{do(F=1)}] - E[Y_{do(F=0)}]``. Each arm is bounded by :func:`msm_policy_value_bounds`
    (``target_propensities_on`` = ``pi_on(a_i | x_i)`` at the logged action, ``..._off`` likewise,
    shared nominal ``e0``) and the contribution interval is the difference

        [ on.lower - off.upper ,  on.upper - off.lower ].

    Always *valid* (it contains the true difference for any targets); **sharp** when the two
    target supports are disjoint — e.g. the deterministic one-hot arms ``1{F=1}`` / ``1{F=0}``
    that partition the logged units, so the two arms' weight perturbations are independent — and
    *conservative* otherwise. Collapses to the difference of the two self-normalised IPS points at
    ``gamma = 1`` and widens monotonically with ``gamma``.
    """
    if gamma < 1.0:
        raise ValueError("gamma must be >= 1")
    on = msm_policy_value_bounds(
        outcomes,
        logging_propensities,
        target_propensities_on,
        gamma=gamma,
        return_certificate=False,
    )
    off = msm_policy_value_bounds(
        outcomes,
        logging_propensities,
        target_propensities_off,
        gamma=gamma,
        return_certificate=False,
    )
    return Interval(on.lower - off.upper, on.upper - off.lower)


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
        iv = ipw_sensitivity_bounds(r_t, e_t, gamma=gamma, return_certificate=False)
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
        iv = ipw_sensitivity_bounds(
            v[mask].tolist(), e[mask].tolist(), gamma=gamma, return_certificate=False
        )
        lower += w * iv.lower
        upper += w * iv.upper
    return Interval(lower, upper)
