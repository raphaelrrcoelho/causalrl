"""Partial-identification bounds for confounded causal effects (Manski; sensitivity models)."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Literal, NamedTuple, overload

import numpy as np

from causalrl._deprecation import warn_certificate_default_flip
from causalrl.data.dataset import ConfoundedTrajectoryDataset
from causalrl.exceptions import NotIdentifiableError

if TYPE_CHECKING:
    from causalrl.certify.certificate import Certificate


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


@overload
def ipw_sensitivity_bounds(
    outcomes: Sequence[float],
    propensities: Sequence[float],
    *,
    gamma: float,
    return_certificate: Literal[True],
) -> Certificate: ...
@overload
def ipw_sensitivity_bounds(
    outcomes: Sequence[float],
    propensities: Sequence[float],
    *,
    gamma: float,
    return_certificate: Literal[False] | None = ...,
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

    ``return_certificate`` (I9 deprecation): leave unset for the current :class:`Interval` plus a
    ``FutureWarning`` that causalrl 2.0 will return a ``BOUNDED`` :class:`Certificate` by default;
    pass ``False`` to keep the :class:`Interval` silently, or ``True`` for the certificate now.
    """
    if return_certificate is None:
        warn_certificate_default_flip("ipw_sensitivity_bounds", "ipw_sensitivity_bounds_certified")
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
    return_certificate: Literal[True],
) -> Certificate: ...
@overload
def msm_policy_value_bounds(
    outcomes: Sequence[float],
    logging_propensities: Sequence[float],
    target_propensities: Sequence[float],
    *,
    gamma: float,
    return_certificate: Literal[False] | None = ...,
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

    ``return_certificate`` (I9 deprecation): leave unset for the current :class:`Interval` plus a
    ``FutureWarning`` that causalrl 2.0 will return a ``BOUNDED`` :class:`Certificate` by default;
    pass ``False`` to keep the :class:`Interval` silently, or ``True`` for the certificate now.
    """
    if return_certificate is None:
        warn_certificate_default_flip(
            "msm_policy_value_bounds", "msm_policy_value_bounds_certified"
        )
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


class PivotalityCertificate(NamedTuple):
    """Result of :func:`pivotality_certificate` (one-sided sign-robustness certificate)."""

    certified: bool
    naive: float
    bias_bound: float
    mi_flip: float
    mi_measured: float | None


def mi_flip_threshold(
    naive: float,
    span_treated: float,
    span_control: float,
    p_treated: float,
) -> float:
    """Channel capacity (nats) below which NO hidden confounder can flip the naive sign.

    The decision-pivotality threshold (sharp form)

        ``MI_flip = 2 naive^2 / (M1^2/p + M0^2/(1-p))``:

    if the mutual information between the logged binary action and a hidden variable ``Z`` is
    below this value, the omitted-``Z`` bias is strictly smaller than ``|naive|`` (omitted-
    variable bias in TV form + Pinsker with the KL budget split optimally across arms), so the
    ``Z``-adjusted contrast has the same sign as the naive one. This constant is SHARP: a
    binary-channel family attains the corresponding bias bound with ratio -> 1 in the small-MI
    limit (THEORY_pivotality.md, tightness section); the additive form
    ``2(|naive|/(M1/sqrt(p)+M0/sqrt(1-p)))^2`` is its (looser) Cauchy-Schwarz relaxation.
    Combined with the data-processing inequality ``MI(F;Z) <= MI(I;Z)`` — the logging actor's
    policy reads only its information set ``I`` — the *information structure of the
    environment* caps the reachable confounding budget.

    ``span_treated`` / ``span_control`` are the spans of ``E[Y | F=f, Z=z]`` over ``z`` (the
    regression spans ``M1`` / ``M0``); with ``Z`` unmeasured, use the outcome span for both
    (valid, looser).
    """
    if not 0.0 < p_treated < 1.0:
        raise ValueError("p_treated must be in (0, 1)")
    if span_treated < 0.0 or span_control < 0.0:
        raise ValueError("spans must be non-negative")
    denom2 = span_treated**2 / p_treated + span_control**2 / (1.0 - p_treated)
    if denom2 == 0.0:
        return float("inf")
    return float(2.0 * naive**2 / denom2)


def confounding_bias_bound(
    outcomes: Sequence[float],
    treated: Sequence[int],
    confounder_bins: Sequence[int],
    *,
    form: str = "tv",
) -> float:
    """Upper bound on the omitted-variable bias ``|naive - Z-adjusted|`` from logged rows.

    Omitted-variable bias in total-variation form:

        ``|bias| <= M1 * TV(P_{Z|F=1}, P_Z) + M0 * TV(P_{Z|F=0}, P_Z)``

    with ``M_f`` the span over ``Z``-bins of ``E[Y | F=f, Z]`` (each signed measure integrates
    to zero, so the midrange trick gives the span, not the sup; sharp — attained with equality
    by an explicit two-point family at every parameter value). ``form="mi"`` applies Pinsker
    with the KL budget ``p*KL1 + (1-p)*KL0 = MI(F;Z)`` split optimally across arms (sharp
    small-MI constant):

        ``|bias| <= sqrt( MI/2 * (M1^2/p + M0^2/(1-p)) )``  (capped at the trivial ``M1+M0``),

    the (looser-than-TV) form whose budget the environment's information structure can cap.
    Every stratum must contain both arms (positivity); restrict to an overlap population first —
    a logger that conditions hard on ``Z`` destroys overlap, which is a finding, not a nuisance.
    """
    if form not in ("tv", "mi"):
        raise ValueError("form must be 'tv' or 'mi'")
    y = np.asarray(outcomes, dtype=float)
    f = np.asarray(treated, dtype=bool)
    z = np.asarray(confounder_bins)
    if not (y.shape == f.shape == z.shape):
        raise ValueError("outcomes, treated, confounder_bins must have equal length")
    if not (f.any() and (~f).any()):
        raise ValueError("both arms must be present")
    p = float(f.mean())
    bins = np.unique(z)
    m1: list[float] = []
    m0: list[float] = []
    pz: list[float] = []
    pz1: list[float] = []
    pz0: list[float] = []
    for b in bins:
        in_b = z == b
        if not ((in_b & f).any() and (in_b & ~f).any()):
            raise ValueError(
                f"stratum {b!r} lacks one arm (positivity violated) — restrict to an overlap "
                "population before bounding"
            )
        m1.append(float(y[in_b & f].mean()))
        m0.append(float(y[in_b & ~f].mean()))
        pz.append(float(in_b.mean()))
        pz1.append(float((in_b & f).sum() / f.sum()))
        pz0.append(float((in_b & ~f).sum() / (~f).sum()))
    span1 = max(m1) - min(m1)
    span0 = max(m0) - min(m0)
    pz_a = np.asarray(pz, dtype=float)
    pz1_a = np.asarray(pz1, dtype=float)
    pz0_a = np.asarray(pz0, dtype=float)
    if form == "tv":
        tv1 = 0.5 * float(np.abs(pz1_a - pz_a).sum())
        tv0 = 0.5 * float(np.abs(pz0_a - pz_a).sum())
        return span1 * tv1 + span0 * tv0
    mi = _plugin_mi(pz_a, pz1_a, pz0_a, p)
    sharp = float(np.sqrt(mi / 2.0 * (span1**2 / p + span0**2 / (1.0 - p))))
    return min(sharp, span1 + span0)  # trivial cap: TV_f <= 1


def confounding_bias_per_step_bounds(
    outcomes: Sequence[float],
    treated_by_step: Sequence[Sequence[int]],
    confounder_bins: Sequence[int],
    *,
    form: str = "tv",
) -> list[float]:
    """Per-step channel-cap bounds for sequential per-factor credits (THEORY Theorem-seq).

    One row per unit (e.g. per episode); ``treated_by_step[t]`` is step ``t``'s binary action
    factor for every unit, ``confounder_bins`` the shared episode-level hidden variable. Each
    step's omitted-``Z`` credit bias ``|naive_t - adjusted_t|`` is bounded by
    :func:`confounding_bias_bound` at that step's arm split; a cumulative additive credit's
    bias is bounded by the sum. This is the average-budget (information) sibling of the
    uniform-odds :func:`msm_per_step_bounds`: there each step gets an MSM ``gamma`` budget,
    here each step gets the channel budget ``MI(F_t;Z) <= MI(I_t;Z)`` of ITS OWN information
    set — in games, later information sets typically leak more about ``Z``, so later credits
    are systematically less protected (the growing-channel prediction).
    """
    return [
        confounding_bias_bound(outcomes, f_t, confounder_bins, form=form) for f_t in treated_by_step
    ]


def _plugin_mi(pz: np.ndarray, pz1: np.ndarray, pz0: np.ndarray, p: float) -> float:
    """Plug-in ``MI(F;Z)`` (nats) from the binned conditionals."""
    mi = 0.0
    for pzf, pf in ((pz1, p), (pz0, 1.0 - p)):
        good = (pzf > 0) & (pz > 0)
        mi += pf * float(np.sum(pzf[good] * np.log(pzf[good] / pz[good])))
    return max(mi, 0.0)


def pivotality_certificate(
    outcomes: Sequence[float],
    treated: Sequence[int],
    confounder_bins: Sequence[int] | None = None,
    *,
    mi_cap: float | None = None,
) -> PivotalityCertificate:
    """One-sided sign-robustness certificate for a naive contrast under hidden confounding.

    ``certified=True`` means: no hidden variable consistent with the supplied information can
    flip the sign of ``E[Y|F=1] - E[Y|F=0]``. Two modes:

    * ``confounder_bins`` given (a *measured* hidden variable, e.g. post-hoc showdown/oracle
      data): certify iff the TV-form :func:`confounding_bias_bound` is strictly below
      ``|naive|``; ``mi_measured`` reports the plug-in channel.
    * ``mi_cap`` given (a *structural* cap on ``MI(I;Z)`` from the environment's information
      rules — the data-processing route): certify iff ``mi_cap < mi_flip`` computed with the
      outcome-span relaxation (no ``Z`` needed anywhere).

    The cheapest layer of the decision stack — certificate, then MSM band
    (:func:`msm_contribution_bounds`), then abstention (:func:`tipping_gamma`). One-sided:
    failure to certify is NOT evidence of a flip. Verified against measured ground truth in
    three game-log regimes in ``experiments/games/theory/verify_pivotality.py``.
    """
    y = np.asarray(outcomes, dtype=float)
    f = np.asarray(treated, dtype=bool)
    if not (f.any() and (~f).any()):
        raise ValueError("both arms must be present")
    p = float(f.mean())
    naive = float(y[f].mean() - y[~f].mean())
    if confounder_bins is not None:
        bound = confounding_bias_bound(outcomes, treated, confounder_bins, form="tv")
        z = np.asarray(confounder_bins)
        bins = np.unique(z)
        pz = np.array([(z == b).mean() for b in bins])
        pz1 = np.array([((z == b) & f).sum() / f.sum() for b in bins])
        pz0 = np.array([((z == b) & ~f).sum() / (~f).sum() for b in bins])
        mi_measured: float | None = _plugin_mi(pz, pz1, pz0, p)
        m1 = [float(y[(z == b) & f].mean()) for b in bins]
        m0 = [float(y[(z == b) & ~f].mean()) for b in bins]
        mi_flip = mi_flip_threshold(naive, max(m1) - min(m1), max(m0) - min(m0), p)
    elif mi_cap is not None:
        if mi_cap < 0.0:
            raise ValueError("mi_cap must be >= 0")
        span = float(y.max() - y.min())
        mi_flip = mi_flip_threshold(naive, span, span, p)
        # sharp MI form with M1 = M0 = span: span * sqrt(C / (2 p (1-p))), trivially capped
        bound = min(span * float(np.sqrt(mi_cap / (2 * p * (1 - p)))), 2 * span)
        mi_measured = None
    else:
        raise ValueError("supply confounder_bins (measured Z) or mi_cap (structural cap)")
    return PivotalityCertificate(
        certified=bool(bound < abs(naive)),
        naive=naive,
        bias_bound=float(bound),
        mi_flip=float(mi_flip),
        mi_measured=mi_measured,
    )


def tipping_gamma(
    bound: Callable[[float], Interval],
    *,
    reference: float = 0.0,
    gamma_max: float = 10.0,
    tol: float = 1e-3,
) -> float | None:
    """Sensitivity tipping point: the smallest ``gamma >= 1`` at which the partial-ID interval
    ``bound(gamma)`` first contains ``reference``.

    This is the causal-sensitivity *reporting* companion to the MSM bound kernels: it answers
    "how strong would unmeasured confounding have to be (on the MSM/Rosenbaum odds-ratio scale)
    to overturn the conclusion that the estimand lies strictly on one side of ``reference``?".
    A larger tipping ``gamma`` ⇒ a more robust conclusion — the odds-ratio-scale analog of the
    E-value (VanderWeele & Ding, *Ann. Intern. Med.* 2017) and of Rosenbaum's ``Gamma``.

    ``bound`` maps a sensitivity level ``gamma`` to an :class:`Interval`; it must collapse to a
    point at ``gamma = 1`` and widen monotonically with ``gamma`` (as every MSM kernel here does —
    e.g. ``lambda g: msm_contribution_bounds(y, e0, on, off, gamma=g)``). Returns ``1.0`` if the
    point already sits on ``reference``, and ``None`` if the interval never reaches ``reference``
    by ``gamma_max`` (the conclusion is robust to confounding at least that strong).
    """
    if gamma_max < 1.0:
        raise ValueError("gamma_max must be >= 1")
    eps = 1e-12
    at1 = bound(1.0)
    if at1.lower - eps <= reference <= at1.upper + eps:
        return 1.0
    if not (bound(gamma_max).lower - eps <= reference <= bound(gamma_max).upper + eps):
        return None  # robust: never reaches `reference` within [1, gamma_max]
    lo, hi = 1.0, gamma_max
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        iv = bound(mid)
        if iv.lower - eps <= reference <= iv.upper + eps:
            hi = mid
        else:
            lo = mid
    return float(hi)
