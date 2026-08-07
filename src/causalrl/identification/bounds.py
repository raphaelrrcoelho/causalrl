"""Partial-identification and sign-robustness kernels for identification (Manski; pivotality).

The value-side sensitivity kernels an RL caller reaches for — ``causal_q_bounds``,
``ipw_sensitivity_bounds``, ``msm_policy_value_bounds``, ``msm_contribution_bounds``,
``msm_per_step_bounds``, ``msm_stratified_bounds`` — live in :mod:`causalrl.ope.bounds` with the
rest of off-policy evaluation. What stays here is what identification itself needs: the
:class:`Interval` type every bound returns, observational :func:`manski_bounds`, and the
decision-pivotality layer (:func:`pivotality_certificate`, :func:`confounding_bias_bound`,
:func:`mi_flip_threshold`) plus the model-agnostic :func:`tipping_gamma` reporter, which takes a
``gamma -> Interval`` callable and so depends on no particular bound.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import NamedTuple

import numpy as np


class Interval(NamedTuple):
    """A partial-identification interval ``[lower, upper]``.

    Tuple-compatible: ``lo, hi = interval`` and ``interval[0]`` work, so existing
    tuple-consuming callers are unaffected; ``.lower`` / ``.upper`` read more clearly.
    """

    lower: float
    upper: float


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
    ``action``. The observational counterpart of :func:`~causalrl.ope.bounds.causal_q_bounds`.
    """
    x = np.asarray(data[treatment])
    y = np.asarray(data[outcome], dtype=float)
    y_min, y_max = outcome_range
    mask = x == action
    p = float(mask.mean())
    observed = float(y[mask].mean()) if bool(mask.any()) else 0.0
    return Interval(observed * p + y_min * (1.0 - p), observed * p + y_max * (1.0 - p))


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
    (:func:`~causalrl.ope.bounds.msm_contribution_bounds`), then abstention
    (:func:`tipping_gamma`). One-sided:
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
