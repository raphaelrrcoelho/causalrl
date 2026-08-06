"""Split / weighted conformal prediction intervals (plan §7.4).

Distribution-free finite-sample coverage: given exchangeable calibration nonconformity scores, the
conformal quantile yields an interval whose marginal coverage is at least ``1 - alpha``. Provided:

* ``conformal_quantile`` — the (optionally weighted) conformal quantile of scores, with the standard
  ``+inf`` test-point mass so coverage is finite-sample valid (Vovk et al.; Tibshirani et al. 2019
  for the weighted / covariate-shift case).
* ``split_conformal_interval`` — symmetric residual interval around a point prediction.
* ``cqr_interval`` — conformalized quantile regression (Romano, Patterson & Candes 2019): adaptive
  intervals for heteroscedastic / heavy-tailed targets.
* ``certify_conformal_interval`` — wraps the above in a ``kind=EMPIRICAL`` certificate that records
  the (weighted-)exchangeability assumption. It is an *observational* (``query="see"``) predictive
  interval: split conformal around a fitted prediction consumes no causal assumption and licenses
  no interventional claim.
* ``conformal_action_value`` — the off-policy caller of the weighted path: the calibration
  likelihood ratio ``dP_test/dP_cal`` is the propensity ratio ``pi_target / pi_behavior``, so the
  same machinery calibrates the return of a decision taken under a target policy from confounded
  logs. This is the finite-sample layer :func:`causalrl.certify_policy` gates on.

Formula-level implementations of the cited methods; no third-party code is ported.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from causalrl.certify.certificate import (
    Assumption,
    Certificate,
    EstimandSpec,
    Hedge,
    Kind,
    Provenance,
)
from causalrl.data.dataset import ConfoundedTrajectoryDataset
from causalrl.identification.bounds import Interval

__all__ = [
    "certify_conformal_interval",
    "conformal_action_value",
    "conformal_quantile",
    "cqr_interval",
    "split_conformal_interval",
]


def conformal_quantile(
    scores: Sequence[float],
    alpha: float = 0.1,
    *,
    weights: Sequence[float] | None = None,
    test_weight: float | None = None,
) -> float:
    """Conformal quantile of nonconformity ``scores`` at miscoverage ``alpha``.

    Returns the smallest score whose (weight-)normalised rank reaches ``1 - alpha`` once the test
    point's ``+inf`` mass is included, so ``prediction ± result`` (or the CQR adjustment) is
    marginally valid at level ``1 - alpha`` under (weighted) exchangeability. Returns ``inf`` (a
    valid, infinite interval) when the calibration set is too small for the requested level.
    ``weights`` are calibration likelihood ratios ``dP_test/dP_cal``; ``test_weight`` defaults to
    their mean.
    """
    s = np.asarray(scores, dtype=np.float64)
    n = s.size
    if n == 0:
        raise ValueError("need at least one calibration score")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=np.float64)
    if w.shape != s.shape:
        raise ValueError("weights must match scores in length")
    wt = float(w.mean()) if test_weight is None else float(test_weight)
    order = np.argsort(s)
    s_sorted = s[order]
    w_sorted = w[order]
    total = float(w_sorted.sum()) + wt
    cum = np.cumsum(w_sorted) / total
    idx = int(np.searchsorted(cum, 1.0 - alpha, side="left"))
    return float("inf") if idx >= n else float(s_sorted[idx])


def split_conformal_interval(
    prediction: float,
    cal_y: Sequence[float],
    cal_pred: Sequence[float],
    alpha: float = 0.1,
    *,
    weights: Sequence[float] | None = None,
    test_weight: float | None = None,
) -> Interval:
    """Symmetric split-conformal interval ``prediction ± q`` from absolute calibration residuals."""
    resid = np.abs(np.asarray(cal_y, dtype=np.float64) - np.asarray(cal_pred, dtype=np.float64))
    q = conformal_quantile(resid.tolist(), alpha, weights=weights, test_weight=test_weight)
    return Interval(float(prediction) - q, float(prediction) + q)


def cqr_interval(
    pred_lo: float,
    pred_hi: float,
    cal_y: Sequence[float],
    cal_lo: Sequence[float],
    cal_hi: Sequence[float],
    alpha: float = 0.1,
    *,
    weights: Sequence[float] | None = None,
    test_weight: float | None = None,
) -> Interval:
    """Conformalized quantile regression interval (Romano, Patterson & Candes 2019).

    ``cal_lo``/``cal_hi`` are the calibration lower/upper quantile predictions; the conformity score
    is ``max(cal_lo - y, y - cal_hi)`` and the returned interval widens the test quantile band
    ``[pred_lo, pred_hi]`` by the conformal quantile, giving adaptive finite-sample coverage.
    """
    y = np.asarray(cal_y, dtype=np.float64)
    lo = np.asarray(cal_lo, dtype=np.float64)
    hi = np.asarray(cal_hi, dtype=np.float64)
    scores = np.maximum(lo - y, y - hi)
    q = conformal_quantile(scores.tolist(), alpha, weights=weights, test_weight=test_weight)
    return Interval(float(pred_lo) - q, float(pred_hi) + q)


def certify_conformal_interval(
    prediction: float,
    cal_y: Sequence[float],
    cal_pred: Sequence[float],
    *,
    alpha: float = 0.1,
    weights: Sequence[float] | None = None,
    test_weight: float | None = None,
) -> Certificate:
    """A ``kind=EMPIRICAL`` certificate for a split-conformal prediction interval.

    ``value`` is left ``None`` (the claim is coverage of an individual outcome, not a point
    estimate); the interval is in ``ci`` and ``alpha`` is the miscoverage. The
    (weighted-)exchangeability requirement is recorded as a non-checkable :class:`Assumption`.

    The query is ``"see"`` and is not configurable: residuals of a fitted prediction carry no
    intervention, and ``weights`` only move the *observational* law to a shifted one, so neither
    an interventional nor a counterfactual label would be earned. For the off-policy claim, use
    :func:`conformal_action_value`, which computes the propensity ratio and records the causal
    assumptions that license it.
    """
    interval = split_conformal_interval(
        prediction, cal_y, cal_pred, alpha, weights=weights, test_weight=test_weight
    )
    shifted = weights is not None
    return Certificate(
        claim=(
            f"P(Y in [{interval.lower:.4g}, {interval.upper:.4g}]) >= {1.0 - alpha:.2f} (conformal)"
        ),
        estimand=EstimandSpec(query="see", target="interval"),
        kind=Kind.EMPIRICAL,
        value=None,
        alpha=alpha,
        assumptions=(
            Assumption(
                name="weighted-exchangeability" if shifted else "exchangeability",
                params={"n_calibration": int(np.size(cal_y))},
                checkable=False,
            ),
        ),
        method="weighted-split-conformal" if shifted else "split-conformal",
        witness=None,
        hedge=None,
        provenance=Provenance.create(),
        ci=interval,
    )


def _return_band(
    returns: Sequence[float],
    alpha: float,
    *,
    weights: Sequence[float] | None,
    test_weight: float | None,
) -> Interval:
    """Two-sided conformal band for a fresh return, ``alpha / 2`` spent on each end.

    Scores are the returns themselves (and their negation for the lower end): no predictor is
    fitted, so there is no train/calibration split to violate and each end is exactly the standard
    one-sided (weighted) conformal bound. Union bound over the two ends gives ``1 - alpha``.
    """
    half = alpha / 2.0
    lower = conformal_quantile(
        [-float(r) for r in returns], half, weights=weights, test_weight=test_weight
    )
    upper = conformal_quantile(
        [float(r) for r in returns], half, weights=weights, test_weight=test_weight
    )
    return Interval(-lower, upper)


def _propensity_ratios(
    dataset: ConfoundedTrajectoryDataset, target_actions: Sequence[int]
) -> tuple[list[float], float, list[tuple[int, int]]]:
    """Per-transition ``pi_target(a|s) / pi_behavior(a|s)``, the test-point bound, and the gaps.

    The target policy is deterministic (one action per logged transition), so its probability at
    the logged action is the match indicator and the ratio is ``1[a_i = pi(s_i)] / e0(a_i | s_i)``;
    ``e0`` is the dataset's empirical behaviour propensity, which is positive at every logged
    transition. The test point's own ratio is unknown (it depends on the fresh state) and is
    therefore bounded by the largest ratio the policy can produce on the logged state marginal —
    conservative, so the band only ever widens. The third element lists the ``(state, action)``
    pairs the policy reaches that the logs never played: positivity failures, for which that bound
    is infinite and the band vacuous.
    """
    transitions = dataset.transitions
    if len(target_actions) != len(transitions):
        raise ValueError("target_actions must have one action per logged transition")
    pairs = [(tr.state, int(a)) for a, tr in zip(target_actions, transitions, strict=True)]
    weights = [
        1.0 / dataset.behavior_propensity(tr.state, tr.action) if a == tr.action else 0.0
        for (_s, a), tr in zip(pairs, transitions, strict=True)
    ]
    reached = {(s, a): dataset.behavior_propensity(s, a) for s, a in pairs}
    unsupported = sorted(pair for pair, e0 in reached.items() if e0 <= 0.0)
    supported = [1.0 / e0 for e0 in reached.values() if e0 > 0.0]
    test_weight = max(supported) if not unsupported and supported else float("inf")
    return weights, test_weight, unsupported


def conformal_action_value(
    dataset: ConfoundedTrajectoryDataset,
    target_actions: Sequence[int] | None = None,
    *,
    alpha: float = 0.1,
) -> Certificate:
    """Calibrated band for the return of one decision taken under a target policy.

    The off-policy caller of the weighted path. At logged transition ``i`` the likelihood ratio
    ``dP_target/dP_behavior`` is ``pi_target(a_i | s_i) / pi_behavior(a_i | s_i)``, computed here
    from the dataset's empirical :meth:`~causalrl.ConfoundedTrajectoryDataset.behavior_propensity`
    and ``target_actions[i]`` — the action the policy takes at that transition's state, one per
    logged transition, exactly the argument :func:`causalrl.certify_policy` takes. Passing
    ``target_actions=None`` scores the logging policy itself, whose logged returns need no
    reweighting; that is the reference the lower-bound gate compares against.

    Returns a ``kind=EMPIRICAL`` certificate whose ``ci`` is the band and whose ``value`` is
    ``None``. **The claim is coverage of a single fresh return, not a confidence interval for**
    ``V(pi) = E[return]``: each end is a one-sided conformal bound at ``alpha / 2``, so
    ``ci.lower`` is a distribution-free lower confidence bound on the next return under the target
    policy at level ``1 - alpha/2``, and the two-sided band covers at ``1 - alpha``. Too few
    effectively-weighted samples for the level returns an infinite end rather than a false one.

    Valid under (i) weighted exchangeability of the logged returns; (ii) no unmeasured confounding
    of the logged action, without which the ratio above is not ``dP_target/dP_behavior``; and
    (iii) positivity — every ``(state, action)`` the policy reaches must have been played by the
    logs, else the band is vacuous and carries a positivity :class:`~causalrl.certify.Hedge`. The
    state distribution is taken to be unchanged by the policy: the one-step / terminal-return
    regime :func:`causalrl.certify_policy` documents.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    returns = [tr.reward for tr in dataset.transitions]
    weights: list[float] | None = None
    test_weight: float | None = None
    unsupported: list[tuple[int, int]] = []
    if target_actions is not None:
        weights, test_weight, unsupported = _propensity_ratios(dataset, target_actions)
    band = _return_band(returns, alpha, weights=weights, test_weight=test_weight)

    w = np.ones(len(returns)) if weights is None else np.asarray(weights, dtype=np.float64)
    sq = float(np.square(w).sum())
    ess = float(w.sum() ** 2 / sq) if sq > 0.0 else 0.0
    assumptions = [
        Assumption(
            name="weighted-exchangeability" if weights is not None else "exchangeability",
            params={"n_calibration": len(returns), "effective_sample_size": ess},
            checkable=False,
        )
    ]
    hedge: Hedge | None = None
    if target_actions is not None:
        assumptions.append(Assumption(name="no-unmeasured-confounding", params={}, checkable=False))
        assumptions.append(
            Assumption(
                name="positivity",
                params={"max_propensity_ratio": test_weight},
                checkable=True,
                diagnostic={"unsupported_pairs": [list(pair) for pair in unsupported]},
            )
        )
        if unsupported:
            hedge = Hedge(
                reason="positivity: the target policy takes actions the logs never played there",
                detail={"unsupported_state_action_pairs": [list(pair) for pair in unsupported]},
            )
    label = "behavior" if target_actions is None else "target"
    return Certificate(
        claim=(
            f"P(return of one decision under the {label} policy in "
            f"[{band.lower:.4g}, {band.upper:.4g}]) >= {1.0 - alpha:.2f} (conformal); "
            "a fresh return, not E[return]"
        ),
        estimand=EstimandSpec(query="policy_value", target="quantile", policy=label),
        kind=Kind.EMPIRICAL,
        value=None,
        alpha=alpha,
        assumptions=tuple(assumptions),
        method="weighted-conformal-return-band" if weights is not None else "conformal-return-band",
        witness=None,
        hedge=hedge,
        provenance=Provenance.create(),
        ci=band,
    )
