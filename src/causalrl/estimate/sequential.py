"""Finite-horizon sequential policy-value estimation (plan §7.2, deferred Phase-1 item).

Two estimators of a deterministic target policy's value ``V(pi) = E[Y^{pi}]`` over a fixed horizon
``T`` under **sequential ignorability** (the caller asserts that each stage's supplied history
``H_t`` blocks confounding of ``A_t``; recorded as an explicit, non-checkable assumption — I2/I3):

* ``gcomp`` — iterated conditional expectations (g-computation): a backward sequence of outcome
  regressions ``q_t = E[q_{t+1} | H_t, A_t]`` each evaluated at the target action, terminal
  ``q_{T+1} = Y``. Point-identified under sequential ignorability.
* ``dr`` — the sequentially doubly-robust (LTMLE-style) estimator: the same backward regressions
  plus a per-stage inverse-propensity augmentation along the policy-following path, cross-fitted so
  each ``q_1`` is an influence-function contribution (mean = value, spread = SE).

The ``dr`` estimator at horizon ``T = 1`` with a constant target action reduces **exactly** to the
shipped single-stage AIPW (pinned in the tests). Step-major data layout matches the shipped
``msm_per_step_bounds`` convention: one array per time step.

References: J. Robins (1986, g-computation); Bang & Robins (2005, sequential DR); M. van der Laan &
S. Gruber, *Targeted Minimum Loss Based Estimation of Longitudinal Effects* (LTMLE, 2012);
Chernozhukov et al. (2018, cross-fitting). Formula-level implementation; no third-party code ported.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from causalrl.certify.certificate import (
    Assumption,
    Certificate,
    EstimandSpec,
    Hedge,
    Kind,
    Provenance,
    Witness,
)
from causalrl.estimate._stats import norm_ppf
from causalrl.estimate.nuisance import (
    Classifier,
    LogisticRegressor,
    Regressor,
    RidgeRegressor,
)
from causalrl.identification.bounds import Interval

__all__ = [
    "SequentialValueEstimate",
    "certify_sequential_value",
    "estimate_sequential_value",
    "sequential_ice_values",
]

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.intp]
OutcomeFactory = Callable[[], Regressor]
PropensityFactory = Callable[[], Classifier]
_METHODS = ("gcomp", "dr")


@dataclass(frozen=True)
class SequentialValueEstimate:
    """A finite-horizon policy-value estimate with an (influence-function) CI and per-stage overlap.

    ``overlap_by_step[t]`` records ``min_propensity`` / ``max_propensity`` of the stage-``t``
    estimated propensity (``nan`` for the propensity-free ``gcomp`` estimator); a certifier
    downgrades to a hedge when any stage's positivity is destroyed (I3). ``std_error`` / ``ci`` are
    influence-function based for ``dr`` (cross-fitted) and a coarse pseudo-outcome-spread
    approximation for ``gcomp``.
    """

    value: float
    std_error: float
    ci: Interval
    alpha: float
    n: int
    horizon: int
    method: str
    overlap_by_step: list[dict[str, float]]
    n_folds: int | None = None


def _as2d(a: object) -> FloatArray:
    arr = np.asarray(a, dtype=np.float64)
    return arr[:, None] if arr.ndim == 1 else arr


def _with_action(h: FloatArray, a: FloatArray) -> FloatArray:
    """Stage outcome-regression features ``[H_t, A_t]`` (the learner adds the intercept)."""
    return np.concatenate([h, a.reshape(-1, 1)], axis=1)


def _prob1(model: Classifier, h: FloatArray) -> FloatArray:
    p = np.asarray(model.predict_proba(h), dtype=np.float64)
    return p[:, 1] if p.ndim == 2 else p


def _predict(model: Regressor, x: FloatArray) -> FloatArray:
    return np.asarray(model.predict(x), dtype=np.float64)


def _default_outcome() -> Regressor:
    return RidgeRegressor()


def _default_propensity() -> Classifier:
    return LogisticRegressor()


def _ci(value: float, se: float, alpha: float) -> Interval:
    z = float(norm_ppf(1.0 - alpha / 2.0))
    return Interval(value - z * se, value + z * se)


def _check_shapes(
    histories: Sequence[FloatArray],
    treatments: Sequence[FloatArray],
    target_actions: Sequence[FloatArray],
    outcome: FloatArray,
) -> tuple[int, int]:
    t = len(histories)
    if t == 0:
        raise ValueError("need at least one stage")
    if not (len(treatments) == len(target_actions) == t):
        raise ValueError("histories, treatments, target_actions must have equal horizon length")
    n = len(outcome)
    for stage, (a, ap) in enumerate(zip(treatments, target_actions, strict=True)):
        for name, arr in (("treatment", a), ("target_action", ap)):
            uniq = set(np.unique(np.asarray(arr, dtype=np.float64)).tolist())
            if not uniq <= {0.0, 1.0}:
                raise ValueError(f"stage {stage} {name} must be binary 0/1; got {sorted(uniq)}")
    return t, n


def _dr_fold(
    fit_idx: IntArray,
    eval_idx: IntArray,
    histories: Sequence[FloatArray],
    treatments: Sequence[FloatArray],
    target_actions: Sequence[FloatArray],
    outcome: FloatArray,
    *,
    outcome_factory: OutcomeFactory,
    propensity_factory: PropensityFactory,
    clip: float,
    overlaps: list[list[FloatArray]],
) -> FloatArray:
    """Backward sequentially-DR recursion; fit on ``fit_idx``, return ``q_1`` on ``eval_idx``.

    The augmentation propagates the deeper-stage residuals up through ``q_next``; each stage adds an
    inverse-propensity correction along the policy-following path, giving the sequentially
    doubly-robust pseudo-outcome (van der Laan & Gruber LTMLE).
    """
    horizon = len(histories)
    q_fit = outcome[fit_idx]
    q_eval = outcome[eval_idx]
    for t in reversed(range(horizon)):
        h_fit, a_fit, ap_fit = _stage(histories, treatments, target_actions, t, fit_idx)
        h_ev, a_ev, ap_ev = _stage(histories, treatments, target_actions, t, eval_idx)
        m = outcome_factory().fit(_with_action(h_fit, a_fit), q_fit)
        g_model = propensity_factory().fit(h_fit, a_fit)
        q_fit = _dr_update(m, g_model, h_fit, a_fit, ap_fit, q_fit, clip)
        e_ev = _prob1(g_model, h_ev)
        overlaps[t].append(e_ev)
        q_eval = _dr_update(m, g_model, h_ev, a_ev, ap_ev, q_eval, clip, e_raw=e_ev)
    return q_eval


def _stage(
    histories: Sequence[FloatArray],
    treatments: Sequence[FloatArray],
    target_actions: Sequence[FloatArray],
    t: int,
    idx: IntArray,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    return histories[t][idx], treatments[t][idx], target_actions[t][idx]


def _dr_update(
    outcome_model: Regressor,
    prop_model: Classifier,
    h: FloatArray,
    a: FloatArray,
    a_pi: FloatArray,
    q_next: FloatArray,
    clip: float,
    *,
    e_raw: FloatArray | None = None,
) -> FloatArray:
    qbar_obs = _predict(outcome_model, _with_action(h, a))
    qbar_pi = _predict(outcome_model, _with_action(h, a_pi))
    e = np.clip(_prob1(prop_model, h) if e_raw is None else e_raw, clip, 1.0 - clip)
    g_at_obs = np.where(a == 1.0, e, 1.0 - e)
    follow = (a == a_pi).astype(np.float64)
    return qbar_pi + follow / g_at_obs * (q_next - qbar_obs)


def _gcomp(
    histories: Sequence[FloatArray],
    treatments: Sequence[FloatArray],
    target_actions: Sequence[FloatArray],
    outcome: FloatArray,
    *,
    outcome_factory: OutcomeFactory,
) -> FloatArray:
    """In-sample iterated conditional expectations; returns the per-unit ``q_1`` plug-in."""
    horizon = len(histories)
    q = outcome
    for t in reversed(range(horizon)):
        h, a, a_pi = histories[t], treatments[t], target_actions[t]
        m = outcome_factory().fit(_with_action(h, a), q)
        q = _predict(m, _with_action(h, a_pi))
    return q


def sequential_ice_values(
    histories: Sequence[object],
    treatments: Sequence[object],
    target_actions: Sequence[object],
    outcome: object,
    *,
    outcome_model: OutcomeFactory | None = None,
) -> FloatArray:
    """Per-unit iterated-conditional-expectation values ``q_1(H_1^i)`` under the target policy.

    The g-computation backbone shared by :func:`estimate_sequential_value` (``gcomp``) and the
    identified sequential-transport subcase: the per-unit conditional policy value given the
    baseline history, *before* averaging — so a downstream caller may re-average it over a
    different (e.g. transported) baseline distribution.
    """
    hist = [_as2d(h) for h in histories]
    treat = [np.asarray(a, dtype=np.float64) for a in treatments]
    target = [np.asarray(a, dtype=np.float64) for a in target_actions]
    y = np.asarray(outcome, dtype=np.float64)
    _check_shapes(hist, treat, target, y)
    of = outcome_model or _default_outcome
    return _gcomp(hist, treat, target, y, outcome_factory=of)


def _fingerprint(outcome: FloatArray, treatments: Sequence[FloatArray]) -> str:
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(outcome).tobytes())
    for a in treatments:
        h.update(np.ascontiguousarray(np.asarray(a, dtype=np.float64)).tobytes())
    return h.hexdigest()[:16]


def estimate_sequential_value(
    histories: Sequence[object],
    treatments: Sequence[object],
    target_actions: Sequence[object],
    outcome: object,
    *,
    method: str = "dr",
    alpha: float = 0.05,
    n_folds: int = 5,
    seed: int = 0,
    outcome_model: OutcomeFactory | None = None,
    propensity_model: PropensityFactory | None = None,
    clip: float = 1e-3,
) -> SequentialValueEstimate:
    """Estimate a deterministic target policy's finite-horizon value under sequential ignorability.

    ``histories[t]`` is the stage-``t`` adjustment history ``H_t`` (shape ``(n,)`` or ``(n, d_t)``);
    ``treatments[t]`` / ``target_actions[t]`` are binary ``(n,)`` observed / target actions.
    ``method`` is ``"gcomp"`` (g-computation) or ``"dr"`` (cross-fitted sequentially doubly-robust).
    Nuisances are pluggable sklearn-style factories (defaults: ridge outcome, logistic propensity).
    """
    if method not in _METHODS:
        raise ValueError(f"method must be one of {_METHODS}, got {method!r}")
    hist = [_as2d(h) for h in histories]
    treat = [np.asarray(a, dtype=np.float64) for a in treatments]
    target = [np.asarray(a, dtype=np.float64) for a in target_actions]
    y = np.asarray(outcome, dtype=np.float64)
    horizon, n = _check_shapes(hist, treat, target, y)
    of = outcome_model or _default_outcome
    pf = propensity_model or _default_propensity

    if method == "gcomp":
        q1 = _gcomp(hist, treat, target, y, outcome_factory=of)
        value = float(q1.mean())
        se = float(q1.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
        overlap_by_step = [
            {"min_propensity": float("nan"), "max_propensity": float("nan")} for _ in range(horizon)
        ]
        return SequentialValueEstimate(
            value, se, _ci(value, se, alpha), alpha, n, horizon, method, overlap_by_step, None
        )

    idx: IntArray = np.arange(n, dtype=np.intp)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    q1 = np.empty(n, dtype=np.float64)
    overlaps: list[list[FloatArray]] = [[] for _ in range(horizon)]
    for te_raw in np.array_split(idx, n_folds):
        te = np.asarray(te_raw, dtype=np.intp)
        tr = np.asarray(np.setdiff1d(idx, te, assume_unique=True), dtype=np.intp)
        q1[te] = _dr_fold(
            tr,
            te,
            hist,
            treat,
            target,
            y,
            outcome_factory=of,
            propensity_factory=pf,
            clip=clip,
            overlaps=overlaps,
        )
    value = float(q1.mean())
    se = float(q1.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    overlap_by_step = [
        {
            "min_propensity": float(np.concatenate(o).min()),
            "max_propensity": float(np.concatenate(o).max()),
        }
        for o in overlaps
    ]
    return SequentialValueEstimate(
        value, se, _ci(value, se, alpha), alpha, n, horizon, method, overlap_by_step, n_folds
    )


def certify_sequential_value(
    histories: Sequence[object],
    treatments: Sequence[object],
    target_actions: Sequence[object],
    outcome: object,
    *,
    method: str = "dr",
    alpha: float = 0.05,
    n_folds: int = 5,
    seed: int = 0,
    outcome_model: OutcomeFactory | None = None,
    propensity_model: PropensityFactory | None = None,
    overlap_eps: float = 0.01,
    clip: float = 1e-3,
    policy: str = "pi",
) -> Certificate:
    """Certify a deterministic policy's finite-horizon value under sequential ignorability.

    Returns a ``kind=IDENTIFIED`` :class:`Certificate` with the point value (``value``), its
    influence-function confidence interval (``ci``), a per-stage ``sequential-ignorability``
    witness, and provenance — or a hedged certificate when any stage's estimated positivity falls
    below ``overlap_eps`` (I3). Sequential ignorability is a non-checkable assumption the caller
    asserts (identification from a graph is out of scope); it is recorded explicitly, never assumed
    silently.
    """
    est = estimate_sequential_value(
        histories,
        treatments,
        target_actions,
        outcome,
        method=method,
        alpha=alpha,
        n_folds=n_folds,
        seed=seed,
        outcome_model=outcome_model,
        propensity_model=propensity_model,
        clip=clip,
    )
    treat = [np.asarray(a, dtype=np.float64) for a in treatments]
    y = np.asarray(outcome, dtype=np.float64)
    fingerprint = _fingerprint(y, treat)

    for t, ov in enumerate(est.overlap_by_step):
        min_e = ov.get("min_propensity", float("nan"))
        max_e = ov.get("max_propensity", float("nan"))
        if np.isfinite(min_e) and (min_e < overlap_eps or max_e > 1.0 - overlap_eps):
            return Certificate(
                claim=f"V({policy}) over horizon {est.horizon} refused: overlap-violation",
                estimand=EstimandSpec(query="policy_value", target="mean", policy=policy),
                kind=Kind.IDENTIFIED,
                value=None,
                alpha=alpha,
                assumptions=(),
                method="refused",
                witness=None,
                hedge=Hedge(
                    reason="overlap-violation",
                    detail={"step": t, "overlap_eps": overlap_eps, **ov},
                ),
                provenance=Provenance.create(seeds=(seed,), data_fingerprint=fingerprint),
            )

    xfit = f" (cross-fit K={est.n_folds})" if est.n_folds else ""
    return Certificate(
        claim=f"V({policy}) over horizon {est.horizon} = {est.value:.4g}",
        estimand=EstimandSpec(query="policy_value", target="mean", policy=policy),
        kind=Kind.IDENTIFIED,
        value=est.value,
        alpha=alpha,
        assumptions=(
            Assumption(
                name="sequential-ignorability",
                params={"horizon": est.horizon, "adjustment": "per-stage H_t"},
                checkable=False,
            ),
            Assumption(
                name="overlap",
                params={"eps": overlap_eps},
                checkable=True,
                diagnostic={"by_step": est.overlap_by_step},
            ),
        ),
        method=f"sequential-{est.method}{xfit}",
        witness=Witness(
            kind="sequential-adjustment",
            detail={"horizon": est.horizon, "stages": [f"H_{t}" for t in range(est.horizon)]},
        ),
        hedge=None,
        provenance=Provenance.create(seeds=(seed,), data_fingerprint=fingerprint),
        ci=est.ci,
    )
