"""certify_decision: a one-call decision-certificate front door over the decision stack.

Given confounded / off-policy logs of a binary decision — "is the treated arm better than the
control arm?" — report whether that decision is robust to hidden confounding. It is an ergonomic
orchestrator, not new theory: it composes the documented decision stack from
:mod:`causalrl.identification.bounds` — the cheap sign-robustness certificate
(:func:`pivotality_certificate`) and, when logging propensities are supplied, the
marginal-sensitivity-model tipping point (:func:`tipping_gamma` over
:func:`msm_contribution_bounds`) — into a single call with a human-readable verdict.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

import numpy as np

from causalrl.identification.bounds import (
    Interval,
    PivotalityCertificate,
    msm_contribution_bounds,
    pivotality_certificate,
    tipping_gamma,
)
from causalrl.identification.estimate import PolicyValueContrast


class DecisionCertificate(NamedTuple):
    """Result of :func:`certify_decision`.

    ``certified`` is the headline: is the naive (logged) decision robust to hidden confounding,
    by the strongest layer that ran? When the structural/measured layer ran it carries its
    one-sided guarantee (no confounder consistent with the supplied information can flip the
    sign); otherwise it reports whether the MSM layer found the decision robust up to ``gamma_max``.
    The component fields and ``summary`` make the exact guarantee explicit.
    """

    decision: str  # "prefer treated" | "prefer control" | "indifferent" (sign of naive_contrast)
    naive_contrast: float  # E[Y | F=1] - E[Y | F=0]
    certified: bool
    pivotality: PivotalityCertificate | None  # structural/measured sign-robustness layer, if run
    tipping_gamma: float | None  # MSM odds-ratio at which the decision tips; None if not run/robust
    msm_certified: bool | None  # MSM layer robust to gamma_max? None if the MSM layer did not run
    summary: str

    def __str__(self) -> str:
        return self.summary


def certify_estimate(
    estimate: PolicyValueContrast,
    *,
    gamma_max: float = 10.0,
    labels: tuple[str, str] = ("pi_on", "pi_off"),
) -> DecisionCertificate:
    """Certify whether an off-policy value contrast's sign is robust to hidden confounding.

    Runs the MSM tipping layer over the general ``V(pi_on) - V(pi_off)`` contribution bound when
    ``estimate`` carries logging propensities, and the structural pivotality layer when it carries a
    binary-arm reduction (``treated`` + ``confounder_bins`` / ``mi_cap``). Returns the same
    :class:`DecisionCertificate` as :func:`certify_decision`; ``labels`` names the two policies in
    the verdict (``labels[0]`` when the contrast is positive).

    Honest scope: the MSM sensitivity is on the logging propensities (sharp when the two target
    supports are disjoint, valid-but-conservative otherwise); the pivotality layer is defined only
    for the binary-arm contrast. No sensitivity claim is made for arbitrary outcome-model
    estimators.
    """
    y = np.asarray(estimate.outcomes, dtype=float)

    treated = estimate.treated
    pivot: PivotalityCertificate | None = None
    if treated is not None and (
        estimate.confounder_bins is not None or estimate.mi_cap is not None
    ):
        pivot = pivotality_certificate(
            estimate.outcomes, treated, estimate.confounder_bins, mi_cap=estimate.mi_cap
        )

    if treated is not None:
        fb = np.asarray(treated).astype(bool)
        naive = float(y[fb].mean() - y[~fb].mean())
    else:
        naive = 0.0  # replaced from the MSM point below (a target-only contrast always has MSM)

    g_tip: float | None = None
    msm_certified: bool | None = None
    logging_propensities = estimate.logging_propensities
    if logging_propensities is not None:
        on, off = estimate.target_on, estimate.target_off
        assert on is not None and off is not None  # guaranteed by PolicyValueContrast.__post_init__
        outcomes = estimate.outcomes
        e0_l, on_l, off_l = list(logging_propensities), list(on), list(off)

        def _band(g: float) -> Interval:
            return msm_contribution_bounds(outcomes, e0_l, on_l, off_l, gamma=g)

        if treated is None:
            naive = float(_band(1.0).lower)
        g_tip = tipping_gamma(_band, reference=0.0, gamma_max=gamma_max)
        msm_certified = g_tip is None

    decision = (
        f"prefer {labels[0]}"
        if naive > 0
        else f"prefer {labels[1]}"
        if naive < 0
        else "indifferent"
    )
    certified = pivot.certified if pivot is not None else bool(msm_certified)
    summary = _summarise(decision, naive, pivot, g_tip, msm_certified, gamma_max)
    return DecisionCertificate(
        decision=decision,
        naive_contrast=naive,
        certified=certified,
        pivotality=pivot,
        tipping_gamma=g_tip,
        msm_certified=msm_certified,
        summary=summary,
    )


def certify_decision(
    outcomes: Sequence[float],
    treated: Sequence[int],
    *,
    confounder_bins: Sequence[int] | None = None,
    mi_cap: float | None = None,
    propensities: Sequence[float] | None = None,
    gamma_max: float = 10.0,
) -> DecisionCertificate:
    """Certify whether a binary decision from confounded logs is robust to hidden confounding.

    ``outcomes`` are logged rewards ``Y_i``; ``treated`` is the binary arm indicator (1 = the
    arm under test, 0 = baseline). The naive decision is the sign of the logged contrast
    ``E[Y | F=1] - E[Y | F=0]``. Supply at least one evidence source — they compose:

    * ``confounder_bins`` (a measured hidden variable ``Z``) or ``mi_cap`` (a structural cap on
      the information channel ``MI(I; Z)``) runs the **sign-robustness layer**
      (:func:`pivotality_certificate`): certifies iff no hidden confounder consistent with that
      information can flip the contrast's sign. One-sided — failure to certify is not evidence of
      a flip.
    * ``propensities`` (the nominal logging propensities ``e0(a_i | x_i)`` at the logged action)
      runs the **MSM tipping layer**: the smallest Tan odds-ratio ``Gamma`` at which the
      off-policy (IPS) value contrast band first includes zero (:func:`tipping_gamma` over the
      sharp one-hot :func:`msm_contribution_bounds`). ``tipping_gamma is None`` means the decision
      is robust to confounding at least as strong as ``gamma_max``.

    With informative propensities the MSM layer concerns the inverse-propensity-weighted
    off-policy contrast, which coincides with the raw logged contrast only under uniform logging.
    """
    y = np.asarray(outcomes, dtype=float)
    f = np.asarray(treated)
    fb = f.astype(bool)
    if not (fb.any() and (~fb).any()):
        raise ValueError("both arms must be present in `treated`")
    if confounder_bins is None and mi_cap is None and propensities is None:
        raise ValueError(
            "supply at least one evidence source: confounder_bins (measured Z), "
            "mi_cap (structural channel cap), or propensities (MSM sensitivity)"
        )

    naive = float(y[fb].mean() - y[~fb].mean())
    decision = "prefer treated" if naive > 0 else "prefer control" if naive < 0 else "indifferent"

    pivot: PivotalityCertificate | None = None
    if confounder_bins is not None or mi_cap is not None:
        pivot = pivotality_certificate(outcomes, treated, confounder_bins, mi_cap=mi_cap)

    g_tip: float | None = None
    msm_certified: bool | None = None
    if propensities is not None:
        on = fb.astype(float).tolist()
        off = (~fb).astype(float).tolist()
        logging_props = list(propensities)  # concrete (non-Optional) capture for the closure

        def _band(g: float) -> Interval:
            return msm_contribution_bounds(outcomes, logging_props, on, off, gamma=g)

        g_tip = tipping_gamma(_band, reference=0.0, gamma_max=gamma_max)
        msm_certified = g_tip is None

    certified = pivot.certified if pivot is not None else bool(msm_certified)
    summary = _summarise(decision, naive, pivot, g_tip, msm_certified, gamma_max)
    return DecisionCertificate(
        decision=decision,
        naive_contrast=naive,
        certified=certified,
        pivotality=pivot,
        tipping_gamma=g_tip,
        msm_certified=msm_certified,
        summary=summary,
    )


def _summarise(
    decision: str,
    naive: float,
    pivot: PivotalityCertificate | None,
    g_tip: float | None,
    msm_certified: bool | None,
    gamma_max: float,
) -> str:
    parts = [f"{decision} (naive contrast {naive:+.3f})."]
    if pivot is not None:
        if pivot.certified:
            parts.append(
                f"Sign-robust to hidden confounding: CERTIFIED "
                f"(omitted-variable bias ≤ {pivot.bias_bound:.3f} < |{naive:+.3f}|)."
            )
        else:
            parts.append(
                f"Sign-robust to hidden confounding: NOT certified "
                f"(bias bound {pivot.bias_bound:.3f} ≥ {abs(naive):.3f}; one-sided test, "
                "so this is not evidence of a flip)."
            )
    if msm_certified is not None:
        if msm_certified:
            parts.append(f"Off-policy (MSM) decision robust to confounding up to Γ={gamma_max:g}.")
        elif g_tip == 1.0:
            parts.append(
                "Off-policy (MSM) contrast already brackets zero at Γ=1 "
                "(no decision even without confounding)."
            )
        else:
            parts.append(
                f"Off-policy (MSM) decision tips at Γ≈{g_tip:.2f}: unmeasured confounding "
                "with at least that odds-ratio could overturn it."
            )
    return " ".join(parts)
