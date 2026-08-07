"""Adapt the shipped bespoke certificates onto the unified :class:`Certificate` (§5.2, I1).

Additive only: the shipped ``DecisionCertificate`` / ``PivotalityCertificate`` /
``TransportRegretCertificate`` are unchanged; :func:`as_certificate` produces a conforming view.
Dispatch lives here (not on the shipped types) so the dependency runs certify -> identification one
way, without a cycle.
"""

from __future__ import annotations

from typing import Any

from causalrl.certify.certificate import (
    Assumption,
    Certificate,
    EstimandSpec,
    Hedge,
    Kind,
    Provenance,
    Witness,
)
from causalrl.identification.bounds import Interval, PivotalityCertificate
from causalrl.identification.decision import DecisionCertificate
from causalrl.identification.transport_regret import TransportRegretCertificate


def _pivotality_to_certificate(c: PivotalityCertificate) -> Certificate:
    params: dict[str, Any] = {"mi_flip": c.mi_flip, "mi_measured": c.mi_measured}
    return Certificate(
        claim=(
            "naive decision sign is robust to hidden confounding (pivotality)"
            if c.certified
            else "naive decision sign may flip under hidden confounding"
        ),
        estimand=EstimandSpec(query="policy_value", target="mean"),
        kind=Kind.BOUNDED,
        value=c.bias_bound,
        alpha=None,
        assumptions=(Assumption("mi-cap", params, checkable=c.mi_measured is not None),),
        method="pivotality_certificate",
        witness=(
            Witness("pivotality", {"naive": c.naive, "bias_bound": c.bias_bound})
            if c.certified
            else None
        ),
        hedge=(
            None
            if c.certified
            else Hedge("bias-may-exceed-naive", {"naive": c.naive, "bias_bound": c.bias_bound})
        ),
        provenance=Provenance.create(),
    )


def _decision_to_certificate(c: DecisionCertificate) -> Certificate:
    assumptions: list[Assumption] = []
    if c.tipping_gamma is not None or c.msm_certified is not None:
        msm: dict[str, Any] = {"tipping_gamma": c.tipping_gamma, "msm_certified": c.msm_certified}
        assumptions.append(Assumption("MSM", msm))
    if c.pivotality is not None:
        mi: dict[str, Any] = {
            "mi_flip": c.pivotality.mi_flip,
            "mi_measured": c.pivotality.mi_measured,
        }
        assumptions.append(Assumption("mi-cap", mi, checkable=c.pivotality.mi_measured is not None))
    if c.conformal_lcb is not None:
        assumptions.append(
            Assumption(
                "weighted-exchangeability",
                {"conformal_lcb": c.conformal_lcb},
                checkable=False,
            )
        )
    # Which layer refused? The confounding layer's own verdict, exactly as certify_estimate
    # computes it — so a decision refused only by the finite-sample gate is not mislabelled
    # "not robust to confounding".
    confounding_ok = c.pivotality.certified if c.pivotality is not None else bool(c.msm_certified)
    return Certificate(
        claim=c.summary,
        estimand=EstimandSpec(query="policy_value", target="mean"),
        kind=Kind.BOUNDED,
        value=c.naive_contrast,
        alpha=None,
        assumptions=tuple(assumptions),
        method="certify_decision",
        witness=None,
        # recommendation == "act" iff certified; abstain surfaces as a hedge (I3).
        hedge=(
            None
            if c.certified
            else Hedge(
                "not-robust-to-confounding" if not confounding_ok else "downside-not-certified",
                {
                    "decision": c.decision,
                    "tipping_gamma": c.tipping_gamma,
                    "conformal_lcb": c.conformal_lcb,
                },
            )
        ),
        provenance=Provenance.create(),
    )


def _transport_regret_to_certificate(c: TransportRegretCertificate) -> Certificate:
    witness_vars = sorted(c.non_transportable_witness)
    zero_regret = c.transportable and c.regret_bound == Interval(0.0, 0.0)
    if zero_regret:
        claim = "queried effect transports with zero transfer regret"
    elif c.transportable:
        claim = "queried effect transports with bounded transfer regret"
    else:
        claim = "queried effect is not transportable; regret bounded by policy value range"
    detail: dict[str, Any] = {
        "formula": None if c.formula is None else str(c.formula),
        "non_transportable_witness": witness_vars,
    }
    hedge_detail: dict[str, Any] = {
        "non_transportable_witness": witness_vars,
        "regret_upper": c.regret_bound.upper,
        "decision_dependence": c.decision_dependence,
    }
    return Certificate(
        claim=claim,
        estimand=EstimandSpec(query="transport", target="mean"),
        kind=Kind.BOUNDED,
        value=c.regret_bound,
        alpha=None,
        assumptions=(
            Assumption(
                "selection-nodes-S",
                {"witness": witness_vars, "reweight_required": c.reweight_required},
            ),
        ),
        method="transport_regret_certificate",
        witness=Witness("transport-formula", detail),
        hedge=(
            None
            if zero_regret
            else Hedge("transfer-regret" if c.transportable else "non-transportable", hedge_detail)
        ),
        provenance=Provenance.create(),
    )


def as_certificate(obj: object) -> Certificate:
    """Convert a shipped certificate to a unified :class:`Certificate` (additive view).

    Accepts ``DecisionCertificate``, ``PivotalityCertificate``, ``TransportRegretCertificate``, or a
    ``Certificate`` (returned unchanged). Raises :class:`TypeError` for anything else.
    """
    if isinstance(obj, Certificate):
        return obj
    if isinstance(obj, DecisionCertificate):
        return _decision_to_certificate(obj)
    if isinstance(obj, PivotalityCertificate):
        return _pivotality_to_certificate(obj)
    if isinstance(obj, TransportRegretCertificate):
        return _transport_regret_to_certificate(obj)
    raise TypeError(f"cannot adapt {type(obj).__name__} to a Certificate")
