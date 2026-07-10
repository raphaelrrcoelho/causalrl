"""Unified certificate protocol (plan §5.2; invariants I1-I3).

One serializable :class:`Certificate` type. The shipped bespoke certificates adapt into it via
``as_certificate`` (see :mod:`causalrl.certify.adapters`), and every new inferential routine returns
it. Leaf package: imports only bounds/graphs primitives, never interop/scale/experimental.
"""

from __future__ import annotations

from causalrl.certify.adapters import as_certificate
from causalrl.certify.certificate import (
    Assumption,
    Certificate,
    EstimandSpec,
    Hedge,
    Kind,
    Provenance,
    Witness,
)
from causalrl.certify.routines import (
    identify_effect_certified,
    ipw_sensitivity_bounds_certified,
    msm_policy_value_bounds_certified,
)

__all__ = [
    "Assumption",
    "Certificate",
    "EstimandSpec",
    "Hedge",
    "Kind",
    "Provenance",
    "Witness",
    "as_certificate",
    "identify_effect_certified",
    "ipw_sensitivity_bounds_certified",
    "msm_policy_value_bounds_certified",
]
