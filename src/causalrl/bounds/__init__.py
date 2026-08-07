"""Partial-identification bounds (plan §4 leaf package).

One ``causalrl.bounds`` home that re-exports the shipped nominal-propensity kernels from
:mod:`causalrl.identification.bounds` alongside the continuous / estimated-propensity addition in
:mod:`causalrl.bounds.continuous`. The shipped module stays the source of truth; this package adds
beside it without moving anything (re-export shim, per plan §4).
"""

from causalrl.bounds.continuous import (
    certify_sensitivity_bounds,
    msm_sensitivity_bounds,
)
from causalrl.bounds.streaming import stream_msm_bounds
from causalrl.identification.bounds import (
    Interval,
    causal_q_bounds,
    ipw_sensitivity_bounds,
    manski_bounds,
    msm_contribution_bounds,
    msm_per_step_bounds,
    msm_policy_value_bounds,
    msm_stratified_bounds,
    tipping_gamma,
)

__all__ = [
    "Interval",
    "causal_q_bounds",
    "certify_sensitivity_bounds",
    "ipw_sensitivity_bounds",
    "manski_bounds",
    "msm_contribution_bounds",
    "msm_per_step_bounds",
    "msm_policy_value_bounds",
    "msm_sensitivity_bounds",
    "msm_stratified_bounds",
    "stream_msm_bounds",
    "tipping_gamma",
]
