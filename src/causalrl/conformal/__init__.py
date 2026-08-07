"""Finite-sample conformal wrappers (plan §7.4).

Distribution-free prediction intervals with marginal coverage at least ``1 - alpha``: split
conformal, conformalized quantile regression, and weighted conformal for covariate shift, plus a
``kind=EMPIRICAL`` certificate wrapper. :func:`conformal_action_value` is the off-policy caller —
the calibration likelihood ratio is the propensity ratio ``pi_target / pi_behavior`` read off a
:class:`~causalrl.ConfoundedTrajectoryDataset` — and is what :func:`causalrl.certify_policy` gates
on for safe policy improvement.
"""

from causalrl.conformal.core import (
    certify_conformal_interval,
    conformal_action_value,
    conformal_quantile,
    cqr_interval,
    split_conformal_interval,
)

__all__ = [
    "certify_conformal_interval",
    "conformal_action_value",
    "conformal_quantile",
    "cqr_interval",
    "split_conformal_interval",
]
