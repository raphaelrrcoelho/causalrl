"""Finite-sample conformal wrappers (plan §7.4).

Distribution-free prediction intervals with marginal coverage at least ``1 - alpha``: split
conformal, conformalized quantile regression, and weighted conformal for covariate shift /
off-policy targets, plus a ``kind=EMPIRICAL`` certificate wrapper.
"""

from causalrl.conformal.core import (
    certify_conformal_interval,
    conformal_quantile,
    cqr_interval,
    split_conformal_interval,
)

__all__ = [
    "certify_conformal_interval",
    "conformal_quantile",
    "cqr_interval",
    "split_conformal_interval",
]
