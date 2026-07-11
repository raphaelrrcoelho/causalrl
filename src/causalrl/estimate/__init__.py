"""Identification-aware estimation of identified functionals (plan §7.2).

The continuous/DR counterpart to the shipped discrete estimand evaluator
(:func:`causalrl.identification.id_algorithm.estimate_effect`): it compiles a graph query into a
back-door adjustment plan and estimates it with plug-in, self-normalised IPW, AIPW, or cross-fitted
DML, returning a unified :class:`~causalrl.certify.certificate.Certificate`. Non-identifiable or
unsupported queries are hedged, never silently point-estimated.
"""

from causalrl.estimate.compiler import (
    EstimandNotSupportedError,
    EstimatorPlan,
    certify_effect,
    compile_estimand,
)
from causalrl.estimate.estimators import EffectEstimate, estimate_ate
from causalrl.estimate.nuisance import (
    Classifier,
    LogisticRegressor,
    Regressor,
    RidgeRegressor,
)
from causalrl.estimate.streaming import stream_policy_value, stream_quantile_certificate

__all__ = [
    "Classifier",
    "EffectEstimate",
    "EstimandNotSupportedError",
    "EstimatorPlan",
    "LogisticRegressor",
    "Regressor",
    "RidgeRegressor",
    "certify_effect",
    "compile_estimand",
    "estimate_ate",
    "stream_policy_value",
    "stream_quantile_certificate",
]
