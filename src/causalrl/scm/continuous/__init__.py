"""Continuous causal core (plan §7.1): neural / invertible mechanisms + posterior abduction.

torch-backed (the ``[torch]`` extra). Additive-noise MLP and invertible location-scale mechanisms;
exact abduction for the invertible family and an amortized-VI black-box path, both feeding the
Phase-0 ``NoisePosterior`` protocol.
"""

from causalrl.scm.continuous.abduction import (
    AmortizedGaussianAbduction,
    AmortizedNoisePosterior,
    InvertibleMechanism,
    PointNoisePosterior,
    abduct_invertible,
    abduct_location_scale,
    certify_counterfactual,
    posterior_predictive_check,
)
from causalrl.scm.continuous.mechanisms import (
    ConditionalFlowMechanism,
    LocationScaleMechanism,
    MLPMechanism,
)

__all__ = [
    "AmortizedGaussianAbduction",
    "AmortizedNoisePosterior",
    "ConditionalFlowMechanism",
    "InvertibleMechanism",
    "LocationScaleMechanism",
    "MLPMechanism",
    "PointNoisePosterior",
    "abduct_invertible",
    "abduct_location_scale",
    "certify_counterfactual",
    "posterior_predictive_check",
]
