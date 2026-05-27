import warnings

from causalrl.experimental.ope import confounding_sensitivity_bounds as _sensitivity_bounds


def ipw_value(
    actions: list[int],
    rewards: list[float],
    behavior_probs: list[float],
    target_probs: list[float],
) -> float:
    """Inverse-propensity-weighted off-policy value estimate.

    Samples with zero behavior propensity (``b == 0``) contribute 0 to the sum but are
    still counted in the ``n`` denominator, which biases the estimate toward 0 — pass only
    samples with positive behavior propensity for an unbiased estimate.
    """
    n = len(actions)
    total = 0.0
    for _a, r, b, t in zip(actions, rewards, behavior_probs, target_probs, strict=True):
        total += (t / b) * r if b > 0 else 0.0
    return total / n


def confounding_sensitivity_bounds(point: float, gamma: float) -> tuple[float, float]:
    """Deprecated bridge to :func:`causalrl.experimental.ope.confounding_sensitivity_bounds`.

    The helper is qualitative rather than a validated published estimator and therefore no
    longer belongs to the stable evaluation API.
    """
    warnings.warn(
        "confounding_sensitivity_bounds moved to causalrl.experimental.ope; "
        "it is a qualitative helper, not a validated estimator",
        DeprecationWarning,
        stacklevel=2,
    )
    return _sensitivity_bounds(point, gamma)
