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
    """A monotone confounding-sensitivity interval around `point`, parameterised by gamma.

    gamma = 1 returns the point estimate (no unobserved confounding); larger gamma widens
    the interval (by ``(gamma-1)/(gamma+1)``), clipped to [0, 1] for bounded (e.g.
    Bernoulli) rewards.

    SCOPE (v0.1): this is a monotone STAND-IN with the right qualitative shape, NOT the
    published marginal-sensitivity-model bound (Kallus-Zhou / Tan's Gamma). Do not report
    these as MSM bounds; a faithful estimator is deferred to a later version.
    """
    if gamma < 1.0:
        raise ValueError("gamma must be >= 1")
    half_width = (gamma - 1.0) / (gamma + 1.0)
    lo = max(0.0, point - half_width)
    hi = min(1.0, point + half_width)
    return lo, hi
