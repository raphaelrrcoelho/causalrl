"""Off-policy evaluation utilities."""


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
