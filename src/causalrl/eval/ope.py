from __future__ import annotations


def ipw_value(
    actions: list[int],
    rewards: list[float],
    behavior_probs: list[float],
    target_probs: list[float],
) -> float:
    """Inverse-propensity-weighted off-policy value estimate."""
    n = len(actions)
    total = 0.0
    for _a, r, b, t in zip(actions, rewards, behavior_probs, target_probs, strict=True):
        total += (t / b) * r if b > 0 else 0.0
    return total / n


def confounding_sensitivity_bounds(point: float, gamma: float) -> tuple[float, float]:
    """Manski-style interval under a confounding strength gamma >= 1.

    gamma = 1 returns the point estimate (no unobserved confounding). Larger gamma widens
    the interval, clipped to [0, 1] for bounded (e.g. Bernoulli) rewards.
    """
    if gamma < 1.0:
        raise ValueError("gamma must be >= 1")
    half_width = (gamma - 1.0) / (gamma + 1.0)
    lo = max(0.0, point - half_width)
    hi = min(1.0, point + half_width)
    return lo, hi
