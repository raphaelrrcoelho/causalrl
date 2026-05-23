def cumulative_regret(rewards: list[float], optimal_per_step: float) -> float:
    """Total regret = sum over steps of (optimal expected reward - realized reward)."""
    return sum(optimal_per_step - r for r in rewards)
