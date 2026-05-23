def cumulative_regret(rewards: list[float], optimal_per_step: float) -> float:
    """Total regret = sum over steps of (optimal expected reward - realized reward)."""
    return sum(optimal_per_step - r for r in rewards)


def finite_horizon_regret(returns: list[float], optimal_return: float) -> float:
    """Total regret over episodes = sum of (optimal episode return - realized return)."""
    return sum(optimal_return - r for r in returns)
