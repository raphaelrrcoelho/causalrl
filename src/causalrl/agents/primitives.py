from causalrl.data.dataset import ConfoundedTrajectoryDataset
from causalrl.ope.bounds import causal_q_bounds


def bounds_table(
    dataset: ConfoundedTrajectoryDataset,
) -> dict[tuple[int, int], tuple[float, float]]:
    """Manski [lower, upper] bound on E[R|do(a), s] for every (state, action)."""
    return {
        (s, a): causal_q_bounds(dataset, s, a)
        for s in range(dataset.n_states)
        for a in range(dataset.n_actions)
    }


def non_dominated_actions(dataset: ConfoundedTrajectoryDataset, state: int) -> list[int]:
    """Actions whose upper bound is >= the best lower bound at `state`.

    An action a is dominated when its upper bound < max_a' lower(a'); such actions cannot be
    optimal and may be dropped before online exploration. (With Manski natural bounds this is
    typically a no-op, returning all actions — a correct safety mechanism, not a bug.)
    """
    bounds = [causal_q_bounds(dataset, state, a) for a in range(dataset.n_actions)]
    best_lower = max(lo for lo, _ in bounds)
    return [a for a, (_, hi) in enumerate(bounds) if hi >= best_lower]
