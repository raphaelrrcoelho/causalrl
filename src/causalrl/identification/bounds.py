from causalrl.data.dataset import ConfoundedTrajectoryDataset
from causalrl.exceptions import NotIdentifiableError


def causal_q_bounds(
    dataset: ConfoundedTrajectoryDataset,
    state: int,
    action: int,
    *,
    require_identified: bool = False,
) -> tuple[float, float]:
    """Manski natural bounds on E[return | do(action), state] from confounded logs.

    For a return in [0, 1] with empirical mean m = E[R|s,a] and propensity p = P(a|s):
        lower = m * p,  upper = m * p + (1 - p).
    A never-logged action (p = 0) yields the vacuous [0, 1] — not identifiable from the
    logs alone. With `require_identified=True`, a vacuous bound raises NotIdentifiableError
    carrying (state, action) as the witness.
    """
    p = dataset.behavior_propensity(state, action)
    m = dataset.mean_reward(state, action)
    lower = m * p
    upper = m * p + (1.0 - p)
    if require_identified and p == 0.0:
        raise NotIdentifiableError(
            f"E[R|do(a={action}), s={state}] is not identifiable: action never logged "
            f"in this state (vacuous bound [0, 1])",
            witness=(state, action),
        )
    return lower, upper
