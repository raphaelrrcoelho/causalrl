"""Bridge causalrl offline logs into a d3rlpy dataset (lazy import of the optional d3rlpy).

d3rlpy trains; causalrl certifies. Convert your :class:`~causalrl.ConfoundedTrajectoryDataset` to a
d3rlpy ``MDPDataset``, train any d3rlpy offline algo on it, then hand the learned policy's greedy
actions to :func:`causalrl.certify_policy`.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from causalrl.data.dataset import ConfoundedTrajectoryDataset


def to_mdp_dataset(dataset: ConfoundedTrajectoryDataset) -> Any:
    """Convert a causalrl :class:`~causalrl.ConfoundedTrajectoryDataset` to a d3rlpy ``MDPDataset``.

    Tabular integer states are one-hot encoded to float observations d3rlpy can train on; actions,
    rewards, and terminal flags carry over. Requires the ``causalrl[scale]`` extra (d3rlpy); the
    import is lazy so causalrl core never depends on d3rlpy.
    """
    # d3rlpy is an optional dependency (causalrl[scale]); import lazily inside the function.
    import d3rlpy  # type: ignore

    transitions = dataset.transitions
    n = len(transitions)
    observations = np.zeros((n, dataset.n_states), dtype=np.float32)
    actions = np.empty(n, dtype=np.int64)
    rewards = np.empty(n, dtype=np.float32)
    terminals = np.empty(n, dtype=np.float32)
    for i, tr in enumerate(transitions):
        observations[i, tr.state] = 1.0
        actions[i] = tr.action
        rewards[i] = tr.reward
        terminals[i] = 1.0 if tr.done else 0.0
    return d3rlpy.dataset.MDPDataset(  # type: ignore
        observations=observations,
        actions=actions,
        rewards=rewards,
        terminals=terminals,
    )
