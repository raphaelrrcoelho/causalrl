"""Bridge causalrl offline logs into a d3rlpy dataset (lazy import of the optional d3rlpy).

d3rlpy trains; causalrl certifies. Convert your :class:`~causalrl.ConfoundedTrajectoryDataset` or a
:class:`~causalrl.data.trajectory.TrajectoryLog` to a d3rlpy ``MDPDataset`` (and back), train any
d3rlpy offline algo on it, hand the learned policy's greedy actions (:func:`policy_actions`, a
``do()`` policy handle) to :func:`causalrl.certify_policy`, and — for a model-based off-policy value
— wrap a fitted-Q evaluation with :func:`certify_fqe` (an honest ``EMPIRICAL`` certificate). Deepens
the shipped one-way tabular bridge to both directions and onto the unified ``Certificate`` (plan
§10); d3rlpy stays an optional ``causalrl[scale]`` extra, imported lazily.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from causalrl.certify.certificate import (
    Assumption,
    Certificate,
    EstimandSpec,
    Kind,
    Provenance,
)
from causalrl.data.dataset import ConfoundedTrajectoryDataset, Transition
from causalrl.data.trajectory import TrajectoryLog


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


def mdp_dataset_to_confounded_dataset(
    mdp: Any, *, n_actions: int | None = None
) -> ConfoundedTrajectoryDataset:
    """Reverse of :func:`to_mdp_dataset`: a d3rlpy ``MDPDataset`` back to a causalrl dataset.

    One-hot observations are decoded to integer states (``argmax`` over the observation width);
    ``next_state`` is the next row's state in-episode (a terminal row points to itself).
    ``n_actions`` defaults to ``max(action) + 1``. Requires no d3rlpy import — it reads the plain
    ``observations`` / ``actions`` / ``rewards`` / ``terminals`` arrays any ``MDPDataset`` exposes.
    """
    observations = np.asarray(mdp.observations, dtype=np.float64)
    actions = np.asarray(mdp.actions).reshape(-1)
    rewards = np.asarray(mdp.rewards, dtype=np.float64).reshape(-1)
    terminals = np.asarray(mdp.terminals, dtype=np.float64).reshape(-1)
    n_states = int(observations.shape[1])
    states = observations.argmax(axis=1)
    n = int(actions.shape[0])
    transitions: list[Transition] = []
    for i in range(n):
        done = bool(terminals[i] > 0.5)
        next_state = int(states[i + 1]) if (i + 1 < n and not done) else int(states[i])
        transitions.append(
            Transition(int(states[i]), int(actions[i]), float(rewards[i]), next_state, done)
        )
    resolved_actions = n_actions if n_actions is not None else (int(actions.max()) + 1 if n else 0)
    return ConfoundedTrajectoryDataset(transitions, n_states=n_states, n_actions=resolved_actions)


def trajectory_log_to_mdp_dataset(log: TrajectoryLog) -> Any:
    """Convert a :class:`~causalrl.data.trajectory.TrajectoryLog` to a d3rlpy ``MDPDataset``.

    Composes the shipped lossless ``TrajectoryLog`` → ``ConfoundedTrajectoryDataset`` bridge with
    :func:`to_mdp_dataset`; the log must carry the ``n_states`` / ``n_actions`` metadata that bridge
    requires.
    """
    return to_mdp_dataset(log.to_confounded_dataset())


def mdp_dataset_to_trajectory_log(mdp: Any, *, n_actions: int | None = None) -> TrajectoryLog:
    """Convert a d3rlpy ``MDPDataset`` to a :class:`~causalrl.data.trajectory.TrajectoryLog`.

    The reverse of :func:`trajectory_log_to_mdp_dataset`, via
    :func:`mdp_dataset_to_confounded_dataset` and the shipped columnar bridge.
    """
    return TrajectoryLog.from_confounded_dataset(
        mdp_dataset_to_confounded_dataset(mdp, n_actions=n_actions)
    )


def policy_actions(policy: Any, observations: Any) -> list[int]:
    """Greedy actions a trained (duck-typed) policy takes on ``observations`` — a ``do()`` handle.

    Calls the policy's ``predict`` (d3rlpy's algorithm interface); returns integer actions ready
    to pass as ``target_actions`` to :func:`causalrl.certify_policy`, or as a policy intervention.
    Never imports d3rlpy; any object with a ``predict(observations)`` method works.
    """
    predictions = np.asarray(policy.predict(np.asarray(observations)))
    return [int(a) for a in predictions.reshape(-1).tolist()]


def certify_fqe(
    estimated_value: float, *, policy: str | None = None, method: str = "FQE"
) -> Certificate:
    """Wrap a fitted-Q-evaluation off-policy value estimate in an honest ``EMPIRICAL`` certificate.

    Fitted-Q evaluation (d3rlpy's ``FQE``) is a model-based off-policy value estimate with no
    identification guarantee under hidden confounding, so the certificate is ``kind=EMPIRICAL``
    (I2/I3) — it reports the estimate and its provenance, never a causal point claim. Upgrade to an
    identified ``certify_effect`` (§7.2) or bound it with :func:`causalrl.certify_policy` when the
    structure licenses it. ``estimated_value`` is the scalar FQE value (e.g. the initial-state value
    a trained ``FQE`` predicts for the evaluated policy).
    """
    value = float(estimated_value)
    return Certificate(
        claim=f"FQE off-policy value estimate V(pi) ~= {value:.4g}",
        estimand=EstimandSpec(query="policy_value", target="mean", policy=policy),
        kind=Kind.EMPIRICAL,
        value=value,
        alpha=None,
        assumptions=(Assumption(name="fqe-model", params={"method": method}, checkable=False),),
        method=f"fitted-Q evaluation ({method})",
        witness=None,
        hedge=None,
        provenance=Provenance.create(),
        ci=None,
    )
