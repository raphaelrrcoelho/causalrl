# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownMemberType=false
"""Causal-Gymnasium (CausalAILab) → :class:`~causalrl.data.dataset.ConfoundedTrajectoryDataset`
adapter (DESIGN §7).

Rolls out any Gymnasium-API env under a supplied behavior policy and logs each step into a tabular
``ConfoundedTrajectoryDataset`` — the offline structure the causal-MBRL probes and the certificate
layer consume. Fully duck-typed: the external ``causal_gym`` package is *never imported* (matching
the PettingZoo / DoWhy / EconML interop house style), so any Gymnasium-shaped env (a real
Causal-Gymnasium benchmark or a lightweight stand-in) drives it unchanged. Install Causal-Gymnasium
yourself to run it against a real benchmark.

Data-level only: it consumes behavior-policy rollouts, not live SCM surgery. A CausalGym env's
``info`` carries the realized natural action and hidden confounders for oracle checks; those are
available to the caller but not stored in the tabular dataset (which keys on the discrete state).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from causalrl.data.dataset import ConfoundedTrajectoryDataset, Transition

__all__ = ["from_causal_gym"]

# behavior_policy(observation) -> discrete action; state_fn(observation) -> discrete state index
BehaviorPolicy = Callable[[Any], int]
StateFn = Callable[[Any], int]


def _default_state_index(obs: Any) -> int:
    """Map an observation to a discrete state: an int, ``obs['state']``, or a size-1 array."""
    if isinstance(obs, dict):
        if "state" in obs:
            return int(obs["state"])
        raise ValueError("dict observation has no 'state' key; pass state_fn")
    arr = np.asarray(obs)
    if arr.ndim == 0 or arr.size == 1:
        return int(arr.reshape(-1)[0])
    raise ValueError(f"cannot map observation of shape {arr.shape} to a state index; pass state_fn")


def _reset(env: Any, seed: int) -> Any:
    result: Any = env.reset(seed=seed)
    # Gymnasium returns (obs, info); tolerate an env that returns just obs.
    return result[0] if isinstance(result, tuple) else result


def _step(env: Any, action: int) -> tuple[Any, float, bool]:
    result: Any = env.step(action)
    if len(result) == 5:  # Gymnasium: (obs, reward, terminated, truncated, info)
        obs, reward, terminated, truncated = result[0], result[1], result[2], result[3]
        return obs, float(reward), bool(terminated or truncated)
    obs, reward, done = result[0], result[1], result[2]  # legacy: (obs, reward, done, info)
    return obs, float(reward), bool(done)


def from_causal_gym(
    env: Any,
    behavior_policy: BehaviorPolicy,
    *,
    n_episodes: int = 1000,
    max_steps: int = 100,
    seed: int = 0,
    n_states: int | None = None,
    n_actions: int | None = None,
    state_fn: StateFn | None = None,
) -> ConfoundedTrajectoryDataset:
    """Roll ``env`` out under ``behavior_policy`` into a ``ConfoundedTrajectoryDataset``.

    ``behavior_policy(observation) -> action`` chooses the discrete action; ``state_fn`` maps
    an observation to a discrete state index (defaults to an int observation or ``obs['state']``).
    ``n_states`` / ``n_actions`` are taken from the env's attributes when present, else from these
    arguments, else inferred from the logged data. Gymnasium
    ``reset(seed=...) -> (obs, info)`` and ``step(action) -> (obs, reward, terminated, truncated,
    info)`` are assumed (a legacy 4-tuple ``step`` is also tolerated).
    """
    to_state = state_fn or _default_state_index
    transitions: list[Transition] = []
    max_state = 0
    max_action = 0
    for episode in range(n_episodes):
        obs = _reset(env, seed + episode)
        for _ in range(max_steps):
            state = to_state(obs)
            action = int(behavior_policy(obs))
            next_obs, reward, done = _step(env, action)
            next_state = to_state(next_obs)
            transitions.append(Transition(state, action, reward, next_state, done))
            max_state = max(max_state, state, next_state)
            max_action = max(max_action, action)
            obs = next_obs
            if done:
                break
    resolved_states = n_states or int(getattr(env, "n_states", 0)) or max_state + 1
    resolved_actions = n_actions or int(getattr(env, "n_actions", 0)) or max_action + 1
    return ConfoundedTrajectoryDataset(
        transitions, n_states=resolved_states, n_actions=resolved_actions
    )
