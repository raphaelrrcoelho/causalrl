# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownMemberType=false
"""PettingZoo ParallelEnv → :class:`~causalrl.data.trajectory.TrajectoryLog` adapter (plan §8.3).

Rolls out a PettingZoo ``ParallelEnv`` (or any object with its ``reset`` / ``step`` / ``agents``
surface) under supplied per-agent policies and logs every agent-step into a columnar
``TrajectoryLog`` with ``entity_id`` = the agent (its index in ``possible_agents``). Fully
duck-typed — PettingZoo is *never imported*, matching the DoWhy/EconML interop house style — so any
ParallelEnv-shaped object (a real benchmark or a lightweight stand-in) drives it unchanged. Install
PettingZoo yourself to run it against a real benchmark.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from causalrl.data.trajectory import TrajectoryLog

__all__ = ["pettingzoo_to_trajectory_log"]

# policy(observation) -> action
Policy = Callable[[Any], Any]


def _reset(env: Any, seed: int) -> dict[str, Any]:
    result: Any = env.reset(seed=seed)
    # New PettingZoo API returns (observations, infos); older returns just observations.
    obs: Any = result[0] if isinstance(result, tuple) else result
    return dict(obs)


def _step(env: Any, actions: Mapping[str, Any]) -> tuple[Any, Any, Any, Any]:
    result: Any = env.step(dict(actions))
    obs, rewards, terms, truncs = result[0], result[1], result[2], result[3]
    return obs, rewards, terms, truncs


def _obs_columns(obs: Any) -> dict[str, float]:
    """Flatten an agent's observation to named float columns (``obs`` or ``obs_0``, ``obs_1`` …)."""
    if obs is None:
        return {}
    arr = np.asarray(obs, dtype=np.float64).reshape(-1)
    if arr.size == 1:
        return {"obs": float(arr[0])}
    return {f"obs_{j}": float(v) for j, v in enumerate(arr.tolist())}


def pettingzoo_to_trajectory_log(
    env: Any,
    policies: Mapping[str, Policy],
    *,
    n_episodes: int = 1,
    max_steps: int = 100,
    seed: int = 0,
) -> TrajectoryLog:
    """Roll ``env`` out under ``policies`` and log every agent-step to a ``TrajectoryLog``.

    ``policies[agent](observation) -> action`` drives each active agent. Each agent-step logs the
    observation columns, the chosen ``action``, and the ``reward``, keyed by ``entity_id`` (the
    agent's index), ``episode_id``, and ``t``. Raises ``KeyError`` if an active agent has no policy.
    """
    possible = list(getattr(env, "possible_agents", []))
    agent_index = {a: i for i, a in enumerate(possible)}
    rows: list[dict[str, Any]] = []
    for episode in range(n_episodes):
        obs = dict(_reset(env, seed + episode))
        for t in range(max_steps):
            active = list(getattr(env, "agents", []))
            if not active:
                break
            actions = {a: policies[a](obs.get(a)) for a in active}
            next_obs, rewards, terms, truncs = _step(env, actions)
            for a in active:
                ei = agent_index.get(a, len(agent_index))
                agent_index.setdefault(a, ei)
                cols = {**_obs_columns(obs.get(a)), "action": float(actions[a])}
                cols["reward"] = float(dict(rewards).get(a, 0.0))
                kinds = {"action": "action", "reward": "reward"}
                for name, value in cols.items():
                    rows.append(
                        {
                            "entity_id": ei,
                            "episode_id": episode,
                            "t": t,
                            "kind": kinds.get(name, "obs"),
                            "name": name,
                            "value": value,
                            "regime": "observed",
                            "observed": True,
                        }
                    )
            obs = dict(next_obs)
            terms_d, truncs_d = dict(terms), dict(truncs)
            if all(terms_d.get(a, False) or truncs_d.get(a, False) for a in active):
                break
    return TrajectoryLog.from_rows(rows)
