from typing import Any

import numpy as np
import torch

from causalrl.agents.base import Agent
from causalrl.agents.primitives import bounds_table
from causalrl.data.dataset import ConfoundedTrajectoryDataset


class _QNet(torch.nn.Module):
    """One-hot state -> Q-values per action."""

    def __init__(self, n_states: int, n_actions: int) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(  # type: ignore[reportUnknownMemberType]
            torch.nn.Linear(n_states, 64),  # type: ignore[reportPrivateImportUsage]
            torch.nn.ReLU(),  # type: ignore[reportPrivateImportUsage]
            torch.nn.Linear(64, n_actions),  # type: ignore[reportPrivateImportUsage]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # type: ignore[reportUnknownMemberType,no-any-return]


class DeepDeconfoundedQ(Agent):
    """DQN-style agent whose bootstrap targets are clamped into the Manski causal bounds.

    Function approximation can extrapolate Q outside the causally-valid envelope; clamping
    each (state, action) target into [lower, upper] keeps the learned values consistent with
    what the confounded offline data can support, while online experience tightens them.
    A small one-step bootstrap is used (sufficient for the toy demo); a full offline-RL
    backbone (d3rlpy) is the designated reuse path at real scale.
    """

    def __init__(self, n_states: int, n_actions: int, seed: int | None = None) -> None:
        self.n_states = n_states
        self.n_actions = n_actions
        torch.manual_seed(0 if seed is None else seed)  # type: ignore[reportPrivateImportUsage]
        self._rng = np.random.default_rng(seed)
        self._q = _QNet(n_states, n_actions)
        self._opt = torch.optim.Adam(self._q.parameters(), lr=1e-2)  # type: ignore[reportPrivateImportUsage]
        self._eps = 0.2
        self._lower = np.zeros((n_states, n_actions))
        self._upper = np.ones((n_states, n_actions))

    def ingest_offline(self, dataset: ConfoundedTrajectoryDataset) -> None:
        for (s, a), (lo, hi) in bounds_table(dataset).items():
            self._lower[s, a] = lo
            self._upper[s, a] = hi

    def bound(self, state: int, action: int) -> tuple[float, float]:
        return float(self._lower[state, action]), float(self._upper[state, action])

    def clamp_target(self, state: int, action: int, target: float) -> float:
        lo, hi = self.bound(state, action)
        return float(min(max(target, lo), hi))

    def _onehot(self, state: int) -> torch.Tensor:
        x = torch.zeros(self.n_states)  # type: ignore[reportPrivateImportUsage]
        x[state] = 1.0
        return x

    def act(self, observation: dict[str, Any]) -> int:
        s = int(observation["state"])
        if self._rng.random() < self._eps:
            return int(self._rng.integers(0, self.n_actions))
        with torch.no_grad():  # type: ignore[reportPrivateImportUsage]
            q = self._q(self._onehot(s))
        return int(torch.argmax(q).item())  # type: ignore[reportPrivateImportUsage]

    def update(self, observation: dict[str, Any], action: int, reward: float) -> None:
        s = int(observation["state"])
        target = self.clamp_target(s, action, float(reward))
        q = self._q(self._onehot(s))
        pred = q[action]
        loss = (pred - torch.tensor(target)) ** 2  # type: ignore[reportPrivateImportUsage]
        self._opt.zero_grad()
        loss.backward()
        self._opt.step()  # type: ignore[reportUnknownMemberType]
