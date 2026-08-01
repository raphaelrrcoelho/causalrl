"""Mechanism fitters: (parent columns, child column) -> a fitted structural equation.

Each fitter returns the mechanism, the exogenous distribution its noise is drawn from, whether
that noise is recoverable from (parents, value) — which decides whether counterfactuals at this
node are identified — and a held-out-comparable fit score.
"""

from __future__ import annotations

from typing import NamedTuple, Protocol

import numpy as np
import torch
from torch.distributions import Distribution, Uniform

from causalrl.scm.mechanisms import FunctionalMechanism, Mechanism

Tensor = torch.Tensor


class FittedMechanism(NamedTuple):
    """A fitted structural equation plus what the fit does and does not license."""

    mechanism: Mechanism
    noise: Distribution
    invertible: bool
    score: float


class MechanismFitter(Protocol):
    """Fits ``V = f(parents, noise)`` for one node."""

    def fit(self, parents: dict[str, np.ndarray], child: np.ndarray) -> FittedMechanism: ...


class TabularCPT:
    """Discrete node: a Laplace-smoothed conditional probability table, sampled by inverse CDF.

    The mechanism is ``V = F^-1(U | parents)`` with ``U ~ Uniform(0, 1)``. That construction is
    one of many couplings reproducing the same ``P(V | parents)``, and the data cannot
    distinguish them — hence ``invertible=False``, which makes counterfactuals at this node an
    interval rather than a point.
    """

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha

    def fit(self, parents: dict[str, np.ndarray], child: np.ndarray) -> FittedMechanism:
        values = np.unique(child)
        parent_names = sorted(parents)
        levels = {name: np.unique(parents[name]) for name in parent_names}
        strides: list[int] = []
        size = 1
        for name in parent_names:
            strides.append(size)
            size *= len(levels[name])

        rows = (
            np.zeros(len(child), dtype=int)
            if not parent_names
            else self._config_index(parents, parent_names, levels, strides, size)
        )
        counts = np.full((size, len(values)), self.alpha, dtype=float)
        col_of = {v: j for j, v in enumerate(values)}
        for row, value in zip(rows, child, strict=True):
            counts[row, col_of[value]] += 1.0
        table = counts / counts.sum(axis=1, keepdims=True)

        # Mean conditional log-likelihood of the training child values under the fitted table.
        columns: list[int] = [col_of[v] for v in child]
        score = float(np.log(table[rows, columns]).mean())

        cum = torch.tensor(np.cumsum(table, axis=1), dtype=torch.float32)  # type: ignore[reportPrivateImportUsage]
        value_tensor = torch.tensor(values, dtype=torch.float32)  # type: ignore[reportPrivateImportUsage]
        level_tensors = {
            name: torch.tensor(levels[name], dtype=torch.float32)  # type: ignore[reportPrivateImportUsage]
            for name in parent_names
        }

        def mechanism(parent_values: dict[str, Tensor], noise: Tensor) -> Tensor:
            n = noise.reshape(-1).shape[0]
            row = torch.zeros(n, dtype=torch.long)  # type: ignore[reportPrivateImportUsage]
            for name, stride in zip(parent_names, strides, strict=True):
                column = parent_values[name].reshape(-1).float()
                # Nearest level, so an unseen/off-grid parent value maps to its closest bucket.
                distance = (column.unsqueeze(1) - level_tensors[name].unsqueeze(0)).abs()
                row = row + distance.argmin(dim=1) * stride
            picked = (noise.reshape(-1).unsqueeze(1) > cum[row]).sum(dim=1)
            return value_tensor[picked.clamp(max=len(values) - 1)]

        return FittedMechanism(
            mechanism=FunctionalMechanism(parent_names, mechanism),
            noise=Uniform(0.0, 1.0),
            invertible=False,
            score=score,
        )

    @staticmethod
    def _config_index(
        parents: dict[str, np.ndarray],
        parent_names: list[str],
        levels: dict[str, np.ndarray],
        strides: list[int],
        size: int,
    ) -> np.ndarray:
        n = len(next(iter(parents.values()))) if parents else 0
        if not parent_names:
            return np.zeros(max(n, 1) if parents else 0, dtype=int)
        rows = np.zeros(n, dtype=int)
        for name, stride in zip(parent_names, strides, strict=True):
            rows += np.searchsorted(levels[name], parents[name]) * stride
        return rows
