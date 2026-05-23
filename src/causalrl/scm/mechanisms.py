from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, runtime_checkable

import torch
from torch import Tensor  # pyright treats Tensor as exported


@runtime_checkable
class Mechanism(Protocol):
    """A structural equation V_i = f_i(parents, noise)."""

    parents: list[str]

    def __call__(self, parent_values: dict[str, Tensor], noise: Tensor) -> Tensor: ...


class FunctionalMechanism:
    """Wrap an arbitrary callable f(parent_values, noise) -> Tensor."""

    def __init__(
        self, parents: list[str], fn: Callable[[dict[str, Tensor], Tensor], Tensor]
    ) -> None:
        self.parents = parents
        self._fn = fn

    def __call__(self, parent_values: dict[str, Tensor], noise: Tensor) -> Tensor:
        return self._fn(parent_values, noise)


class LinearGaussianMechanism:
    """V_i = sum_j w_j * parent_j + bias + noise."""

    def __init__(self, parents: list[str], weights: dict[str, float], bias: float = 0.0) -> None:
        self.parents = parents
        self._weights = weights
        self._bias = bias

    def __call__(self, parent_values: dict[str, Tensor], noise: Tensor) -> Tensor:
        # torch.tensor / torch.cat are public API; pyright strict flags them as not
        # re-exported by torch stubs — suppress only these two occurrences.
        out: Tensor = torch.tensor(self._bias, dtype=noise.dtype) + noise  # type: ignore[reportPrivateImportUsage]
        for p in self.parents:
            out = out + self._weights[p] * parent_values[p]
        return out


class NeuralMechanism(torch.nn.Module):
    """V_i = net([parents, noise]). Makes the SCM a neural causal model (NCM)."""

    def __init__(self, parents: list[str], net: torch.nn.Module) -> None:
        super().__init__()
        self.parents = parents
        self.net = net

    def forward(self, parent_values: dict[str, Tensor], noise: Tensor) -> Tensor:
        cols = [parent_values[p].reshape(-1, 1) for p in self.parents] + [noise.reshape(-1, 1)]
        return self.net(torch.cat(cols, dim=1)).squeeze(-1)  # type: ignore[reportPrivateImportUsage]
