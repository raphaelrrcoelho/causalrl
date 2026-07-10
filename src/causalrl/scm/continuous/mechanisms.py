"""Continuous / neural structural mechanisms (plan §7.1); torch, behind the ``[torch]`` extra.

Additive-noise MLP mechanisms and invertible location-scale mechanisms conforming to the shipped
:class:`~causalrl.scm.mechanisms.Mechanism` protocol. The location-scale family is invertible in its
exogenous noise (``noise = (V_i - loc) / scale``), which licenses EXACT abduction and hence exact
continuous counterfactuals — see :mod:`causalrl.scm.continuous.abduction`.
"""

from __future__ import annotations

from itertools import pairwise

import torch
from torch import Tensor

__all__ = ["LocationScaleMechanism", "MLPMechanism"]


def _mlp(in_dim: int, hidden: tuple[int, ...], out_dim: int) -> torch.nn.Module:
    sizes = [in_dim, *hidden, out_dim]
    layers: list[torch.nn.Module] = []
    for a, b in pairwise(sizes):
        layers.append(torch.nn.Linear(a, b))  # type: ignore[reportPrivateImportUsage]
        layers.append(torch.nn.Tanh())  # type: ignore[reportPrivateImportUsage]
    return torch.nn.Sequential(*layers[:-1])  # type: ignore[reportUnknownMemberType]


def _stack_parents(parents: list[str], parent_values: dict[str, Tensor], ref: Tensor) -> Tensor:
    """Column-stack the parent tensors; a zero column for a root (parent-free) mechanism."""
    if not parents:
        return torch.zeros((ref.reshape(-1).shape[0], 1))  # type: ignore[reportPrivateImportUsage]
    cols = [parent_values[p].reshape(-1, 1) for p in parents]
    return torch.cat(cols, dim=1)  # type: ignore[reportPrivateImportUsage]


class MLPMechanism(torch.nn.Module):
    """Additive-noise neural mechanism ``V_i = MLP(parents) + noise``."""

    def __init__(self, parents: list[str], hidden: tuple[int, ...] = (32, 32)) -> None:
        super().__init__()
        self.parents = parents
        self.net = _mlp(max(len(parents), 1), hidden, 1)

    def forward(self, parent_values: dict[str, Tensor], noise: Tensor) -> Tensor:
        feats = _stack_parents(self.parents, parent_values, noise)
        return self.net(feats).squeeze(-1) + noise  # type: ignore[reportUnknownMemberType]


class LocationScaleMechanism(torch.nn.Module):
    """Invertible location-scale mechanism ``V_i = loc(parents) + scale(parents) * noise``.

    ``scale = softplus(raw_scale(parents)) + min_scale > 0``, so the map is invertible in ``noise``:
    ``noise = (V_i - loc) / scale``. Exact abduction is therefore available via
    :func:`causalrl.scm.continuous.abduction.abduct_location_scale`.
    """

    def __init__(
        self, parents: list[str], hidden: tuple[int, ...] = (32,), min_scale: float = 1e-3
    ) -> None:
        super().__init__()
        self.parents = parents
        self.min_scale = float(min_scale)
        self.loc_net = _mlp(max(len(parents), 1), hidden, 1)
        self.scale_net = _mlp(max(len(parents), 1), hidden, 1)

    def _loc_scale(self, feats: Tensor) -> tuple[Tensor, Tensor]:
        loc = self.loc_net(feats).squeeze(-1)  # type: ignore[reportUnknownMemberType]
        raw = self.scale_net(feats).squeeze(-1)  # type: ignore[reportUnknownMemberType]
        scale = torch.nn.functional.softplus(raw) + self.min_scale  # type: ignore[reportPrivateImportUsage]
        return loc, scale

    def forward(self, parent_values: dict[str, Tensor], noise: Tensor) -> Tensor:
        loc, scale = self._loc_scale(_stack_parents(self.parents, parent_values, noise))
        return loc + scale * noise

    def invert(self, parent_values: dict[str, Tensor], observed: Tensor) -> Tensor:
        """Recover the exogenous noise exactly: ``(observed - loc) / scale``."""
        loc, scale = self._loc_scale(_stack_parents(self.parents, parent_values, observed))
        return (observed - loc) / scale
