"""Continuous / neural structural mechanisms (plan §7.1); torch, behind the ``[torch]`` extra.

Additive-noise MLP mechanisms, invertible location-scale mechanisms, and conditional
normalizing-flow mechanisms, all conforming to the shipped
:class:`~causalrl.scm.mechanisms.Mechanism` protocol. The location-scale and flow families are
invertible in their scalar exogenous noise, which licenses EXACT abduction and hence exact
continuous counterfactuals — see :mod:`causalrl.scm.continuous.abduction`.
"""

from __future__ import annotations

from itertools import pairwise

import torch
from torch import Tensor

__all__ = ["ConditionalFlowMechanism", "LocationScaleMechanism", "MLPMechanism"]


def build_mlp(in_dim: int, hidden: tuple[int, ...], out_dim: int) -> torch.nn.Module:
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
        self.net = build_mlp(max(len(parents), 1), hidden, 1)

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
        self.loc_net = build_mlp(max(len(parents), 1), hidden, 1)
        self.scale_net = build_mlp(max(len(parents), 1), hidden, 1)

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


def _leaky_relu_inverse(y: Tensor, negative_slope: float) -> Tensor:
    """Closed-form inverse of ``LeakyReLU``: ``y`` for ``y >= 0`` else ``y / negative_slope``.

    ``LeakyReLU`` preserves sign for a positive slope, so the branch on the *output* sign inverts it
    exactly.
    """
    return torch.where(y >= 0, y, y / negative_slope)  # type: ignore[reportPrivateImportUsage]


class ConditionalFlowMechanism(torch.nn.Module):
    """Conditional normalizing-flow mechanism, invertible in its scalar exogenous noise.

    ``V_i = flow(noise; parents)`` is a composition of ``n_blocks`` conditional affine maps
    ``u -> loc_k(parents) + scale_k(parents) * u`` (with ``scale_k = softplus(.) + min_scale > 0``)
    interleaved with a fixed invertible ``LeakyReLU`` (``negative_slope`` in ``(0, 1)``), ending on
    an affine map. Each factor is strictly increasing in the scalar ``u``, so the whole map is
    strictly monotone and the noise is recovered in closed form (:meth:`invert`) — exact abduction
    and exact continuous counterfactuals are therefore available, as for
    :class:`LocationScaleMechanism` but with a strictly more expressive, non-affine transform.

    References: Rezende & Mohamed, *Variational Inference with Normalizing Flows* (2015);
    Papamakarios et al., *Normalizing Flows for Probabilistic Modeling and Inference* (JMLR 2021).
    Formula-level implementation; no third-party code is ported.
    """

    def __init__(
        self,
        parents: list[str],
        hidden: tuple[int, ...] = (32,),
        *,
        n_blocks: int = 2,
        negative_slope: float = 0.1,
        min_scale: float = 1e-3,
    ) -> None:
        super().__init__()
        if n_blocks < 1:
            raise ValueError("n_blocks must be >= 1")
        if not 0.0 < negative_slope < 1.0:
            raise ValueError("negative_slope must lie in (0, 1) to stay invertible")
        self.parents = parents
        self.negative_slope = float(negative_slope)
        self.min_scale = float(min_scale)
        in_dim = max(len(parents), 1)
        self.locs = torch.nn.ModuleList(  # type: ignore[reportPrivateImportUsage]
            [build_mlp(in_dim, hidden, 1) for _ in range(n_blocks)]
        )
        self.log_scales = torch.nn.ModuleList(  # type: ignore[reportPrivateImportUsage]
            [build_mlp(in_dim, hidden, 1) for _ in range(n_blocks)]
        )

    def _affine(self, k: int, feats: Tensor) -> tuple[Tensor, Tensor]:
        loc = self.locs[k](feats).squeeze(-1)  # type: ignore[reportUnknownMemberType]
        raw = self.log_scales[k](feats).squeeze(-1)  # type: ignore[reportUnknownMemberType]
        scale = torch.nn.functional.softplus(raw) + self.min_scale  # type: ignore[reportPrivateImportUsage]
        return loc, scale

    def forward(self, parent_values: dict[str, Tensor], noise: Tensor) -> Tensor:
        feats = _stack_parents(self.parents, parent_values, noise)
        n = len(self.locs)
        u = noise
        for k in range(n):
            loc, scale = self._affine(k, feats)
            u = loc + scale * u
            if k < n - 1:
                u = torch.nn.functional.leaky_relu(u, self.negative_slope)  # type: ignore[reportPrivateImportUsage]
        return u

    def invert(self, parent_values: dict[str, Tensor], observed: Tensor) -> Tensor:
        """Recover the exogenous noise exactly by undoing each factor in reverse order."""
        feats = _stack_parents(self.parents, parent_values, observed)
        n = len(self.locs)
        v = observed
        for k in reversed(range(n)):
            if k < n - 1:
                v = _leaky_relu_inverse(v, self.negative_slope)
            loc, scale = self._affine(k, feats)
            v = (v - loc) / scale
        return v
