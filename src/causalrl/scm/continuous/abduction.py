"""Posterior abduction for continuous mechanisms (plan §7.1); torch, behind the ``[torch]`` extra.

Two ways into the Phase-0 ``NoisePosterior`` protocol
(:class:`causalrl.protocols.NoisePosterior`):

* **Exact** (``abduct_location_scale``) — for an invertible location-scale mechanism the exogenous
  noise is recovered exactly by inversion, so a black-box abduction reproduces the exact-known
  counterfactual. Returns a point-mass ``PointNoisePosterior``; counterfactuals may claim
  ``kind=IDENTIFIED``.
* **Amortized VI** (``AmortizedGaussianAbduction``) — a Gaussian encoder ``q(u | parents, y)``
  trained by ELBO (standard-normal prior) for general, non-invertible mechanisms; returns an
  ``AmortizedNoisePosterior``. Such counterfactuals are ``kind=EMPIRICAL`` (I2).

``posterior_predictive_check`` reconstructs the outcome from a posterior noise draw; the result is
attached to the certificate as a checkable assumption.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor

from causalrl.certify.certificate import (
    Assumption,
    Certificate,
    EstimandSpec,
    Kind,
    Provenance,
)
from causalrl.scm.continuous.mechanisms import LocationScaleMechanism, build_mlp

__all__ = [
    "AmortizedGaussianAbduction",
    "AmortizedNoisePosterior",
    "PointNoisePosterior",
    "abduct_location_scale",
    "certify_counterfactual",
    "posterior_predictive_check",
]


def _fit_n(t: Tensor, n: int) -> Tensor:
    flat = t.reshape(-1)
    if int(flat.shape[0]) >= n:
        return flat[:n]
    reps = (n + int(flat.shape[0]) - 1) // int(flat.shape[0])
    return flat.repeat(reps)[:n]  # type: ignore[reportUnknownMemberType]


class PointNoisePosterior:
    """A ``NoisePosterior`` point mass at exactly recovered noise (invertible mechanisms)."""

    def __init__(self, noise: Mapping[str, Tensor]) -> None:
        self._noise = dict(noise)

    def sample(self, n: int, *, seed: int | None = None) -> Mapping[str, Any]:
        return {k: _fit_n(v, n) for k, v in self._noise.items()}


class AmortizedNoisePosterior:
    """A Gaussian posterior over one node's exogenous noise, produced by amortized VI."""

    def __init__(self, mean: Tensor, scale: Tensor, *, name: str = "U") -> None:
        self.mean = mean.detach()
        self.scale = scale.detach()
        self.name = name

    def sample(self, n: int, *, seed: int | None = None) -> Mapping[str, Any]:
        if seed is not None:
            torch.manual_seed(seed)  # type: ignore[reportPrivateImportUsage]
        eps = torch.randn((n, int(self.mean.shape[0])))  # type: ignore[reportPrivateImportUsage]
        return {self.name: self.mean + self.scale * eps}


def abduct_location_scale(
    mechanism: LocationScaleMechanism,
    parent_values: Mapping[str, Tensor],
    observed: Tensor,
    *,
    name: str = "U",
) -> PointNoisePosterior:
    """Exactly recover a location-scale mechanism's exogenous noise from an observed outcome."""
    return PointNoisePosterior({name: mechanism.invert(dict(parent_values), observed)})


class AmortizedGaussianAbduction(torch.nn.Module):
    """Amortized VI encoder ``q(u | parents, y) = N(mu, sigma)`` for a fixed mechanism.

    Trained by ELBO (Gaussian reparameterisation; standard-normal prior) to reconstruct the observed
    outcome through the mechanism. General — works for non-invertible mechanisms — so its
    counterfactuals are ``kind=EMPIRICAL``.
    """

    def __init__(
        self,
        mechanism: torch.nn.Module,
        parents: list[str],
        hidden: tuple[int, ...] = (32, 32),
        *,
        name: str = "U",
    ) -> None:
        super().__init__()
        self.mechanism = mechanism
        self.parents = parents
        self.name = name
        self.encoder = build_mlp(len(parents) + 1, hidden, 2)

    def _q(self, parent_values: Mapping[str, Tensor], y: Tensor) -> tuple[Tensor, Tensor]:
        cols = [parent_values[p].reshape(-1, 1) for p in self.parents] + [y.reshape(-1, 1)]
        h = self.encoder(torch.cat(cols, dim=1))  # type: ignore[reportPrivateImportUsage,reportUnknownMemberType]
        scale = torch.nn.functional.softplus(h[:, 1]) + 1e-4  # type: ignore[reportPrivateImportUsage]
        return h[:, 0], scale

    def fit(
        self,
        parent_values: Mapping[str, Tensor],
        y: Tensor,
        *,
        steps: int = 400,
        lr: float = 1e-2,
        beta: float = 1.0,
        seed: int = 0,
    ) -> AmortizedGaussianAbduction:
        torch.manual_seed(seed)  # type: ignore[reportPrivateImportUsage]
        opt = torch.optim.Adam(self.parameters(), lr=lr)  # type: ignore[reportPrivateImportUsage]
        pv = dict(parent_values)
        for _ in range(steps):
            mean, scale = self._q(pv, y)
            u = mean + scale * torch.randn_like(mean)  # type: ignore[reportPrivateImportUsage]
            recon = ((self.mechanism(pv, u) - y) ** 2).mean()
            kl = (0.5 * (mean**2 + scale**2 - 1.0) - torch.log(scale)).mean()  # type: ignore[reportPrivateImportUsage]
            loss = recon + beta * kl
            opt.zero_grad()
            loss.backward()
            opt.step()  # type: ignore[reportUnknownMemberType]
        return self

    def posterior(self, parent_values: Mapping[str, Tensor], y: Tensor) -> AmortizedNoisePosterior:
        with torch.no_grad():  # type: ignore[reportPrivateImportUsage]
            mean, scale = self._q(dict(parent_values), y)
        return AmortizedNoisePosterior(mean, scale, name=self.name)


def posterior_predictive_check(
    mechanism: torch.nn.Module,
    parent_values: Mapping[str, Tensor],
    y: Tensor,
    noise: Tensor,
) -> dict[str, float]:
    """Reconstruct ``y`` from ``noise`` through ``mechanism``; report the RMSE/bias PPC."""
    resid = mechanism(dict(parent_values), noise.reshape(-1)).reshape(-1) - y.reshape(-1)
    rmse = float(torch.sqrt((resid**2).mean()))  # type: ignore[reportPrivateImportUsage]
    return {"ppc_rmse": rmse, "ppc_bias": float(resid.mean())}


def certify_counterfactual(
    claim: str, ppc: dict[str, float], *, exact: bool, alpha: float | None = None
) -> Certificate:
    """Wrap a counterfactual query in a certificate.

    ``kind`` is IDENTIFIED for the exact-inversion path and EMPIRICAL for amortized VI (I2); the PPC
    is recorded as a checkable assumption.
    """
    method = "exact-inversion" if exact else "amortized-vi"
    return Certificate(
        claim=claim,
        estimand=EstimandSpec(query="counterfactual", target="mean"),
        kind=Kind.IDENTIFIED if exact else Kind.EMPIRICAL,
        value=None,
        alpha=alpha,
        assumptions=(
            Assumption(name="abduction", params={"method": method}, checkable=True, diagnostic=ppc),
        ),
        method=method,
        witness=None,
        hedge=None,
        provenance=Provenance.create(),
    )
