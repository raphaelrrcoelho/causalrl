# pyright: reportMissingImports=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
"""NUTS / NumPyro posterior abduction (plan §7.1, deferred Phase-1 item).

The slow, sampling-based counterpart to the amortized-VI encoder in
:mod:`causalrl.scm.continuous.abduction`: a No-U-Turn Sampler (Hoffman & Gelman, 2014) draws the
exact posterior over a mechanism's exogenous noise given an observed outcome. It is expressible
**wherever the mechanism forward map is written in JAX** (``jax.numpy``), which is the regime
NumPyro requires — so the forward is passed as a JAX callable ``forward(parents, u) -> mean``,
decoupled from the torch mechanism library. Counterfactuals built on it are ``kind=EMPIRICAL``
(sampling evidence), as for VI (I2).

Optional backend, behind the ``[numpyro]`` extra: NumPyro (and its JAX dependency) is imported
lazily inside :func:`abduct_nuts` — never at import time — so this module loads without it, and a
clear ``ImportError`` names the extra when the sampler is invoked. The returned
:class:`NUTSNoisePosterior` is pure NumPy, so nothing downstream needs JAX. Formula-level model; no
third-party code is ported.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
from numpy.typing import NDArray

from causalrl.certify.certificate import (
    Assumption,
    Certificate,
    EstimandSpec,
    Kind,
    Provenance,
)

__all__ = [
    "NUTSNoisePosterior",
    "abduct_nuts",
    "certify_nuts_counterfactual",
    "nuts_posterior_predictive_check",
]

FloatArray = NDArray[np.float64]
# A JAX-expressible forward map ``forward(parent_values, u) -> mean`` (elementwise over units).
JaxForward = Callable[[Mapping[str, Any], Any], Any]


def _require_numpyro() -> Any:
    try:
        import numpyro
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "NUTS abduction requires NumPyro (and JAX); install the 'causalrl[numpyro]' extra"
        ) from exc
    return numpyro


class NUTSNoisePosterior:
    """A ``NoisePosterior`` backed by NUTS draws of a mechanism's per-unit exogenous noise.

    ``mean`` / ``std`` summarise the posterior per unit; ``sample(n)`` returns ``n`` posterior draws
    (each a full per-unit noise vector), tiling if more are requested than were drawn. Pure NumPy —
    it holds no JAX state, so it composes with the torch/NumPy stack unchanged.
    """

    def __init__(self, samples: FloatArray, *, name: str = "U") -> None:
        self._samples = np.asarray(samples, dtype=np.float64)  # (num_samples, n_units)
        self.name = name
        self.mean: FloatArray = self._samples.mean(axis=0)
        self.std: FloatArray = self._samples.std(axis=0)

    def sample(self, n: int, *, seed: int | None = None) -> Mapping[str, Any]:
        s = self._samples
        if n <= s.shape[0]:
            return {self.name: s[:n]}
        reps = (n + s.shape[0] - 1) // s.shape[0]
        return {self.name: np.tile(s, (reps, 1))[:n]}


def abduct_nuts(
    forward: JaxForward,
    parent_values: Mapping[str, Any],
    observed: Any,
    *,
    noise_scale: float = 0.1,
    num_warmup: int = 500,
    num_samples: int = 1000,
    seed: int = 0,
    name: str = "U",
) -> NUTSNoisePosterior:
    """Draw the posterior over a mechanism's exogenous noise by NUTS (NumPyro).

    Model: a standard-normal prior ``u_i ~ N(0, 1)`` per unit and a Gaussian likelihood
    ``observed_i ~ N(forward(parents, u)_i, noise_scale)``. ``forward`` must be written in
    ``jax.numpy`` (elementwise over units). Returns a pure-NumPy :class:`NUTSNoisePosterior`.
    Requires the ``[numpyro]`` extra.
    """
    _require_numpyro()
    import jax
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS

    y = jnp.asarray(np.asarray(observed, dtype=np.float64))
    n_units = int(y.shape[0])
    pv = {k: jnp.asarray(np.asarray(v, dtype=np.float64)) for k, v in parent_values.items()}

    def model() -> None:
        with numpyro.plate("units", n_units):
            u = numpyro.sample("u", dist.Normal(0.0, 1.0))
            mean = forward(pv, u)
            numpyro.sample("obs", dist.Normal(mean, noise_scale), obs=y)

    mcmc = MCMC(NUTS(model), num_warmup=num_warmup, num_samples=num_samples, progress_bar=False)
    mcmc.run(jax.random.PRNGKey(seed))
    draws = np.asarray(mcmc.get_samples()["u"], dtype=np.float64)  # (num_samples, n_units)
    return NUTSNoisePosterior(draws, name=name)


def nuts_posterior_predictive_check(
    forward: JaxForward, parent_values: Mapping[str, Any], observed: Any, noise: Any
) -> dict[str, float]:
    """Reconstruct ``observed`` from a posterior noise summary through ``forward`` (RMSE / bias)."""
    pv = {k: np.asarray(v, dtype=np.float64) for k, v in parent_values.items()}
    recon = np.asarray(forward(pv, np.asarray(noise, dtype=np.float64)), dtype=np.float64)
    resid = recon.reshape(-1) - np.asarray(observed, dtype=np.float64).reshape(-1)
    return {"ppc_rmse": float(np.sqrt((resid**2).mean())), "ppc_bias": float(resid.mean())}


def certify_nuts_counterfactual(
    claim: str, ppc: dict[str, float], *, alpha: float | None = None
) -> Certificate:
    """Wrap a NUTS-abducted counterfactual in an ``EMPIRICAL`` certificate (sampling evidence; I2).

    The posterior-predictive check is recorded as a checkable assumption.
    """
    return Certificate(
        claim=claim,
        estimand=EstimandSpec(query="counterfactual", target="mean"),
        kind=Kind.EMPIRICAL,
        value=None,
        alpha=alpha,
        assumptions=(
            Assumption(name="abduction", params={"method": "nuts"}, checkable=True, diagnostic=ppc),
        ),
        method="nuts",
        witness=None,
        hedge=None,
        provenance=Provenance.create(),
    )
