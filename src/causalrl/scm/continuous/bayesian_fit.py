# pyright: reportMissingImports=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
"""Posterior mechanism fitting: a Bayesian linear structural equation via NUTS.

Where :class:`~causalrl.scm.fitters.LinearGaussianFit` returns one point-estimated mechanism, this
returns the posterior-mean mechanism *plus* the draws behind it, so a caller can carry mechanism
uncertainty into the estimand rather than discarding it. That makes the SCM belief continuous,
where :func:`~causalrl.scm.fit.fit_scm_mec` makes it discrete.

Implements standard Bayesian linear regression with weakly-informative priors, sampled by NUTS
(Hoffman & Gelman, *The No-U-Turn Sampler*, JMLR 2014) via NumPyro; no external code is ported.

See also `pathmc <https://github.com/pymc-labs/pathmc>`_ (PyMC Labs, Apache-2.0), which builds
Bayesian structural causal models around a lavaan-style formula DSL. It is a different set of
trade-offs — a PyMC backend and Python >= 3.12, against this library's NumPyro extra — and reading
it helped confirm that posterior structural coefficients were worth having here.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from torch.distributions import Normal

from causalrl.scm.fitters import (
    FittedMechanism,
    _attach_residual,  # type: ignore[reportPrivateUsage]
    _r2,  # type: ignore[reportPrivateUsage]
)
from causalrl.scm.mechanisms import LinearGaussianMechanism

__all__ = ["BayesianLinearFit"]


def _require_numpyro() -> Any:
    try:
        import numpyro
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "BayesianLinearFit requires NumPyro (and JAX); install the 'causalrl[numpyro]' extra"
        ) from exc
    return numpyro


class BayesianLinearFit:
    """Continuous node: Bayesian linear structural equation, posterior sampled by NUTS (NumPyro).

    Where :class:`~causalrl.scm.fitters.LinearGaussianFit` solves a single OLS point estimate,
    this samples the full posterior over ``(intercept, weights, sigma)`` and returns the
    posterior-**mean** as a :class:`~causalrl.scm.mechanisms.LinearGaussianMechanism` -- so every
    invariant the point-estimated continuous families satisfy still holds here: ``invertible=True``,
    a ``residual`` closure is attached, the round-trip identity holds exactly, and
    ``evaluate_holdout`` scores it by R^2 like its siblings. The draws behind that mean are attached
    as ``mechanism.posterior``, a ``dict[str, np.ndarray]`` keyed ``"intercept"``, each parent name,
    and ``"sigma"`` -- plain NumPy, not JAX, so a caller without JAX in scope can use them directly.
    Materialising one SCM per draw is a natural follow-on; this fitter does not build it.

    Model: ``intercept ~ Normal(0, 10)``, ``w_j ~ Normal(0, 10)`` per parent, ``sigma ~
    HalfNormal(1)``, ``y ~ Normal(intercept + X @ w, sigma)``. ``X`` is standardised internally so
    the weakly-informative priors stay sensible regardless of the parents' scale; the posterior
    draws are unstandardised back onto the original data scale before being exposed.
    """

    def __init__(self, draws: int = 1000, warmup: int = 1000, seed: int = 0) -> None:
        self.draws = draws
        self.warmup = warmup
        self.seed = seed

    def fit(self, parents: dict[str, np.ndarray], child: np.ndarray) -> FittedMechanism:
        _require_numpyro()
        import jax
        import jax.numpy as jnp
        import numpyro
        import numpyro.distributions as dist
        from numpyro.infer import MCMC, NUTS

        y = np.asarray(child, dtype=float)
        n = len(y)
        names = sorted(parents)
        n_parents = len(names)
        raw = (
            np.column_stack([np.asarray(parents[name], dtype=float) for name in names])
            if names
            else np.zeros((n, 0))
        )
        # Standardise so N(0, 10) stays weakly-informative regardless of the parents' native
        # scale; a constant column (std 0) is left unscaled rather than divided by zero.
        mean_x = raw.mean(axis=0)
        scale_x = raw.std(axis=0)
        safe_scale_x = np.where(scale_x > 1e-9, scale_x, 1.0)
        x_std = (raw - mean_x) / safe_scale_x

        x_jax = jnp.asarray(x_std)
        y_jax = jnp.asarray(y)

        def model(x: Any, y_obs: Any) -> None:
            intercept = numpyro.sample("intercept", dist.Normal(0.0, 10.0))
            if n_parents > 0:
                w = numpyro.sample("w", dist.Normal(0.0, 10.0).expand([n_parents]).to_event(1))
                mean = intercept + x @ w
            else:
                mean = intercept + jnp.zeros(y_obs.shape[0])
            sigma = numpyro.sample("sigma", dist.HalfNormal(1.0))
            with numpyro.plate("data", y_obs.shape[0]):
                numpyro.sample("y", dist.Normal(mean, sigma), obs=y_obs)

        mcmc = MCMC(NUTS(model), num_warmup=self.warmup, num_samples=self.draws, progress_bar=False)
        mcmc.run(jax.random.PRNGKey(self.seed), x_jax, y_jax)
        samples = mcmc.get_samples()

        intercept_std = np.asarray(samples["intercept"], dtype=float)
        sigma_draws = np.asarray(samples["sigma"], dtype=float)
        if n_parents > 0:
            w_std = np.asarray(samples["w"], dtype=float)  # (num_draws, n_parents)
            weight_draws = {name: w_std[:, i] / safe_scale_x[i] for i, name in enumerate(names)}
            intercept_draws = intercept_std - sum(
                weight_draws[name] * mean_x[i] for i, name in enumerate(names)
            )
        else:
            weight_draws = {}
            intercept_draws = intercept_std

        posterior: dict[str, np.ndarray] = {
            "intercept": intercept_draws,
            **weight_draws,
            "sigma": sigma_draws,
        }

        intercept_mean = float(intercept_draws.mean())
        weight_means = {name: float(weight_draws[name].mean()) for name in names}
        sigma_mean = float(sigma_draws.mean())
        coefficients = np.concatenate(([intercept_mean], [weight_means[name] for name in names]))

        def mean_fn(columns: np.ndarray) -> np.ndarray:
            return np.column_stack([np.ones(len(columns)), columns]) @ coefficients

        mechanism = LinearGaussianMechanism(names, weight_means, bias=intercept_mean)
        _attach_residual(mechanism, names, mean_fn)
        mechanism.posterior = posterior  # type: ignore[attr-defined]
        mechanism.coefficients = coefficients  # type: ignore[attr-defined]

        predicted = mean_fn(raw)
        return FittedMechanism(
            mechanism=mechanism,
            noise=Normal(0.0, max(sigma_mean, 1e-6)),
            invertible=True,
            score=_r2(y, predicted),
        )
