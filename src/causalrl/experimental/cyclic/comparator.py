"""Equilibrium-vs-unrolling comparator for cyclic SCMs (experimental; plan §11).

A cyclic SCM has two natural interventional semantics: the **equilibrium** ``do`` (replace the
mechanism, re-solve the fixed point) and the **unrolled** ``do`` (run the dynamics
``x_{k+1} = B x_k + u`` forward under the intervention). :func:`compare_equilibrium_unrolling` asks
whether the long-run unrolled ``do`` converges to the equilibrium ``do`` and returns a
:class:`~causalrl.certify.certificate.Certificate`:

* **IDENTIFIED** when the (intervened) iteration is stable -- for the naive unrolling that is
  contractivity (``rho(B) < 1``); for ``learning_rate=gamma`` it is stability of the damped map
  ``x + gamma (B x + E[u] - x)``, which for small ``gamma`` holds exactly when the mean dynamics
  ``x' = (B - I) x + u`` are stable (a positive :meth:`LinearCyclicSCM.stability_margin`) -- and
  the measured gap confirms convergence;
* **EMPIRICAL** (hedged) when the iteration is unstable (no convergence guarantee) or has not yet
  converged at the chosen horizon; the hedge carries the stability diagnostics
  (``stability_margin``, ``max_stable_learning_rate``), so a non-contractive-but-stable system is
  reported with the learning rate that *would* certify it;
* **EMPIRICAL** with the solvability hedge when there is no unique equilibrium to compare against.

The unrolled side is computed directly as the linear mean dynamics ``x_{k+1} = B x_k + E[u]`` (or
its damped form) from ``x_0 = 0``. (The shipped :func:`~causalrl.scm.unrolled.build_unrolled_scm` is scalar-per-node --
its ``StructuralCausalModel`` reshapes every node to one scalar per unit -- so it cannot represent a
multi-variable vector state in a single unrolled chain; the direct dynamics are the same object.)
Everything here is pure NumPy.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from causalrl.certify.certificate import (
    Assumption,
    Certificate,
    EstimandSpec,
    Hedge,
    Kind,
    Provenance,
    Witness,
)
from causalrl.experimental.cyclic.scm import FloatArray, LinearCyclicSCM

_METHOD = "compare_equilibrium_unrolling"


def compare_equilibrium_unrolling(
    scm: LinearCyclicSCM,
    *,
    do: Mapping[str, float] | None = None,
    horizon: int = 256,
    tol: float = 1e-3,
    seed: int = 0,
    learning_rate: float | None = None,
) -> Certificate:
    """Does the long-run unrolled ``do`` converge to the equilibrium ``do``?

    Parameters
    ----------
    scm:
        The linear cyclic SCM.
    do:
        Optional intervention applied to both the equilibrium and the unrolled dynamics.
    horizon:
        Number of unrolling steps (the "long run").
    tol:
        Sup-norm gap below which the unrolled and equilibrium means are deemed converged.
    seed:
        Seed recorded in the provenance.
    learning_rate:
        ``None`` unrolls the naive dynamics ``x_{k+1} = B x_k + E[u]``. A ``gamma`` in ``(0, 1]``
        unrolls the damped adaptive dynamics ``x_{k+1} = x_k + gamma (B x_k + E[u] - x_k)`` -- the
        constant-gain (Euler) form of the mean learning dynamics. For
        ``gamma < max_stable_learning_rate()`` this converges exactly when the mean dynamics are
        stable (positive ``stability_margin``), a strictly larger class than the contractive one.
    """
    if learning_rate is not None and not 0.0 < learning_rate <= 1.0:
        raise ValueError(f"learning_rate must be in (0, 1], got {learning_rate}")
    equilibrium = scm.solve(do=do)
    if not equilibrium.solved:
        return _no_equilibrium_certificate(equilibrium.hedge, seed)
    unrolled_mean = _unrolled_state_mean(scm, do, horizon, learning_rate)
    return _convergence_certificate(
        scm, do, equilibrium.mean, unrolled_mean, horizon, tol, seed, learning_rate
    )


def _unrolled_state_mean(
    scm: LinearCyclicSCM,
    do: Mapping[str, float] | None,
    horizon: int,
    learning_rate: float | None = None,
) -> FloatArray:
    """Mean of the unrolled state after ``horizon`` steps from 0 (naive or damped dynamics).

    The exact linear mean dynamics of the (do-intervened) system; whenever the iteration is
    stable it converges to the same equilibrium ``(I - B)^{-1} E[u]`` -- the damping changes the
    convergence class, never the fixed point.
    """
    intervened = scm.intervene(do) if do else scm
    state: FloatArray = np.zeros(intervened.dim)
    for _ in range(horizon):
        step = intervened.coefficients @ state + intervened.noise_mean
        state = step if learning_rate is None else state + learning_rate * (step - state)
    return state


def _iteration_spectral_radius(scm: LinearCyclicSCM, learning_rate: float | None) -> float:
    """Spectral radius of the effective iteration map (``B`` or ``I + gamma (B - I)``)."""
    if scm.dim == 0:
        return 0.0
    if learning_rate is None:
        return scm.spectral_radius()
    eigenvalues = 1.0 + learning_rate * (np.linalg.eigvals(scm.coefficients) - 1.0)
    return float(np.max(np.abs(eigenvalues)))


def _convergence_certificate(
    scm: LinearCyclicSCM,
    do: Mapping[str, float] | None,
    equilibrium_mean: FloatArray | None,
    unrolled_mean: FloatArray,
    horizon: int,
    tol: float,
    seed: int,
    learning_rate: float | None = None,
) -> Certificate:
    assert equilibrium_mean is not None  # only called on a solved equilibrium
    gap = float(np.max(np.abs(unrolled_mean - equilibrium_mean)))
    intervened = scm.intervene(do) if do else scm
    rho = intervened.spectral_radius()
    margin = intervened.stability_margin()
    gamma_star = intervened.max_stable_learning_rate()
    iteration_rho = _iteration_spectral_radius(intervened, learning_rate)
    stable = iteration_rho < 1.0 - 1e-9
    converged = gap <= tol

    witness = Witness(
        "equilibrium-unrolling",
        {
            "equilibrium_mean": [float(v) for v in equilibrium_mean],
            "unrolled_mean": [float(v) for v in unrolled_mean],
            "horizon": horizon,
            "gap": gap,
            "spectral_radius": rho,
            "spectral_abscissa": intervened.spectral_abscissa(),
            "stability_margin": margin,
            "max_stable_learning_rate": gamma_star,
            "learning_rate": learning_rate,
            "iteration_spectral_radius": iteration_rho,
        },
    )
    assumption = Assumption(
        "contractive" if learning_rate is None else "stable-mean-dynamics",
        {
            "spectral_radius": rho,
            "iteration_spectral_radius": iteration_rho,
            "stability_margin": margin,
        },
        checkable=True,
        diagnostic={"gap": gap, "horizon": horizon, "converged": converged},
    )

    claim: str
    kind: Kind
    hedge: Hedge | None
    if stable and converged:
        claim = (
            "long-run unrolled do() converges to the equilibrium do() (contractive linear)"
            if learning_rate is None
            else "long-run adaptive-learning do() converges to the equilibrium do() "
            "(stable mean dynamics)"
        )
        kind = Kind.IDENTIFIED
        hedge = None
    elif stable:
        claim = "equilibrium vs unrolled do(): stable but not yet converged at this horizon"
        kind = Kind.EMPIRICAL
        hedge = Hedge(
            "unrolling has not converged at this horizon (gap > tol); increase the horizon",
            detail={"gap": gap, "tol": tol, "horizon": horizon},
        )
    elif learning_rate is None:
        claim = "equilibrium vs unrolled do(): non-contractive, empirical comparison only"
        kind = Kind.EMPIRICAL
        reason = (
            "non-contractive system (spectral radius >= 1): unrolling is not guaranteed to "
            "converge to the equilibrium"
        )
        if margin > 0.0:
            reason += (
                "; the mean dynamics are nonetheless stable -- an adaptive unrolling with "
                "learning_rate below max_stable_learning_rate converges"
            )
        hedge = Hedge(
            reason,
            detail={
                "spectral_radius": rho,
                "gap": gap,
                "stability_margin": margin,
                "max_stable_learning_rate": gamma_star,
            },
        )
    else:
        claim = "equilibrium vs adaptive-learning do(): unstable at this learning rate"
        kind = Kind.EMPIRICAL
        reason = (
            "adaptive dynamics are unstable at this learning rate"
            + (
                "; reduce learning_rate below max_stable_learning_rate"
                if margin > 0.0
                else "; no learning rate stabilises these mean dynamics (negative "
                "stability margin)"
            )
        )
        hedge = Hedge(
            reason,
            detail={
                "learning_rate": learning_rate,
                "iteration_spectral_radius": iteration_rho,
                "gap": gap,
                "stability_margin": margin,
                "max_stable_learning_rate": gamma_star,
            },
        )

    return Certificate(
        claim=claim,
        estimand=EstimandSpec(query="equilibrium", target="mean"),
        kind=kind,
        value=gap,
        alpha=None,
        assumptions=(assumption,),
        method=_METHOD,
        witness=witness,
        hedge=hedge,
        provenance=Provenance.create(seeds=(seed,)),
    )


def _no_equilibrium_certificate(hedge: Hedge | None, seed: int) -> Certificate:
    return Certificate(
        claim="cannot compare unrolling to equilibrium: no unique equilibrium do()",
        estimand=EstimandSpec(query="equilibrium", target="mean"),
        kind=Kind.EMPIRICAL,
        value=None,
        alpha=None,
        assumptions=(),
        method=_METHOD,
        witness=None,
        hedge=hedge,
        provenance=Provenance.create(seeds=(seed,)),
    )
