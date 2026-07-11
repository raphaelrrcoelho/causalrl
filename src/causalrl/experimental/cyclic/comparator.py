"""Equilibrium-vs-unrolling comparator for cyclic SCMs (experimental; plan §11).

A cyclic SCM has two natural interventional semantics: the **equilibrium** ``do`` (replace the
mechanism, re-solve the fixed point) and the **unrolled** ``do`` (run the dynamics
``x_{k+1} = B x_k + u`` forward under the intervention). :func:`compare_equilibrium_unrolling` asks
whether the long-run unrolled ``do`` converges to the equilibrium ``do`` and returns a
:class:`~causalrl.certify.certificate.Certificate`:

* **IDENTIFIED** when the (intervened) system is contractive -- ``rho(B) < 1`` guarantees the
  unrolling converges to the equilibrium, and the measured gap confirms it;
* **EMPIRICAL** (hedged) when the system is non-contractive (no convergence guarantee) or has not
  yet converged at the chosen horizon;
* **EMPIRICAL** with the solvability hedge when there is no unique equilibrium to compare against.

The unrolled side reuses the shipped :func:`causalrl.scm.unrolled.build_unrolled_scm` (imported
lazily, since it pulls in the optional torch backend); the equilibrium side and all certificate
logic are pure NumPy.
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
from causalrl.experimental.cyclic._unroll import unrolled_state_mean
from causalrl.experimental.cyclic.scm import FloatArray, LinearCyclicSCM

_METHOD = "compare_equilibrium_unrolling"


def compare_equilibrium_unrolling(
    scm: LinearCyclicSCM,
    *,
    do: Mapping[str, float] | None = None,
    horizon: int = 256,
    tol: float = 1e-3,
    seed: int = 0,
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
        Seed threaded into the unrolled sampler and the provenance record.
    """
    equilibrium = scm.solve(do=do)
    if not equilibrium.solved:
        return _no_equilibrium_certificate(equilibrium.hedge, seed)
    unrolled_mean = unrolled_state_mean(scm, do, horizon, seed)
    return _convergence_certificate(scm, do, equilibrium.mean, unrolled_mean, horizon, tol, seed)


def _convergence_certificate(
    scm: LinearCyclicSCM,
    do: Mapping[str, float] | None,
    equilibrium_mean: FloatArray | None,
    unrolled_mean: FloatArray,
    horizon: int,
    tol: float,
    seed: int,
) -> Certificate:
    assert equilibrium_mean is not None  # only called on a solved equilibrium
    gap = float(np.max(np.abs(unrolled_mean - equilibrium_mean)))
    intervened = scm.intervene(do) if do else scm
    rho = intervened.spectral_radius()
    contractive = rho < 1.0 - 1e-9
    converged = gap <= tol

    witness = Witness(
        "equilibrium-unrolling",
        {
            "equilibrium_mean": [float(v) for v in equilibrium_mean],
            "unrolled_mean": [float(v) for v in unrolled_mean],
            "horizon": horizon,
            "gap": gap,
            "spectral_radius": rho,
        },
    )
    assumption = Assumption(
        "contractive",
        {"spectral_radius": rho},
        checkable=True,
        diagnostic={"gap": gap, "horizon": horizon, "converged": converged},
    )

    claim: str
    kind: Kind
    hedge: Hedge | None
    if contractive and converged:
        claim = "long-run unrolled do() converges to the equilibrium do() (contractive linear)"
        kind = Kind.IDENTIFIED
        hedge = None
    elif contractive:
        claim = "equilibrium vs unrolled do(): contractive but not yet converged at this horizon"
        kind = Kind.EMPIRICAL
        hedge = Hedge(
            "unrolling has not converged at this horizon (gap > tol); increase the horizon",
            detail={"gap": gap, "tol": tol, "horizon": horizon},
        )
    else:
        claim = "equilibrium vs unrolled do(): non-contractive, empirical comparison only"
        kind = Kind.EMPIRICAL
        hedge = Hedge(
            "non-contractive system (spectral radius >= 1): unrolling is not guaranteed to "
            "converge to the equilibrium",
            detail={"spectral_radius": rho, "gap": gap},
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
