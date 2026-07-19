"""Experimental cyclic-SCM support (plan §11; post-2.0, 2.x-experimental track).

Feedback systems -- control loops, coupled steady-state models -- are *cyclic* structural causal
models; the shipped core is acyclic-only. This package adds the minimum needed to reason about the
documented solvable classes (linear / contractive), always hedging outside them:

* :class:`CyclicCausalGraph` -- a directed graph that may contain cycles, with its
  strongly-connected-component structure and the Forré-Mooij acyclification.
* :func:`sigma_separated` -- sigma-separation, the cyclic Markov property (reduces to the shipped
  d-separation on acyclic inputs).
* :class:`LinearCyclicSCM` -- a linear cyclic SCM with solvability analysis and ``solve`` /
  equilibrium ``do`` that return a distribution over solutions or a typed hedge, never a silently
  chosen arbitrary solution.
* :func:`compare_equilibrium_unrolling` -- does long-run unrolled ``do`` converge to equilibrium
  ``do``? Returns a :class:`~eqcert.certify.certificate.Certificate`.

**Stability:** nothing in the stable public API imports this package; it is not API-frozen and is
outside eqcert's semver guarantees until promoted out of ``experimental/`` (plan §4, §14, §15).
"""

from __future__ import annotations

from eqcert.experimental.cyclic.comparator import compare_equilibrium_unrolling
from eqcert.experimental.cyclic.graph import CyclicCausalGraph
from eqcert.experimental.cyclic.scm import (
    CyclicSolution,
    CyclicSolveError,
    LinearCyclicSCM,
)
from eqcert.experimental.cyclic.separation import sigma_separated

__all__ = [
    "CyclicCausalGraph",
    "CyclicSolution",
    "CyclicSolveError",
    "LinearCyclicSCM",
    "compare_equilibrium_unrolling",
    "sigma_separated",
]
