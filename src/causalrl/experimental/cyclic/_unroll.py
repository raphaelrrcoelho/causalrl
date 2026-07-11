# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false, reportUnknownParameterType=false
# pyright: reportMissingParameterType=false, reportArgumentType=false, reportUnknownLambdaType=false
# pyright: reportAttributeAccessIssue=false, reportMissingTypeStubs=false
# pyright: reportPrivateImportUsage=false
# Isolated torch glue: torch's dynamic attributes (torch.eye/zeros/float64, tensor methods) are not
# fully resolvable by pyright; end-to-end correctness (incl. the StructuralCausalModel.see call) is
# verified by the torch-gated runtime tests on the main CI matrix, not by static types here.
"""Torch-backed unrolling for the cyclic comparator (experimental; plan §11).

Isolated in its own module so the torch-stub boundary (whose member types pyright cannot fully
resolve) does not relax strict type-checking of the pure-NumPy comparator logic -- the same
isolation the optional JAX backend uses. torch is imported lazily inside the function, so importing
this module never pulls in the optional deep backend.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from causalrl.experimental.cyclic.scm import FloatArray, LinearCyclicSCM


def unrolled_state_mean(
    scm: LinearCyclicSCM, do: Mapping[str, float] | None, horizon: int, seed: int
) -> FloatArray:
    """Mean of the unrolled state at ``horizon``, built with the shipped ``build_unrolled_scm``.

    Encodes the (do-intervened) linear dynamics ``x_{k+1} = B x_k + u`` as a shared-latent unrolled
    SCM (point-mass exogenous, i.e. mean dynamics) and reads back ``state_{horizon}``.
    """
    import torch
    from torch.distributions import MultivariateNormal

    from causalrl.scm.unrolled import build_unrolled_scm

    intervened = scm.intervene(do) if do else scm
    dim = intervened.dim
    coefficients = torch.tensor(intervened.coefficients, dtype=torch.float64)
    drive_mean = torch.tensor(intervened.noise_mean, dtype=torch.float64)
    point_mass = torch.eye(dim, dtype=torch.float64) * 1e-12  # ~deterministic mean dynamics

    def transition(state, action, latents, noise):
        return state @ coefficients.T + latents["drive"]

    unrolled = build_unrolled_scm(
        transition,
        horizon,
        state0_dist=MultivariateNormal(torch.zeros(dim, dtype=torch.float64), point_mass),
        latents={"drive": MultivariateNormal(drive_mean, point_mass)},
    )
    samples = unrolled.see(1, seed=seed)
    final = samples[f"state_{horizon}"].mean(dim=0)
    return np.asarray(final.detach().cpu().numpy(), dtype=np.float64)
