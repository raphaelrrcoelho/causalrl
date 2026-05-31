"""Time-unrolled SCM: sequential abduct -> do -> re-roll counterfactuals."""

from __future__ import annotations

import torch
from torch.distributions import Bernoulli, Normal

from causalrl.scm.unrolled import build_unrolled_scm


def _gravity_transition(state, action, latents, noise):
    """x_{t+1} = x_t + (+1 if F else -1). Deterministic; ignores action and noise."""
    f = latents["F"]
    step = torch.where(f > 0.5, torch.ones_like(state), -torch.ones_like(state))
    return state + step


def test_unrolled_counterfactual_reroll_is_exact():
    scm = build_unrolled_scm(
        _gravity_transition,
        horizon=3,
        state0_dist=Normal(0.0, 1.0),
        latents={"F": Bernoulli(0.5)},
    )
    # Pin the (deterministic) start + latent, then re-roll under the SAME / flipped latent.
    post = scm.abduct(known={"state_0": torch.tensor([0.0]), "F": torch.tensor([1.0])}, n=1)
    assert abs(float(post.predict()["state_3"]) - 3.0) < 1e-6  # +1 * 3 steps
    assert abs(float(post.predict(do={"F": 0.0})["state_3"]) + 3.0) < 1e-6  # flipped: -3


def test_unrolled_per_sample_vector_latent():
    """Different latent value per trajectory in one batched abduction."""
    scm = build_unrolled_scm(
        _gravity_transition,
        horizon=2,
        state0_dist=Normal(0.0, 1.0),
        latents={"F": Bernoulli(0.5)},
    )
    post = scm.abduct(
        known={"state_0": torch.tensor([0.0, 10.0]), "F": torch.tensor([1.0, 0.0])}, n=2
    )
    out = post.predict()
    # sample 0: F=1 -> 0 + 2 = 2 ; sample 1: F=0 -> 10 - 2 = 8
    assert torch.allclose(out["state_2"], torch.tensor([2.0, 8.0]), atol=1e-6)


def test_horizon_must_be_positive():
    try:
        build_unrolled_scm(
            _gravity_transition,
            horizon=0,
            state0_dist=Normal(0.0, 1.0),
            latents={"F": Bernoulli(0.5)},
        )
    except ValueError:
        return
    raise AssertionError("expected ValueError for horizon=0")


def test_requires_at_least_one_latent():
    try:
        build_unrolled_scm(_gravity_transition, horizon=2, state0_dist=Normal(0.0, 1.0), latents={})
    except ValueError:
        return
    raise AssertionError("expected ValueError for empty latents")
