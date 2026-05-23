"""Thin numerics seam. PyTorch today; an alternate backend can re-implement this module."""

from __future__ import annotations

import torch

Tensor = torch.Tensor


def default_generator(seed: int | None = None) -> torch.Generator:
    """Return a torch.Generator, optionally seeded, for reproducible sampling."""
    gen = torch.Generator()
    if seed is not None:
        gen.manual_seed(seed)
    return gen
