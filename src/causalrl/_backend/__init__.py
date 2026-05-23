"""Thin numerics seam. PyTorch today; an alternate backend can re-implement this module."""

from __future__ import annotations

import torch

# torch.Generator/Tensor are public API but not re-exported in torch's type stubs.
Tensor = torch.Tensor
Generator = torch.Generator  # type: ignore[reportPrivateImportUsage]


def default_generator(seed: int | None = None) -> Generator:
    """Return a torch.Generator, optionally seeded, for reproducible sampling."""
    gen = Generator()
    if seed is not None:
        gen.manual_seed(seed)
    return gen
