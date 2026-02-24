"""Manual GPT-style GELU (tanh approximation)."""

from __future__ import annotations

import math

import torch


def gelu(x: torch.Tensor) -> torch.Tensor:
    """Apply GPT-style GELU approximation.

    Contract:
    - input: tensor of any shape
    - output: tensor with same shape and floating dtype
    """
    if not torch.is_floating_point(x):
        x = x.float()

    coeff = math.sqrt(2.0 / math.pi)
    out = 0.5 * x * (1.0 + torch.tanh(coeff * (x + 0.044715 * x.pow(3))))

    assert out.shape == x.shape, f"Output shape mismatch: {tuple(out.shape)} vs {tuple(x.shape)}"
    assert torch.is_floating_point(out), f"Expected floating output dtype, got {out.dtype}"
    return out
