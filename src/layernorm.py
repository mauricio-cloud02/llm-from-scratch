"""Manual GPT-style LayerNorm implementation."""

from __future__ import annotations

import torch
from torch import nn


class LayerNorm(nn.Module):
    """Apply layer normalization over the embedding dimension.

    Contract:
    - input: x with shape (b, T, emb_dim)
    - output: same shape as input
    """

    def __init__(self, emb_dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        if emb_dim <= 0:
            raise ValueError("emb_dim must be > 0")
        if eps <= 0:
            raise ValueError("eps must be > 0")

        self.emb_dim = emb_dim
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 3, f"Expected x shape (b, T, emb_dim), got {tuple(x.shape)}"
        assert x.shape[-1] == self.emb_dim, (
            f"Expected last dim emb_dim={self.emb_dim}, got {x.shape[-1]}"
        )

        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        out = x_norm * self.scale + self.shift

        assert out.shape == x.shape, f"Output shape mismatch: {tuple(out.shape)} vs {tuple(x.shape)}"
        return out

