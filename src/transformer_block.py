"""GPT-style Transformer block (pre-norm + residual shortcuts)."""

from __future__ import annotations

import torch
from torch import nn

from src.attention import MultiHeadAttention
from src.feedforward import FeedForward
from src.layernorm import LayerNorm


class TransformerBlock(nn.Module):
    """Single Transformer block used in GPT-style decoders.

    Contract:
    - input: x with shape (batch_size, seq_len, emb_dim)
    - output: same shape as input
    """

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.emb_dim = int(cfg["emb_dim"])

        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"],
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 3, f"Expected x shape (b, T, emb_dim), got {tuple(x.shape)}"
        assert x.shape[-1] == self.emb_dim, (
            f"Expected last dim emb_dim={self.emb_dim}, got {x.shape[-1]}"
        )

        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        assert x.ndim == 3 and x.shape[-1] == self.emb_dim
        return x
