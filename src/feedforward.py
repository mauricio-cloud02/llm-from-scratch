"""GPT-style feed-forward network (MLP)."""

from __future__ import annotations

import torch
from torch import nn

from src.gelu import gelu


class FeedForward(nn.Module):
    """Transformer feed-forward block.

    Contract:
    - input: x with shape (b, T, emb_dim)
    - output: same shape as input
    """

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        if "emb_dim" not in cfg:
            raise KeyError("cfg must include 'emb_dim'")

        emb_dim = int(cfg["emb_dim"])
        if emb_dim <= 0:
            raise ValueError("emb_dim must be > 0")

        if "drop_rate" in cfg:
            drop_p = float(cfg["drop_rate"])
        elif "dropout" in cfg:
            drop_p = float(cfg["dropout"])
        elif "drop_rate_ff" in cfg:
            drop_p = float(cfg["drop_rate_ff"])
        else:
            drop_p = 0.0

        self.emb_dim = emb_dim
        self.fc1 = nn.Linear(emb_dim, 4 * emb_dim)
        self.fc2 = nn.Linear(4 * emb_dim, emb_dim)
        self.dropout = nn.Dropout(drop_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 3, f"Expected x shape (b, T, emb_dim), got {tuple(x.shape)}"
        assert x.shape[-1] == self.emb_dim, (
            f"Expected last dim emb_dim={self.emb_dim}, got {x.shape[-1]}"
        )

        out = self.fc1(x)
        out = gelu(out)
        out = self.fc2(out)
        out = self.dropout(out)

        assert out.shape == x.shape, f"Output shape mismatch: {tuple(out.shape)} vs {tuple(x.shape)}"
        return out

