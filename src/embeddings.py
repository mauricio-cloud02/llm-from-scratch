"""Embedding layers for token and absolute positional representations."""

from __future__ import annotations

import torch
from torch import nn


class TokenEmbedding(nn.Module):
    """Token ID to dense embedding lookup.

    Contract:
    - input: x_ids with shape (b, T), dtype torch.long
    - output: shape (b, T, d_model), floating dtype
    """

    def __init__(self, vocab_size: int, d_model: int) -> None:
        super().__init__()
        if vocab_size <= 0:
            raise ValueError("vocab_size must be > 0")
        if d_model <= 0:
            raise ValueError("d_model must be > 0")

        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x_ids: torch.LongTensor) -> torch.Tensor:
        assert x_ids.ndim == 2, f"Expected x_ids shape (b, T), got {tuple(x_ids.shape)}"
        assert x_ids.dtype == torch.long, f"Expected x_ids dtype long, got {x_ids.dtype}"

        out = self.embedding(x_ids)
        assert out.ndim == 3, f"Expected output shape (b, T, d_model), got {tuple(out.shape)}"
        return out


class AbsolutePositionalEmbedding(nn.Module):
    """Learned absolute positional embedding.

    Contract:
    - input: x_ids with shape (b, T), dtype torch.long
    - output: shape (b, T, d_model), floating dtype
    """

    def __init__(self, context_length: int, d_model: int) -> None:
        super().__init__()
        if context_length <= 0:
            raise ValueError("context_length must be > 0")
        if d_model <= 0:
            raise ValueError("d_model must be > 0")

        self.context_length = context_length
        self.embedding = nn.Embedding(context_length, d_model)

    def forward(self, x_ids: torch.LongTensor) -> torch.Tensor:
        assert x_ids.ndim == 2, f"Expected x_ids shape (b, T), got {tuple(x_ids.shape)}"
        assert x_ids.dtype == torch.long, f"Expected x_ids dtype long, got {x_ids.dtype}"

        bsz, seq_len = x_ids.shape
        assert seq_len <= self.context_length, (
            f"Sequence length T={seq_len} exceeds context_length={self.context_length}"
        )

        positions = torch.arange(seq_len, device=x_ids.device, dtype=torch.long)
        pos = self.embedding(positions)  # (T, d_model)
        pos = pos.unsqueeze(0).expand(bsz, -1, -1)  # (b, T, d_model)

        assert pos.shape[:2] == (bsz, seq_len)
        return pos


class TokenPlusPositionEmbedding(nn.Module):
    """Sum token and absolute position embeddings, then apply dropout."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        context_length: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.context_length = context_length
        self.token_emb = TokenEmbedding(vocab_size=vocab_size, d_model=d_model)
        self.pos_emb = AbsolutePositionalEmbedding(
            context_length=context_length,
            d_model=d_model,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_ids: torch.LongTensor) -> torch.Tensor:
        assert x_ids.ndim == 2, f"Expected x_ids shape (b, T), got {tuple(x_ids.shape)}"
        assert x_ids.dtype == torch.long, f"Expected x_ids dtype long, got {x_ids.dtype}"

        _, seq_len = x_ids.shape
        assert seq_len <= self.context_length, (
            f"Sequence length T={seq_len} exceeds context_length={self.context_length}"
        )

        out = self.token_emb(x_ids) + self.pos_emb(x_ids)
        out = self.dropout(out)
        assert out.ndim == 3, f"Expected output shape (b, T, d_model), got {tuple(out.shape)}"
        assert torch.is_floating_point(out), "Expected floating output tensor"
        return out
