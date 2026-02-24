"""GPT model composed of embeddings, Transformer blocks, and LM head."""

from __future__ import annotations

import torch
from torch import nn

from src.layernorm import LayerNorm
from src.transformer_block import TransformerBlock


class GPTModel(nn.Module):
    """GPT-style decoder-only Transformer model."""

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.cfg = cfg

        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx: torch.LongTensor) -> torch.Tensor:
        assert in_idx.ndim == 2 and in_idx.dtype == torch.long, (
            f"Expected in_idx shape (B, T) and dtype long, got shape={tuple(in_idx.shape)}, "
            f"dtype={in_idx.dtype}"
        )

        batch_size, seq_len = in_idx.shape
        assert seq_len <= self.cfg["context_length"], (
            f"seq_len={seq_len} exceeds context_length={self.cfg['context_length']}"
        )

        tok_embeds = self.tok_emb(in_idx)  # (B, T, D)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))  # (T, D)

        x = tok_embeds + pos_embeds  # (B, T, D)
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)  # (B, T, V)

        assert logits.shape == (batch_size, seq_len, self.cfg["vocab_size"]), (
            "Unexpected logits shape: "
            f"{tuple(logits.shape)} != {(batch_size, seq_len, self.cfg['vocab_size'])}"
        )
        return logits

