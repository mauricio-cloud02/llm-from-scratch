"""Utilities for resolving verdict data paths and token-stream splitting."""

from __future__ import annotations

from pathlib import Path

import torch


def resolve_verdict_path() -> Path:
    """Resolve verdict text path from project-root cwd."""
    path = (Path.cwd() / "data" / "the-verdict.txt").resolve()
    if not path.exists():
        raise FileNotFoundError(f"Missing verdict text file: {path}")
    return path


def resolve_verdict_tokens_path() -> Path:
    """Resolve verdict token path from project-root cwd."""
    path = (Path.cwd() / "data" / "the-verdict.tokens.pt").resolve()
    if not path.exists():
        raise FileNotFoundError(
            "Missing token file: "
            f"{path}. Run `python -m src.tokenize_verdict` first."
        )
    return path


def train_val_split_tokens(
    token_ids: torch.LongTensor,
    train_ratio: float = 0.9,
) -> tuple[torch.LongTensor, torch.LongTensor]:
    """Split a 1D token stream into train/validation portions while preserving order."""
    if not (0.0 < train_ratio < 1.0):
        raise ValueError(f"train_ratio must satisfy 0 < train_ratio < 1, got {train_ratio}")
    if not isinstance(token_ids, torch.Tensor):
        raise TypeError("token_ids must be a torch.Tensor")
    if token_ids.ndim != 1:
        raise ValueError(f"token_ids must be 1D, got shape={tuple(token_ids.shape)}")
    if token_ids.dtype != torch.long:
        raise TypeError(f"token_ids must have dtype torch.long, got {token_ids.dtype}")

    split_idx = int(token_ids.numel() * train_ratio)
    train_ids = token_ids[:split_idx]
    val_ids = token_ids[split_idx:]
    return train_ids, val_ids

