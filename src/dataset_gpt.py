"""Dataset utilities for next-token language modeling."""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import Dataset


class GPTNextTokenDataset(Dataset[tuple[torch.LongTensor, torch.LongTensor]]):
    """Create (x, y) training pairs from a 1D token stream.

    Shape contracts:
    - input token_ids: (N,) long
    - each x: (T,) long
    - each y: (T,) long where y is x shifted by one token in the source stream
    """

    def __init__(self, token_ids: torch.Tensor, context_length: int, stride: int = 1) -> None:
        """Initialize dataset from token IDs.

        Args:
            token_ids: 1D tensor of token IDs with dtype torch.long.
            context_length: Number of tokens per input sample (T), must be > 0.
            stride: Start-step between samples (S), must be > 0.
        """
        if not isinstance(token_ids, torch.Tensor):
            raise TypeError("token_ids must be a torch.Tensor")
        if token_ids.ndim != 1:
            raise ValueError(f"token_ids must be 1D, got shape={tuple(token_ids.shape)}")
        if token_ids.dtype != torch.long:
            raise TypeError(f"token_ids must have dtype torch.long, got {token_ids.dtype}")
        if context_length <= 0:
            raise ValueError("context_length must be > 0")
        if stride <= 0:
            raise ValueError("stride must be > 0")

        self.token_ids = token_ids
        self.context_length = context_length
        self.stride = stride

    def __len__(self) -> int:
        """Return number of valid (x, y) pairs.

        Valid starts are t = 0, S, 2S, ... where t + T + 1 <= N.
        """
        n_tokens = int(self.token_ids.numel())
        usable = n_tokens - (self.context_length + 1)
        if usable < 0:
            return 0
        return (usable // self.stride) + 1

    def __getitem__(self, idx: int) -> tuple[torch.LongTensor, torch.LongTensor]:
        """Return one next-token training pair.

        Args:
            idx: Sample index.

        Returns:
            x: Tensor of shape (T,) with dtype long.
            y: Tensor of shape (T,) with dtype long.
        """
        length = len(self)
        if idx < 0 or idx >= length:
            raise IndexError(f"index {idx} is out of bounds for dataset of length {length}")

        start = idx * self.stride
        end = start + self.context_length

        x = self.token_ids[start:end]
        y = self.token_ids[start + 1 : end + 1]
        return x, y


def load_token_ids(path: str) -> torch.LongTensor:
    """Load token IDs from a .pt file and normalize to 1D torch.LongTensor."""
    obj: Any = torch.load(path, map_location="cpu")

    if isinstance(obj, torch.Tensor):
        token_ids = obj
    elif isinstance(obj, list):
        token_ids = torch.tensor(obj, dtype=torch.long)
    else:
        raise TypeError(
            "Unsupported token data type in .pt file. Expected torch.Tensor or list[int]."
        )

    if token_ids.ndim != 1:
        raise ValueError(f"Loaded token IDs must be 1D, got shape={tuple(token_ids.shape)}")

    if token_ids.dtype != torch.long:
        token_ids = token_ids.to(dtype=torch.long)

    return token_ids
