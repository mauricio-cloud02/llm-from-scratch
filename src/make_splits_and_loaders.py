"""Build train/validation datasets and dataloaders from token IDs."""

from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader

from src import config
from src.data_split import train_val_split_tokens
from src.dataset_gpt import GPTNextTokenDataset, load_token_ids

BATCH_SIZE = int(getattr(config, "BATCH_SIZE"))
TRAIN_CONTEXT_LENGTH = int(getattr(config, "TRAIN_CONTEXT_LENGTH", getattr(config, "CONTEXT_LENGTH")))
TRAIN_STRIDE = int(getattr(config, "TRAIN_STRIDE", getattr(config, "STRIDE")))


def build_train_val_loaders(
    tokens_path: str | Path,
    train_ratio: float = 0.9,
    batch_size: int = BATCH_SIZE,
    context_length: int = TRAIN_CONTEXT_LENGTH,
    stride: int = TRAIN_STRIDE,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """Create train/validation DataLoaders from a token stream."""
    token_ids = load_token_ids(str(tokens_path))
    train_ids, val_ids = train_val_split_tokens(token_ids=token_ids, train_ratio=train_ratio)

    train_ds = GPTNextTokenDataset(
        token_ids=train_ids,
        context_length=context_length,
        stride=stride,
    )
    val_ds = GPTNextTokenDataset(
        token_ids=val_ids,
        context_length=context_length,
        stride=stride,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )
    return train_loader, val_loader

