"""Smoke test for token split and train/validation dataloaders."""

from __future__ import annotations

import torch

from src import config
from src.data_split import resolve_verdict_tokens_path, train_val_split_tokens
from src.dataset_gpt import GPTNextTokenDataset, load_token_ids
from src.make_splits_and_loaders import build_train_val_loaders


def main() -> int:
    """Run split and dataloader shape checks."""
    tokens_path = resolve_verdict_tokens_path()
    token_ids = load_token_ids(str(tokens_path))
    train_ids, val_ids = train_val_split_tokens(token_ids)

    context_length = int(getattr(config, "TRAIN_CONTEXT_LENGTH", getattr(config, "CONTEXT_LENGTH")))
    stride = int(getattr(config, "TRAIN_STRIDE", getattr(config, "STRIDE")))
    requested_batch_size = int(getattr(config, "BATCH_SIZE"))

    train_ds = GPTNextTokenDataset(train_ids, context_length=context_length, stride=stride)
    val_ds = GPTNextTokenDataset(val_ids, context_length=context_length, stride=stride)
    batch_size = min(requested_batch_size, len(train_ds), len(val_ds))
    assert batch_size > 0, "Need at least one sample in both train and val splits"

    train_loader, val_loader = build_train_val_loaders(
        tokens_path=tokens_path,
        batch_size=batch_size,
        context_length=context_length,
        stride=stride,
    )

    x_train, y_train = next(iter(train_loader))
    x_val, y_val = next(iter(val_loader))

    assert x_train.shape == (batch_size, context_length)
    assert y_train.shape == (batch_size, context_length)
    assert x_val.shape == (batch_size, context_length)
    assert y_val.shape == (batch_size, context_length)

    assert x_train.dtype == torch.long and y_train.dtype == torch.long
    assert x_val.dtype == torch.long and y_val.dtype == torch.long

    print(f"tokens total: {token_ids.numel()}")
    print(f"train tokens: {train_ids.numel()}")
    print(f"val tokens: {val_ids.numel()}")
    print(f"train batch shapes: x={tuple(x_train.shape)}, y={tuple(y_train.shape)}")
    print(f"val batch shapes: x={tuple(x_val.shape)}, y={tuple(y_val.shape)}")
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
