"""Smoke test for GPT next-token dataset and dataloader."""

from __future__ import annotations

from pathlib import Path

import torch

from src.dataset_gpt import GPTNextTokenDataset, load_token_ids
from src.make_dataloader import make_dataloader


def main() -> int:
    """Run basic assertions for dataset and dataloader behavior."""
    tokens_path = Path("data/the-verdict.tokens.pt")

    if tokens_path.exists():
        token_ids = load_token_ids(str(tokens_path))
        print(f"Loaded tokens from: {tokens_path}")
    else:
        token_ids = torch.arange(1000, dtype=torch.long)
        print("Token file not found; using synthetic token stream.")

    context_length = 16
    stride = 8

    dataset = GPTNextTokenDataset(
        token_ids=token_ids,
        context_length=context_length,
        stride=stride,
    )

    assert len(dataset) > 0, "Dataset should contain at least one sample"
    x, y = dataset[0]
    assert x.shape == (16,), f"Expected x shape (16,), got {tuple(x.shape)}"
    assert y.shape == (16,), f"Expected y shape (16,), got {tuple(y.shape)}"
    assert x.dtype == torch.long and y.dtype == torch.long, "x and y must be torch.long"
    assert torch.equal(y[:-1], x[1:]), "y[:-1] must equal x[1:] for next-token shift"

    if tokens_path.exists():
        loader = make_dataloader(
            tokens_path=str(tokens_path),
            context_length=context_length,
            stride=stride,
            batch_size=4,
            shuffle=True,
            num_workers=0,
        )
    else:
        loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

    batch_x, batch_y = next(iter(loader))
    assert batch_x.shape == (4, 16), f"Expected batch_x shape (4, 16), got {tuple(batch_x.shape)}"
    assert batch_y.shape == (4, 16), f"Expected batch_y shape (4, 16), got {tuple(batch_y.shape)}"
    assert batch_x.dtype == torch.long and batch_y.dtype == torch.long, "Batch tensors must be torch.long"

    print("Dataset smoke test passed.")
    print(f"dataset length: {len(dataset)}")
    print(f"sample x shape: {tuple(x.shape)}, y shape: {tuple(y.shape)}")
    print(f"batch x shape: {tuple(batch_x.shape)}, y shape: {tuple(batch_y.shape)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
