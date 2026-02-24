"""Smoke test for calc_loss_batch and calc_loss_loader."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.next_token_loss import calc_loss_batch, calc_loss_loader


class DummyModel(nn.Module):
    """Tiny next-token model: token embedding + linear head."""

    def __init__(self, vocab_size: int, emb_dim: int) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.head = nn.Linear(emb_dim, vocab_size)

    def forward(self, x: torch.LongTensor) -> torch.Tensor:
        assert x.ndim == 2 and x.dtype == torch.long
        h = self.emb(x)        # (B, T, D)
        logits = self.head(h)  # (B, T, V)
        return logits


def main() -> int:
    """Run deterministic smoke checks for batch and loader loss helpers."""
    torch.manual_seed(123)

    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    batch_size, seq_len, vocab_size = 2, 4, 10
    emb_dim = 8

    model = DummyModel(vocab_size=vocab_size, emb_dim=emb_dim).to(device)

    input_batch = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long)
    target_batch = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long)

    loss = calc_loss_batch(input_batch, target_batch, model, device)
    assert isinstance(loss, torch.Tensor) and loss.ndim == 0, "loss must be a scalar tensor"
    assert torch.isfinite(loss).item(), "loss must be finite"

    model.zero_grad(set_to_none=True)
    loss.backward()
    has_grad = any(p.grad is not None for p in model.parameters())
    assert has_grad, "expected at least one parameter grad after backward"

    # 2 batches total: 4 samples with batch_size=2
    inputs = torch.randint(0, vocab_size, (4, seq_len), dtype=torch.long)
    targets = torch.randint(0, vocab_size, (4, seq_len), dtype=torch.long)
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=batch_size, shuffle=False)
    assert len(loader) == 2, f"expected 2 batches, got {len(loader)}"

    loader_loss = calc_loss_loader(loader, model, device)
    assert isinstance(loader_loss, float), "loader loss must be float"
    assert torch.isfinite(torch.tensor(loader_loss)).item(), "loader loss must be finite"

    print(f"device: {device}")
    print(f"input_batch shape: {tuple(input_batch.shape)}")
    print(f"target_batch shape: {tuple(target_batch.shape)}")
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
