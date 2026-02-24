"""Smoke test for GPTModel."""

from __future__ import annotations

import torch

from src.gpt_model import GPTModel


def main() -> int:
    """Run shape and gradient checks for GPTModel."""
    cfg = {
        "vocab_size": 1000,
        "context_length": 64,
        "emb_dim": 32,
        "n_heads": 4,
        "n_layers": 2,
        "drop_rate": 0.1,
        "qkv_bias": False,
    }

    model = GPTModel(cfg)
    in_idx = torch.randint(0, cfg["vocab_size"], (2, 16), dtype=torch.long)
    logits = model(in_idx)

    assert logits.shape == (2, 16, cfg["vocab_size"]), (
        f"Expected logits shape {(2, 16, cfg['vocab_size'])}, got {tuple(logits.shape)}"
    )
    assert torch.is_floating_point(logits), f"Expected floating logits dtype, got {logits.dtype}"

    logits.sum().backward()
    has_grad = any(p.grad is not None for p in model.parameters())
    assert has_grad, "Expected at least one parameter to receive gradients"

    print(f"in_idx shape: {tuple(in_idx.shape)}")
    print(f"logits shape: {tuple(logits.shape)}")
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

