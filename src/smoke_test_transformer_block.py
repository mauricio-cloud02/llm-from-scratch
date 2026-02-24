"""Smoke test for TransformerBlock."""

from __future__ import annotations

import torch

from src.transformer_block import TransformerBlock


def main() -> int:
    """Run shape and gradient checks for TransformerBlock."""
    cfg = {
        "emb_dim": 32,
        "context_length": 64,
        "n_heads": 4,
        "drop_rate": 0.1,
        "qkv_bias": False,
    }

    block = TransformerBlock(cfg)
    x = torch.randn(2, 16, cfg["emb_dim"])
    out = block(x)

    assert out.shape == x.shape, f"Expected shape {tuple(x.shape)}, got {tuple(out.shape)}"
    assert torch.is_floating_point(out), f"Expected floating dtype, got {out.dtype}"

    out.sum().backward()
    has_grad = any(p.grad is not None for p in block.parameters())
    assert has_grad, "Expected at least one parameter to receive gradients"

    print(f"input shape: {tuple(x.shape)}")
    print(f"output shape: {tuple(out.shape)}")
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

