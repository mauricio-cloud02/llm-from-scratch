"""Smoke test for FeedForward module."""

from __future__ import annotations

import torch

from src.feedforward import FeedForward


def main() -> int:
    """Run shape and gradient checks for FeedForward."""
    cfg = {"emb_dim": 32, "drop_rate": 0.1}
    x = torch.randn(2, 16, 32, requires_grad=True)

    ff = FeedForward(cfg)
    out = ff(x)

    assert out.shape == x.shape, f"Expected shape {tuple(x.shape)}, got {tuple(out.shape)}"

    out.sum().backward()
    has_grad = any(p.grad is not None for p in ff.parameters())
    assert has_grad, "Expected at least one parameter to receive gradients"

    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

