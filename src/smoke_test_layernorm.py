"""Smoke test for manual LayerNorm."""

from __future__ import annotations

import torch

from src.layernorm import LayerNorm


def main() -> int:
    """Run basic shape and normalization checks."""
    x = torch.randn(2, 16, 32)
    ln = LayerNorm(emb_dim=32)
    out = ln(x)

    assert out.shape == x.shape, f"Expected shape {tuple(x.shape)}, got {tuple(out.shape)}"
    assert torch.is_floating_point(out), f"Expected floating dtype, got {out.dtype}"

    mean = out.mean(dim=-1)
    var = out.var(dim=-1, unbiased=False)

    assert torch.max(torch.abs(mean)).item() < 1e-4, "LayerNorm mean is not close to 0"
    assert torch.max(torch.abs(var - 1.0)).item() < 1e-3, "LayerNorm variance is not close to 1"

    print(f"x shape: {tuple(x.shape)}")
    print(f"out shape: {tuple(out.shape)}")
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

