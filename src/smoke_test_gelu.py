"""Smoke test for manual GELU implementation."""

from __future__ import annotations

import torch

from src.gelu import gelu


def main() -> int:
    """Run basic GELU contract checks."""
    x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    out = gelu(x)

    assert out.shape == x.shape, f"Expected shape {tuple(x.shape)}, got {tuple(out.shape)}"
    assert torch.isclose(out[2], torch.tensor(0.0), atol=1e-7), "Expected GELU(0) ~= 0"
    assert torch.isfinite(out).all(), "GELU output must be finite"

    print(f"input: {x.tolist()}")
    print(f"output: {out.tolist()}")
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
