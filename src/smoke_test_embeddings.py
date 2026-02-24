"""Smoke test for token and absolute positional embeddings."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from src.embeddings import TokenPlusPositionEmbedding
from src.make_dataloader import make_dataloader


def _resolve_vocab_size(default_vocab_size: int = 1000) -> int:
    """Resolve vocab size from metadata when available, else use a safe fallback."""
    meta_path = Path("data/the-verdict.tokens.meta.json")
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        vocab_size = int(meta.get("vocab_size", default_vocab_size))
        if vocab_size > 0:
            return vocab_size
    return default_vocab_size


def main() -> int:
    """Run a minimal forward pass and assert embedding shape contracts."""
    tokens_path = Path("data/the-verdict.tokens.pt")

    context_length = 16
    batch_size = 2

    try:
        from src import config

        d_model = int(getattr(config, "EMBED_DIM", 64))
        stride = int(getattr(config, "STRIDE", 8))
    except Exception:
        d_model = 64
        stride = 8

    vocab_size = _resolve_vocab_size(default_vocab_size=1000)

    if tokens_path.exists():
        loader = make_dataloader(
            tokens_path=str(tokens_path),
            context_length=context_length,
            stride=max(1, stride),
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
        )
        x_ids, _ = next(iter(loader))
        print(f"Using batch from DataLoader: x_ids shape={tuple(x_ids.shape)}")
    else:
        x_ids = torch.randint(0, vocab_size, (batch_size, context_length), dtype=torch.long)
        print("Token file not found; using synthetic x_ids.")

    assert x_ids.shape == (2, 16), f"Expected x_ids shape (2, 16), got {tuple(x_ids.shape)}"
    assert x_ids.dtype == torch.long, f"Expected x_ids dtype long, got {x_ids.dtype}"

    emb = TokenPlusPositionEmbedding(
        vocab_size=vocab_size,
        d_model=d_model,
        context_length=context_length,
        dropout=0.0,
    )

    out = emb(x_ids)
    assert out.shape == (2, 16, d_model), (
        f"Expected output shape (2, 16, {d_model}), got {tuple(out.shape)}"
    )
    assert torch.is_floating_point(out), f"Expected floating output dtype, got {out.dtype}"

    print(f"vocab_size={vocab_size}, d_model={d_model}, context_length={context_length}")
    print(f"output shape={tuple(out.shape)}, dtype={out.dtype}")
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
