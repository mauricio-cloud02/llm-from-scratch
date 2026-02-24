"""Runnable training entrypoint for the verdict token dataset."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from src import config
from src.data_split import resolve_verdict_tokens_path
from src.gpt_model import GPTModel
from src.make_splits_and_loaders import build_train_val_loaders
from src.training import train_model_simple


def _resolve_model_cfg_from_config() -> dict:
    """Resolve MODEL_CFG from src.config with minimal compatibility fallbacks."""
    cfg = dict(getattr(config, "MODEL_CFG", {}))
    if cfg:
        return cfg

    meta_path = Path.cwd() / "data" / "the-verdict.tokens.meta.json"
    vocab_size = int(getattr(config, "VOCAB_SIZE", 0))
    if vocab_size <= 0 and meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if isinstance(meta, dict) and "vocab_size" in meta:
            vocab_size = int(meta["vocab_size"])
    if vocab_size <= 0:
        vocab_size = 50257

    return {
        "vocab_size": vocab_size,
        "context_length": int(getattr(config, "TRAIN_CONTEXT_LENGTH", config.CONTEXT_LENGTH)),
        "emb_dim": int(getattr(config, "EMBED_DIM")),
        "n_heads": int(getattr(config, "NUM_HEADS")),
        "n_layers": int(getattr(config, "NUM_LAYERS")),
        "drop_rate": float(getattr(config, "DROP_RATE", 0.1)),
        "qkv_bias": bool(getattr(config, "QKV_BIAS", False)),
    }


def main() -> int:
    """Train GPT model for a few epochs on verdict tokens."""
    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    tokens_path = resolve_verdict_tokens_path()
    model_cfg = _resolve_model_cfg_from_config()

    context_length = int(getattr(config, "TRAIN_CONTEXT_LENGTH", config.CONTEXT_LENGTH))
    stride = int(getattr(config, "TRAIN_STRIDE", config.STRIDE))
    batch_size = int(config.BATCH_SIZE)
    learning_rate = float(config.LEARNING_RATE)

    train_loader, val_loader = build_train_val_loaders(
        tokens_path=tokens_path,
        train_ratio=0.9,
        batch_size=batch_size,
        context_length=context_length,
        stride=stride,
        num_workers=0,
    )

    model = GPTModel(model_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.1)

    train_losses, val_losses, tokens_seen = train_model_simple(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=2,
        eval_freq=5,
        eval_iter=5,
        start_context="",
        tokenizer=None,
    )

    if train_losses and val_losses:
        print(f"Final train loss: {train_losses[-1]:.4f}")
        print(f"Final val loss: {val_losses[-1]:.4f}")
    print(f"Tracked eval points: {len(tokens_seen)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
