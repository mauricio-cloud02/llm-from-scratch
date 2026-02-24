"""Minimal interactive text-generation demo for GPTModel."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from src import config
from src.generate import generate
from src.gpt_model import GPTModel
from src.tokenizer_bpe import BPETokenizer


def _resolve_model_cfg(tokenizer: BPETokenizer) -> dict:
    """Load MODEL_CFG/GPT2_SMALL_MODEL from config with a minimal fallback."""
    if hasattr(config, "MODEL_CFG") and isinstance(config.MODEL_CFG, dict):
        return dict(config.MODEL_CFG)
    if hasattr(config, "GPT2_SMALL_MODEL") and isinstance(config.GPT2_SMALL_MODEL, dict):
        return dict(config.GPT2_SMALL_MODEL)

    meta_path = Path.cwd() / "data" / "the-verdict.tokens.meta.json"
    vocab_size = tokenizer.vocab_size
    if meta_path.exists():
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            vocab_size = int(meta.get("vocab_size", vocab_size))
        except Exception:
            pass

    return {
        "vocab_size": vocab_size,
        "context_length": int(getattr(config, "CONTEXT_LENGTH", 128)),
        "emb_dim": int(getattr(config, "EMBED_DIM", 128)),
        "n_heads": int(getattr(config, "NUM_HEADS", 4)),
        "n_layers": int(getattr(config, "NUM_LAYERS", 4)),
        "drop_rate": float(getattr(config, "DROP_RATE", 0.1)),
        "qkv_bias": bool(getattr(config, "QKV_BIAS", False)),
    }


def _load_checkpoint_if_available(model: GPTModel) -> None:
    """Load model weights from artifacts/*.pt when possible."""
    artifacts_dir = Path.cwd() / "artifacts"
    candidates = [
        artifacts_dir / "pretrain_small.pt",
        artifacts_dir / "model.pt",
    ]
    candidates.extend(sorted(artifacts_dir.glob("*.pt")))

    seen: set[Path] = set()
    for ckpt_path in candidates:
        if ckpt_path in seen or not ckpt_path.exists() or ckpt_path.stat().st_size == 0:
            continue
        seen.add(ckpt_path)
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu")
            state_dict = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
            if isinstance(state_dict, dict):
                model.load_state_dict(state_dict, strict=False)
                print(f"Loaded checkpoint: {ckpt_path}")
                return
        except Exception as exc:
            print(f"Skipping checkpoint {ckpt_path}: {exc}")

    print("No loadable checkpoint found in artifacts/. Using random weights.")


def main() -> int:
    """Run a small interactive generation demo."""
    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    tokenizer = BPETokenizer("gpt2")
    model_cfg = _resolve_model_cfg(tokenizer)

    model = GPTModel(model_cfg).to(device)
    model.eval()
    _load_checkpoint_if_available(model)

    default_prompt = "Every effort moves you forward."
    try:
        user_prompt = input("Prompt (leave empty for default): ").strip()
    except EOFError:
        user_prompt = ""
    prompt = user_prompt if user_prompt else default_prompt

    idx = tokenizer.encode_to_tensor(prompt, device=device).unsqueeze(0)

    out = generate(
        model=model,
        idx=idx,
        max_new_tokens=40,
        context_size=int(model_cfg["context_length"]),
        temperature=0.8,
        top_k=40,
    )

    decoded = tokenizer.decode(out[0].tolist())
    print("\n=== Generated Text ===")
    print(decoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

