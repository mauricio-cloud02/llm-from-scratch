"""Tokenize the verdict dataset with a BPE tokenizer."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from src.data_reading import read_text_file
from src.tokenizer_bpe import BPETokenizer


def _resolve_source_file() -> Path:
    """Resolve the input verdict text file path."""
    this_file = Path(__file__).resolve()
    project_root = this_file.parent.parent

    try:
        from src import config

        config_candidates = [
            getattr(config, "THE_VERDICT_PATH", None),
            getattr(config, "VERDICT_FILE", None),
            getattr(config, "SOURCE_FILE", None),
        ]
    except Exception:
        config_candidates = []

    candidates: list[Path] = []
    for candidate in config_candidates:
        if candidate:
            candidates.append(Path(str(candidate)))

    candidates.extend(
        [
            Path("data/the-verdict.txt"),
            project_root / "data" / "the-verdict.txt",
            this_file.parent / "data" / "the-verdict.txt",
        ]
    )

    for candidate in candidates:
        resolved = candidate if candidate.is_absolute() else (Path.cwd() / candidate)
        if resolved.exists():
            return resolved

    for root in [Path.cwd(), project_root]:
        matches = list(root.rglob("the-verdict.txt"))
        if matches:
            return matches[0]

    raise FileNotFoundError("Could not locate data/the-verdict.txt")


def main() -> int:
    """Run end-to-end tokenization for the verdict dataset."""
    try:
        tokenizer = BPETokenizer(encoding_name="gpt2")
    except ImportError as exc:
        print(str(exc))
        print("Install with: pip install tiktoken")
        return 1

    source_file = _resolve_source_file()
    text = read_text_file(str(source_file))
    token_ids = tokenizer.encode(text)

    print(f"number of characters: {len(text)}")
    print(f"number of tokens: {len(token_ids)}")
    print(f"first 30 token IDs: {token_ids[:30]}")
    print(f"decoded first 30 tokens: {tokenizer.decode(token_ids[:30])}")

    output_tokens_path = source_file.with_suffix(".tokens.pt")
    output_meta_path = source_file.with_suffix(".tokens.meta.json")

    torch.save(torch.tensor(token_ids, dtype=torch.long), output_tokens_path)

    metadata = {
        "encoding_name": tokenizer.encoding_name,
        "vocab_size": tokenizer.vocab_size,
        "num_tokens": len(token_ids),
        "source_file": str(source_file),
    }
    with open(output_meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"saved token tensor: {output_tokens_path}")
    print(f"saved metadata: {output_meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
