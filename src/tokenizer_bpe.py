"""BPE tokenizer wrapper backed by tiktoken."""

from __future__ import annotations

import torch


class BPETokenizer:
    """Minimal deterministic BPE tokenizer using a named tiktoken encoding."""

    def __init__(self, encoding_name: str = "gpt2") -> None:
        self.encoding_name = encoding_name
        try:
            import tiktoken
        except ImportError as exc:
            raise ImportError(
                "Missing dependency 'tiktoken'. Install it with: pip install tiktoken"
            ) from exc

        self._encoding = tiktoken.get_encoding(encoding_name)

    def encode(self, text: str) -> list[int]:
        """Encode text into token IDs."""
        return self._encoding.encode(text, disallowed_special=())

    def decode(self, token_ids: list[int]) -> str:
        """Decode token IDs back into text."""
        return self._encoding.decode(token_ids)

    def encode_to_tensor(self, text: str, device: str | None = None) -> torch.LongTensor:
        """Encode text and return token IDs as a torch.LongTensor."""
        return torch.tensor(self.encode(text), dtype=torch.long, device=device)

    @property
    def vocab_size(self) -> int:
        """Return vocabulary size for the active BPE encoding."""
        return int(self._encoding.n_vocab)
