"""Smoke test for generate() utility."""

from __future__ import annotations

import torch
from torch import nn

from src.generate import generate


class DummyLM(nn.Module):
    """Tiny dummy LM that maps token IDs to logits."""

    def __init__(self, vocab_size: int, emb_dim: int) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.head = nn.Linear(emb_dim, vocab_size)

    def forward(self, idx: torch.LongTensor) -> torch.Tensor:
        x = self.emb(idx)
        logits = self.head(x)
        return logits


def _check_output(out: torch.LongTensor, vocab_size: int, base_len: int, max_new_tokens: int) -> None:
    assert out.ndim == 2 and out.shape[0] == 1
    assert out.dtype == torch.long
    assert base_len <= out.shape[1] <= base_len + max_new_tokens
    assert torch.all(out >= 0).item()
    assert torch.all(out < vocab_size).item()


def main() -> int:
    """Run generation checks for greedy, sampling, and top-k sampling."""
    torch.manual_seed(123)

    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    vocab_size = 50
    emb_dim = 16
    base_len = 8
    max_new_tokens = 5
    context_size = 8

    model = DummyLM(vocab_size=vocab_size, emb_dim=emb_dim).to(device)
    idx = torch.randint(0, vocab_size, (1, base_len), dtype=torch.long, device=device)

    out_greedy = generate(
        model=model,
        idx=idx.clone(),
        max_new_tokens=max_new_tokens,
        context_size=context_size,
        temperature=0.0,
        top_k=None,
    )
    _check_output(out_greedy, vocab_size, base_len, max_new_tokens)

    out_sampling = generate(
        model=model,
        idx=idx.clone(),
        max_new_tokens=max_new_tokens,
        context_size=context_size,
        temperature=0.8,
        top_k=None,
    )
    _check_output(out_sampling, vocab_size, base_len, max_new_tokens)

    out_topk = generate(
        model=model,
        idx=idx.clone(),
        max_new_tokens=max_new_tokens,
        context_size=context_size,
        temperature=0.8,
        top_k=10,
    )
    _check_output(out_topk, vocab_size, base_len, max_new_tokens)

    print(f"greedy output shape: {tuple(out_greedy.shape)}")
    print(f"sampling output shape: {tuple(out_sampling.shape)}")
    print(f"top-k output shape: {tuple(out_topk.shape)}")
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

