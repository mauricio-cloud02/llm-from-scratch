"""Smoke test for BPETokenizer."""

from src.tokenizer_bpe import BPETokenizer


def main() -> int:
    """Run a minimal encode/decode smoke test."""
    try:
        tokenizer = BPETokenizer()
    except ImportError as exc:
        print(str(exc))
        print("Install with: pip install tiktoken")
        return 1

    text = "Hello from llm-project."
    token_ids = tokenizer.encode(text)
    decoded = tokenizer.decode(token_ids)

    assert token_ids, "Token list should not be empty"
    assert decoded, "Decoded string should not be empty"

    print("Smoke test passed.")
    print(f"tokens: {token_ids}")
    print(f"decoded: {decoded}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
