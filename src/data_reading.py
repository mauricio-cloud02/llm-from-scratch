"""Utilities for reading text data."""


def read_text_file(path: str) -> str:
    """Read a UTF-8 text file safely and return its content."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()
