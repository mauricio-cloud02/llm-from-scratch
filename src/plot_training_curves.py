"""Utility for plotting and saving training/validation loss curves."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def plot_loss_curves(
    train_losses: list[float],
    val_losses: list[float],
    tokens_seen: list[int],
    save_path: str = "results/loss_curve.png",
) -> None:
    """Plot train/validation losses against tokens seen and save to disk."""
    assert len(train_losses) == len(val_losses) == len(tokens_seen), (
        "train_losses, val_losses, and tokens_seen must have equal lengths"
    )
    assert len(train_losses) > 0, "Input lists must be non-empty"

    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(tokens_seen, train_losses, label="Train Loss")
    plt.plot(tokens_seen, val_losses, label="Validation Loss")
    plt.xlabel("Tokens Seen")
    plt.ylabel("Cross-Entropy Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"Saved loss curve to {save_path}")


if __name__ == "__main__":
    print("This module is intended to be imported and called from training script.")

