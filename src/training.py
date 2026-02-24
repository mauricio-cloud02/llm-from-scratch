"""Training and evaluation helpers for next-token language modeling."""

from __future__ import annotations

from typing import Any

import torch

from src.next_token_loss import calc_loss_batch, calc_loss_loader


def evaluate_model(model, train_loader, val_loader, device, eval_iter: int):
    """Evaluate average train/val loss over a fixed number of loader batches."""
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def train_model_simple(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    num_epochs,
    eval_freq,
    eval_iter,
    start_context,
    tokenizer,
):
    """Run a simple training loop with periodic evaluation and token tracking."""
    train_losses: list[float] = []
    val_losses: list[float] = []
    track_tokens_seen: list[int] = []

    tokens_seen = 0
    global_step = -1

    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()

            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    device=device,
                    eval_iter=eval_iter,
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(
                    f"Ep {epoch + 1} (step {global_step:06d}): "
                    f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
                    f"tokens_seen={tokens_seen}"
                )

        if tokenizer is not None:
            try:
                from src.generation import generate_and_print_sample  # type: ignore

                generate_and_print_sample(
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    start_context=start_context,
                )
            except Exception:
                pass

    return train_losses, val_losses, track_tokens_seen

