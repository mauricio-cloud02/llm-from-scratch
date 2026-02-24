import torch
import torch.nn.functional as F

def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)

    logits = model(input_batch)  # (B, T, V)

    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),   # (B*T, V)
        target_batch.reshape(-1)               # (B*T,)
    )
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.0
    if len(data_loader) == 0:
        return float("nan")

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    model.eval()
    with torch.no_grad():
        for i, (input_batch, target_batch) in enumerate(data_loader):
            if i >= num_batches:
                break
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()

    return total_loss / num_batches


if __name__ == "__main__":
    print(
        "next_token_loss.py defines calc_loss_batch and calc_loss_loader. "
        "Import and call them from a training script."
    )
