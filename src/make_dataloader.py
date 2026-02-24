import torch
from pathlib import Path
from typing import Union

from torch.utils.data import DataLoader

from src.dataset_gpt import GPTNextTokenDataset, load_token_ids


def make_dataloader(
    tokens_path: Union[str, Path],
    context_length: int,
    stride: int,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    drop_last: bool = False,
) -> DataLoader:
    """Create a DataLoader for next-token language modeling.

    Returned batches follow shape contracts:
    - x: (batch_size, context_length), dtype torch.long
    - y: (batch_size, context_length), dtype torch.long
    """
    token_ids = load_token_ids(tokens_path)
    dataset = GPTNextTokenDataset(
        token_ids=token_ids,
        context_length=context_length,
        stride=stride,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=torch.cuda.is_available(),
    )
