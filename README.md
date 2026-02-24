# LLM Project: GPT-Style Language Modeling Pipeline

This repository implements a compact, end-to-end GPT-style language modeling pipeline over *The Verdict* dataset, covering tokenization, dataset construction, Transformer components, training/evaluation utilities, and text generation. The code is intentionally minimal and readable, with smoke tests for each stage so you can validate behavior incrementally before scaling experiments.

## Implemented Components

- BPE tokenization using `tiktoken` (`src/tokenizer_bpe.py`)
- UTF-8 data reading helpers (`src/data_reading.py`)
- Token stream dataset for next-token prediction (`src/dataset_gpt.py`)
- Train/validation token splitting and loaders (`src/data_split.py`, `src/make_splits_and_loaders.py`)
- Token and absolute positional embeddings (`src/embeddings.py`)
- Causal multi-head self-attention (`src/attention.py`)
- Manual GPT-style `LayerNorm` (`src/layernorm.py`)
- Manual GELU tanh approximation (`src/gelu.py`)
- GPT-style feed-forward network (`src/feedforward.py`)
- Transformer block (pre-norm + residuals) (`src/transformer_block.py`)
- GPT model assembly (`src/gpt_model.py`)
- Loss helpers for next-token LM (`src/next_token_loss.py`)
- Training loop + evaluation helpers (`src/training.py`)
- Text generation utility (greedy, temperature, top-k) (`src/generate.py`)
- Plot utility for train/val loss curves (`src/plot_training_curves.py`)
- Multiple smoke tests under `src/smoke_test_*.py`

## Supported Hardware and Runtimes

- Python 3.10+
- PyTorch runtime on:
  - Apple Silicon (`mps`)
  - NVIDIA GPU (`cuda`)
  - CPU fallback
- Scripts are designed to run from project root with module mode:

```bash
python -m src.some_module
```

## Project Structure

```text
llm-project/
├── artifacts/
│   ├── pretrain_small.pt
│   └── tokenizer.json
├── configs/
│   └── pretrain_small.yaml
├── data/
│   ├── the-verdict.txt
│   ├── the-verdict.tokens.pt
│   └── the-verdict.tokens.meta.json
├── notebook/
│   └── ch04.ipynb
├── results/
├── src/
│   ├── __init__.py
│   ├── attention.py
│   ├── config.py
│   ├── data_reading.py
│   ├── data_split.py
│   ├── dataset_gpt.py
│   ├── demo_generate.py
│   ├── embeddings.py
│   ├── feedforward.py
│   ├── gelu.py
│   ├── generate.py
│   ├── gpt_model.py
│   ├── layernorm.py
│   ├── make_dataloader.py
│   ├── make_splits_and_loaders.py
│   ├── next_token_loss.py
│   ├── plot_training_curves.py
│   ├── run_train_verdict.py
│   ├── smoke_test_*.py
│   ├── test_attention_shapes.py
│   ├── tokenize_verdict.py
│   ├── tokenizer_bpe.py
│   ├── training.py
│   └── transformer_block.py
├── requirements.txt
└── README.md
```

## Installation Instructions

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch tiktoken matplotlib
```

## Quick Sanity Checks

```bash
python -m src.smoke_test_tokenizer
python -m src.smoke_test_gpt_model
```

## Tokenization Instructions

```bash
python -m src.tokenize_verdict
```

This generates:

- `data/the-verdict.tokens.pt`
- `data/the-verdict.tokens.meta.json`

## Training Instructions

```bash
python -m src.run_train_verdict
```

Example output:

```text
Ep 1 (step 000000): train_loss=10.9710, val_loss=10.9746, tokens_seen=1024
Ep 1 (step 000005): train_loss=10.7240, val_loss=10.7814, tokens_seen=6144
Ep 2 (step 000010): train_loss=10.4267, val_loss=10.5023, tokens_seen=11264
Ep 2 (step 000015): train_loss=10.0206, val_loss=10.1299, tokens_seen=16384
Final train loss: 10.0206
Final val loss: 10.1299
Tracked eval points: 4
```

## Embedding the Loss Curve

After training, call the plotting utility with the tracked arrays returned by `train_model_simple`:

```python
from src.plot_training_curves import plot_loss_curves
plot_loss_curves(train_losses, val_losses, tokens_seen)
```

![Loss Curve](results/loss_curve.png)

## Text Generation

```python
import torch
from src.gpt_model import GPTModel
from src.generate import generate
from src.tokenizer_bpe import BPETokenizer
from src import config

cfg = dict(config.MODEL_CFG)
model = GPTModel(cfg)
tokenizer = BPETokenizer("gpt2")

prompt = "Every effort moves you"
idx = tokenizer.encode_to_tensor(prompt).unsqueeze(0)

out = generate(
    model=model,
    idx=idx,
    max_new_tokens=40,
    context_size=cfg["context_length"],
    temperature=0.8,
    top_k=40,
)

print(tokenizer.decode(out[0].tolist()))
```

You can also run the interactive script:

```bash
python -m src.demo_generate
```

## Configuration

Primary configuration lives in `src/config.py`.

Key values used across scripts include:

- `MODEL_CFG` (or compatible model dict)
- `TRAIN_CONTEXT_LENGTH` (fallback to `CONTEXT_LENGTH`)
- `TRAIN_STRIDE` (fallback to `STRIDE`)
- `BATCH_SIZE`
- `LEARNING_RATE`

If `MODEL_CFG` is not present, some scripts use minimal fallbacks based on available constants and token metadata.

## Notes

- Run commands from the project root (`llm-project/`) so `src` module imports resolve correctly.
- If token files are missing, run tokenization first:

```bash
python -m src.tokenize_verdict
```

- If `tiktoken` is missing, install it:

```bash
pip install tiktoken
```

- If `matplotlib` is missing (for curve plotting), install it:

```bash
pip install matplotlib
```

## Research Context

This codebase follows a pedagogical, component-by-component GPT construction approach inspired by modern LLM training walkthroughs (notably the Raschka-style progression): tokenizer → dataset → embeddings → attention/MLP blocks → full decoder model → loss/training/evaluation → generation. It is intended for learning, controlled experimentation, and reproducible debugging of core language-model building blocks rather than large-scale production pretraining.
