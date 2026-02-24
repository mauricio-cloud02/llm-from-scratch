# LLM Project: GPT-Style Language Modeling Pipeline

This repository contains a from-scratch implementation of a GPT-style decoder-only Transformer in PyTorch, built following Sebastian Raschka’s *Build a Large Language Model from Scratch*.

The goal of this project is to understand the internal mechanics of Transformer-based language models, including training dynamics, architectural components, and sampling behavior, without relying on high-level pretrained APIs. covering tokenization, dataset construction, Transformer components, training/evaluation utilities, and text generation. The code is intentionally minimal and readable, with smoke tests for each stage so you can validate behavior incrementally before scaling experiments.

## Implemented Components

```plain text
src/
  attention.py
  transformer_block.py
  gpt_model.py
  next_token_loss.py
  training.py
  generate.py
  ...
configs/
artifacts/
results/
```

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

Create and activate a virtual environment, then install dependencies:
```bash
pip install -r requirements.txt
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

## Training Instructions (Toy Configuration)

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

Generation supports:

Greedy decoding

Temperature scaling

Top-k filtering

Multinomial sampling

Example usage inside Python:


``` python
from src.generate import generate
```

## Configuration

Model and training hyperparameters are stored in:

```bash
src/config.py

configs/
```

The repository distinguishes between:

A small toy model (fast local training)

A GPT-2–compatible architecture configuration

## Notes

Training is performed on a small demonstration dataset.

The focus of this implementation is architectural clarity and reproducibility rather than large-scale performance.

The model architecture is compatible with GPT-2 configurations.

## Research Context

This codebase follows a pedagogical, component-by-component GPT construction approach: tokenizer → dataset → embeddings → attention/MLP blocks → full decoder model → loss/training/evaluation → generation. It is intended for learning, controlled experimentation, and reproducible debugging of core language-model building blocks rather than large-scale production pretraining. 

The goal was to examine the internal structure and training behavior of large language models directly, including attention mechanisms, loss dynamics, sampling strategies, and architectural trade-offs. By reconstructing the model pipeline end-to-end, this project provides a controlled environment for analyzing how design choices influence generation behavior and training stability.
