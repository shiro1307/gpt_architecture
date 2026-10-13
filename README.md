# Transformer GPT

A GPT-style language model built from scratch in PyTorch, trained on a text corpus using character-level tokenization.

## Files

- `model.py` — Transformer architecture (multi-head attention, feed-forward, positional encoding)
- `tokenizer.py` — Character-level tokenizer, train/val split, batch sampling
- `train.py` — Training loop with loss estimation and checkpoint saving
- `generate.py` — Load a checkpoint and generate text from a prompt

## Usage

**Train:**
```bash
python train.py
```

**Generate:**
```bash
python generate.py
```

Edit the `start_text` variable in `generate.py` to change the prompt.

## Requirements

- Python 3.x
- PyTorch
