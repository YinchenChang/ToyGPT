# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Character-level transformer language model (ToyGPT) trained on Shakespeare text, based on Andrej Karpathy's "Let's build GPT" tutorial. Implemented entirely in Jupyter notebooks using PyTorch.

## Running

```bash
jupyter notebook Andrej_GPT_Test.ipynb              # Auto-detects GPU/CPU
jupyter notebook Andrej_ToyGPT_Test_Merge_CPU.ipynb  # CPU-only
jupyter notebook Andrej_ToyGPT_Test_Merge_GPU.ipynb  # GPU-optimized
```

Requires: PyTorch, Jupyter, and `input.txt` (Shakespeare corpus, ~1.1M chars) in the same directory.

## Notebook Variants

- **Andrej_GPT_Test.ipynb** — Primary notebook with CUDA auto-detection
- **Andrej_ToyGPT_Test_Merge_CPU.ipynb** — Forces `device='cpu'`
- **Andrej_ToyGPT_Test_Merge_GPU.ipynb** — GPU-targeted version

All three implement the same model architecture; they differ only in device targeting.

## Architecture

Pre-norm transformer decoder stack for next-character prediction:

```
Character indices → Token Embedding + Positional Embedding
  → 8x Transformer Blocks (each: LayerNorm → MultiHead Attention → LayerNorm → FFN, with residual connections)
  → LayerNorm → Linear head → logits (vocab_size=65)
```

Key classes: `Head` (single attention head), `MultiHeadAttention`, `FeedFoward`, `Block`, `BigramLanguageModel`.

## Default Hyperparameters

| Parameter | Value |
|-----------|-------|
| n_layer / n_head | 8 / 8 |
| n_embd | 128 |
| block_size (context length) | 128 |
| batch_size | 32 |
| max_iters | 5000 |
| learning_rate | 1e-3 (AdamW) |
| dropout | 0.0 |

Model size: ~1.6M parameters. Training converges to ~1.22 train loss / ~1.54 val loss.
