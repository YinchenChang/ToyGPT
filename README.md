# ToyGPT

A character-level transformer language model trained on Shakespeare text, built from scratch following [Andrej Karpathy's "Let's build GPT from scratch"](https://www.youtube.com/watch?v=kCc8FmEb1nY) video tutorial.

The model learns to generate Shakespeare-like text by predicting the next character, one character at a time.

## How It Works

The model is a decoder-only transformer (~1.6M parameters) that operates at the character level:

```
Input characters → Token + Positional Embeddings
  → 8 Transformer Blocks (LayerNorm → Multi-Head Attention → LayerNorm → Feed-Forward)
  → LayerNorm → Linear → next-character prediction
```

Each transformer block uses pre-norm residual connections, 8 attention heads, and a feed-forward network that expands to 4x the embedding dimension.

## Requirements

- Python 3
- PyTorch

```bash
pip install torch
```

## Usage

Pick the script that matches your hardware:

```bash
# Auto-detect GPU/CPU (recommended)
python Andrej_GPT_Test.py

# Force CPU only
python Andrej_ToyGPT_Test_Merge_CPU.py

# GPU-optimized (uses CUDA if available)
python Andrej_ToyGPT_Test_Merge_GPU.py
```

The script will:
1. Load `input.txt` (Shakespeare corpus, ~1.1M characters, 65-character vocabulary)
2. Train the model for 5000 iterations (~1.22 train loss / ~1.54 val loss)
3. Generate 2000 characters of Shakespeare-like text

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| Layers / Heads | 8 / 8 |
| Embedding dim | 128 |
| Context length | 128 |
| Batch size | 32 |
| Training iterations | 5000 |
| Learning rate | 1e-3 (AdamW) |
| Dropout | 0.0 |

## Sample Output

After training, the model generates text like:

```
YORK:
I brince, and in the servace of Baptista Signion
My hand of bard that us he had by the shoar,
Away, with fall to zoin heavens, to time queen!
```

Not perfect English, but it captures Shakespeare's style — character names, iambic rhythm, and dramatic flair.

## Acknowledgments

Based on Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT) and his ["Let's build GPT"](https://www.youtube.com/watch?v=kCc8FmEb1nY) tutorial.
