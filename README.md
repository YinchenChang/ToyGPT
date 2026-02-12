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

## Modernization (2026)

The codebase has been modernized from the original 2023 tutorial code to PyTorch 2.x industry standards:

**Bug Fix:**
- Fixed critical attention scaling bug — original used `1/sqrt(n_embd)` instead of `1/sqrt(head_size)`, over-dampening attention by ~2.83x

**PyTorch 2.x Features:**
- Flash Attention via `F.scaled_dot_product_attention` (replaces manual attention computation)
- `torch.compile()` for automatic kernel fusion (GPU only)
- Automatic Mixed Precision (AMP) with bfloat16/float16 auto-selection (GPU only)
- TF32 precision for Ampere+ GPUs

**Training Improvements:**
- GELU activation (GPT-2/3 standard, replaces ReLU)
- Cosine learning rate schedule with linear warmup
- Proper weight decay (excluded from biases and LayerNorm parameters)
- Best model checkpointing by validation loss

**Code Quality:**
- `GPTConfig` dataclass replaces global variables
- Accurate class names (`GPTLanguageModel`, `FeedForward`, `CausalSelfAttention`)
- All model classes receive config via constructor (no global dependencies)
- `if __name__ == '__main__':` guard for importability

See the `*_MODERNIZATION_REPORT.md` files for detailed explanations of each change.

## Requirements

- Python 3.10+
- PyTorch 2.0+

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
2. Train the model for 5000 iterations with cosine LR schedule
3. Save the best model checkpoint (`best_model.pt`)
4. Generate 2000 characters of Shakespeare-like text

### Variant Differences

| Feature | `Andrej_GPT_Test.py` | `..._CPU.py` | `..._GPU.py` |
|---------|---------------------|-------------|-------------|
| Device | Auto-detect | Forced CPU | Auto-detect (GPU preferred) |
| torch.compile | Yes (GPU) | Disabled | Yes (GPU) |
| AMP | Yes (GPU) | Disabled | Yes (GPU) |
| TF32 | Yes | N/A | Yes |
| Checkpoint | `best_model.pt` | `best_model_cpu.pt` | `best_model_gpu.pt` |

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| Layers / Heads | 8 / 8 |
| Embedding dim | 128 |
| Context length | 128 |
| Batch size | 32 |
| Training iterations | 5000 |
| Learning rate | 1e-3 → 1e-4 (cosine schedule) |
| Warmup steps | 100 |
| Weight decay | 0.1 |
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

## License

This project is licensed under the [MIT License](LICENSE).
