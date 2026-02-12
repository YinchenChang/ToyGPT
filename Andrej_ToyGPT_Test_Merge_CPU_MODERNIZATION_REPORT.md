# Modernization Report: Andrej_ToyGPT_Test_Merge_CPU.py

**Original:** 2023/02/25 (CPU-only variant, based on Andrej Karpathy's tutorial)
**Modernized:** 2026/02 (PyTorch 2.x standards)
**File:** `Andrej_ToyGPT_Test_Merge_CPU.py` (229 lines → 287 lines)

> For detailed explanations of each change with code examples, math, and references, see
> [Andrej_GPT_Test_MODERNIZATION_REPORT.md](Andrej_GPT_Test_MODERNIZATION_REPORT.md).
> This report covers what is **specific to the CPU variant**.

---

## Variant-Specific Differences

This file originally forced `device = 'cpu'` (line 15) with the CUDA auto-detection commented out.
The modernized version preserves this CPU-only intent through the `GPTConfig` defaults:

```python
# System — CPU-only: GPU features disabled
device: str = 'cpu'
compile_model: bool = False
use_amp: bool = False
```

### What's Disabled (and Why)

| Feature | Status | Reason |
|---------|--------|--------|
| `torch.compile()` | Disabled | Can be slower on CPU due to compilation overhead; limited Windows support |
| AMP (mixed precision) | Disabled | Tensor Cores are GPU-only; no speedup on CPU |
| TF32 precision | Not set | Only relevant for Ampere+ CUDA GPUs |
| `GradScaler` | Not used | Only needed for float16 AMP training |

### What's Still Applied (Benefits CPU Too)

| Feature | Benefit on CPU |
|---------|----------------|
| Flash Attention (`F.scaled_dot_product_attention`) | Correct scaling fix; uses math backend on CPU (no Flash kernel, but still correct and cleaner) |
| **Attention scaling bug fix** | **Critical** — same bug existed here, same 2.83x over-dampening |
| GELU activation | Better gradient flow regardless of device |
| `GPTConfig` dataclass | Clean configuration |
| Cosine LR with warmup | Better convergence |
| Proper weight decay | Exclude biases/LayerNorm |
| Model checkpointing | Saves to `best_model_cpu.pt` |
| `if __name__ == '__main__':` guard | Importable as module |
| Class/variable renames | Code clarity |

### Training Loop Difference

The CPU training loop is simpler since AMP is disabled — no `autocast`, no `GradScaler`:

```python
# CPU: plain forward/backward
xb, yb = get_batch('train', config, train_data, val_data)
logits, loss = model(xb, yb)
optimizer.zero_grad(set_to_none=True)
loss.backward()
optimizer.step()
```

vs. the GPU variant which wraps in `autocast` and uses `scaler`.

---

## All Changes Applied

| # | Change | Same as main report? |
|---|--------|---------------------|
| 1 | Fix attention scaling bug (1/sqrt(n_embd) → 1/sqrt(head_size)) | Yes — Section 1 |
| 2 | Flash Attention (`F.scaled_dot_product_attention`) | Yes — Section 2 |
| 3 | GELU activation (ReLU → GELU) | Yes — Section 3 |
| 4 | `GPTConfig` dataclass (globals → config) | Yes — Section 4 |
| 5 | Class renames (`BigramLanguageModel` → `GPTLanguageModel`, `FeedFoward` → `FeedForward`) | Yes — Section 5 |
| 6 | Remove global variable dependencies | Yes — Section 6 |
| 7 | `torch.compile()` | **Disabled** (CPU-only) |
| 8 | AMP mixed precision | **Disabled** (CPU-only) |
| 9 | TF32 precision | **Not applicable** (CPU-only) |
| 10 | Cosine LR schedule with warmup | Yes — Section 10 |
| 11 | Proper weight decay configuration | Yes — Section 11 |
| 12 | Model checkpointing (saves to `best_model_cpu.pt`) | Yes — Section 12 |
| 13 | Fix variable shadowing (`iter` → `step`, `data` → `split_data`) | Yes — Section 13 |
| 14 | `if __name__ == '__main__':` guard | Yes — Section 14 |
| 15 | Minor cleanups (alias removal, shape comments, print formatting) | Yes — Section 15 |

---

## Note on Performance

CPU training of this 1.6M parameter model with 5000 iterations will be significantly slower than GPU.
Consider reducing hyperparameters for CPU experimentation:

```python
config = GPTConfig(
    n_layer=4,       # 4 instead of 8
    n_head=4,        # 4 instead of 8
    n_embd=64,       # 64 instead of 128
    max_iters=1000,  # 1000 instead of 5000
)
```
