# Modernization Report: Andrej_ToyGPT_Test_Merge_GPU.py

**Original:** 2023/02/25 (GPU-optimized variant, based on Andrej Karpathy's tutorial)
**Modernized:** 2026/02 (PyTorch 2.x standards)
**File:** `Andrej_ToyGPT_Test_Merge_GPU.py` (229 lines → 305 lines)

> For detailed explanations of each change with code examples, math, and references, see
> [Andrej_GPT_Test_MODERNIZATION_REPORT.md](Andrej_GPT_Test_MODERNIZATION_REPORT.md).
> This report covers what is **specific to the GPU variant**.

---

## Variant-Specific Differences

This file originally used `device = 'cuda' if torch.cuda.is_available() else 'cpu'` (line 14)
with a commented-out CPU fallback. The modernized version enables all GPU optimizations:

```python
# System — GPU-optimized: all acceleration features enabled
device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
compile_model: bool = True   # torch.compile for kernel fusion
use_amp: bool = True          # automatic mixed precision
```

### All GPU Optimizations Enabled

| Feature | Expected Speedup | Details |
|---------|-----------------|---------|
| `torch.compile()` | 10-30% | Automatic kernel fusion, reduced memory reads/writes |
| AMP (bfloat16/float16) | ~2x | Half-precision matmuls on Tensor Cores, halved memory |
| TF32 precision | 2-3x matmul speedup | 19-bit internal precision for float32 ops on Ampere+ |
| Flash Attention | 10-30%+ | Fused attention kernel, O(T) memory instead of O(T²) |

### Combined Speedup

With all features active on a modern GPU (RTX 30xx/40xx, A100, H100), expect **3-5x total speedup**
compared to the original code running on the same GPU.

### Training Loop (Full GPU Pipeline)

```python
# GPU: AMP-wrapped forward/backward with loss scaling
with torch.amp.autocast(device_type=config.device, dtype=amp_dtype, enabled=use_amp):
    logits, loss = model(xb, yb)
optimizer.zero_grad(set_to_none=True)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### bfloat16 vs float16 Auto-Selection

The script auto-detects the best half-precision format:

```python
amp_dtype = torch.float16
if use_amp and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
    amp_dtype = torch.bfloat16  # preferred on Ampere+
```

- **Ampere+ GPUs** (RTX 30xx, A100, etc.): Uses bfloat16 — no loss scaling needed
- **Older GPUs** (RTX 20xx, V100): Uses float16 with `GradScaler` for loss scaling

### Graceful CPU Fallback

If no GPU is available, the script automatically falls back to CPU mode:
- `device` defaults to `'cpu'`
- `torch.compile` is skipped (gated behind `config.device == 'cuda'`)
- AMP is skipped (gated behind `config.device == 'cuda'`)
- TF32 settings are set but have no effect on CPU

No code changes needed — it just works.

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
| 7 | `torch.compile()` | **Enabled** — Section 7 |
| 8 | AMP mixed precision | **Enabled** — Section 8 |
| 9 | TF32 precision | **Enabled** — Section 9 |
| 10 | Cosine LR schedule with warmup | Yes — Section 10 |
| 11 | Proper weight decay configuration | Yes — Section 11 |
| 12 | Model checkpointing (saves to `best_model_gpu.pt`) | Yes — Section 12 |
| 13 | Fix variable shadowing (`iter` → `step`, `data` → `split_data`) | Yes — Section 13 |
| 14 | `if __name__ == '__main__':` guard | Yes — Section 14 |
| 15 | Minor cleanups (alias removal, shape comments, print formatting) | Yes — Section 15 |

---

## Comparison: GPU vs CPU vs Auto-Detect Variants

| Feature | `Andrej_GPT_Test.py` | `..._Merge_GPU.py` | `..._Merge_CPU.py` |
|---------|---------------------|--------------------|--------------------|
| Device | Auto-detect | Auto-detect (GPU preferred) | Forced CPU |
| torch.compile | Yes (GPU only) | Yes (GPU only) | Disabled |
| AMP | Yes (GPU only) | Yes (GPU only) | Disabled |
| TF32 | Yes | Yes | Not set |
| Checkpoint file | `best_model.pt` | `best_model_gpu.pt` | `best_model_cpu.pt` |
| Training loop | AMP-wrapped | AMP-wrapped | Plain |

> **Note:** After modernization, the auto-detect variant (`Andrej_GPT_Test.py`) and the GPU variant
> are functionally identical — both auto-detect the device and enable GPU features when available.
> The CPU variant explicitly disables GPU features for guaranteed CPU-only execution.
