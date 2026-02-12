# Modernization Report: Andrej_GPT_Test.py

**Original:** 2023/02/25 (based on Andrej Karpathy's "Let's build GPT" tutorial)
**Modernized:** 2026/02 (PyTorch 2.x industry standards)
**File:** `Andrej_GPT_Test.py` (228 lines → 305 lines)

---

## Table of Contents

1. [Critical Bug Fix: Attention Scaling](#1-critical-bug-fix-attention-scaling)
2. [Flash Attention: Replacing Manual Attention](#2-flash-attention-replacing-manual-attention)
3. [GELU Activation: Replacing ReLU](#3-gelu-activation-replacing-relu)
4. [Configuration Dataclass: Replacing Global Variables](#4-configuration-dataclass-replacing-global-variables)
5. [Class Renaming: Accurate Names](#5-class-renaming-accurate-names)
6. [Removing Global Variable Dependencies](#6-removing-global-variable-dependencies)
7. [torch.compile: Automatic Kernel Fusion](#7-torchcompile-automatic-kernel-fusion)
8. [Automatic Mixed Precision (AMP)](#8-automatic-mixed-precision-amp)
9. [TF32 Precision](#9-tf32-precision)
10. [Cosine Learning Rate Schedule with Warmup](#10-cosine-learning-rate-schedule-with-warmup)
11. [Proper Weight Decay Configuration](#11-proper-weight-decay-configuration)
12. [Model Checkpointing](#12-model-checkpointing)
13. [Variable Shadowing Fixes](#13-variable-shadowing-fixes)
14. [if __name__ == '__main__': Guard](#14-if-__name__--__main__-guard)
15. [Minor Cleanups](#15-minor-cleanups)

---

## 1. Critical Bug Fix: Attention Scaling

### What Changed

**Before (line 92):**
```python
def forward(self, x):
    B, T, C = x.shape  # C is n_embd (128)
    k = self.key(x)     # (B, T, head_size=16)
    q = self.query(x)   # (B, T, head_size=16)
    wei = q @ k.transpose(-2, -1) * C ** -0.5  # BUG: C = n_embd = 128
```

**After (handled internally by `F.scaled_dot_product_attention`):**
```python
# Scales by 1/sqrt(head_size) automatically — which is 1/sqrt(16) = 0.25
out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
```

### Why This Matters

The original "Attention is All You Need" paper (Vaswani et al., 2017) defines scaled dot-product attention as:

```
Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
```

Where `d_k` is the **dimension of the keys** (i.e., `head_size`), not the full embedding dimension.

In the original code:
- `x` enters the `Head` with shape `(B, T, n_embd)` where `n_embd = 128`
- `B, T, C = x.shape` unpacks `C = n_embd = 128`
- After `self.key(x)`, the key has shape `(B, T, head_size)` where `head_size = n_embd // n_head = 128 // 8 = 16`
- The scaling should use `head_size = 16`, not `C = 128`

**The numerical impact:**
- Original scaling: `1/sqrt(128) ≈ 0.088`
- Correct scaling: `1/sqrt(16) = 0.25`
- The original **over-dampened attention scores by a factor of sqrt(8) ≈ 2.83x**

This makes the pre-softmax attention scores smaller than they should be, pushing the softmax output closer to a uniform distribution. The model can't attend as sharply to specific positions, effectively blurring its attention. The model can partially compensate during training (by learning larger Q/K weights), but it's working against an unnecessary handicap.

### How to Spot This Bug

The confusing comments in the original code were actually clues:
```python
k = self.key(x) # (B, T, C), why C but not head_size???
```

The comment was right to question it — `k` is NOT `(B, T, C)`, it's `(B, T, head_size)`. The `nn.Linear(n_embd, head_size)` projects from `n_embd` down to `head_size`. The variable `C` from `x.shape` is `n_embd`, but after projection, the last dimension is `head_size`.

---

## 2. Flash Attention: Replacing Manual Attention

### What Changed

**Before (lines 76-114) — Two classes, ~40 lines:**
```python
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * C ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # Sequential!
        out = self.dropout(self.proj(out))
        return out
```

**After (lines 43-87) — One class, ~45 lines:**
```python
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = config.dropout
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        head_size = C // self.n_head
        q = q.view(B, T, self.n_head, head_size).transpose(1, 2)
        k = k.view(B, T, self.n_head, head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_size).transpose(1, 2)
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None,
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=True,
        )
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.resid_dropout(self.c_proj(out))
        return out
```

### Why This Matters

**Three separate improvements are happening here:**

#### A. Batched Q/K/V Projection

The original creates 8 separate `Head` objects, each with its own `key`, `query`, `value` linear layers. That's 24 separate small matrix multiplications.

The new code uses a single linear layer `c_attn` that projects to `3 * n_embd` and then splits into Q, K, V. This is one large matrix multiplication instead of 24 small ones. GPUs are far more efficient with fewer, larger operations.

#### B. Parallel Head Computation

The original runs heads sequentially via a Python list comprehension:
```python
out = torch.cat([h(x) for h in self.heads], dim=-1)  # Python loop!
```

This is a Python-level for-loop calling 8 separate forward passes. The new code reshapes Q, K, V to `(B, n_head, T, head_size)` and computes all 8 heads simultaneously in a single tensor operation. No Python loop.

#### C. F.scaled_dot_product_attention (Flash Attention)

Introduced in PyTorch 2.0, this function replaces the manual 4-step attention:
```python
# OLD: 4 steps, materializes full T×T attention matrix
wei = q @ k.transpose(-2, -1) * scale    # Step 1: compute scores
wei = wei.masked_fill(mask == 0, -inf)    # Step 2: apply causal mask
wei = F.softmax(wei, dim=-1)              # Step 3: softmax
out = wei @ v                             # Step 4: weighted sum
```

With `F.scaled_dot_product_attention`, PyTorch auto-selects the best backend:
- **Flash Attention** (Dao et al., 2022): Fuses all 4 steps into a single GPU kernel. Never materializes the full T×T attention matrix in GPU memory. Memory usage goes from O(T²) to O(T). For `T=128` this is modest, but for longer sequences it's transformative.
- **Memory-Efficient Attention** (xFormers): Similar benefits, broader hardware support.
- **Math fallback**: Standard implementation for hardware that doesn't support the above.

The `is_causal=True` flag tells PyTorch to apply the lower-triangular causal mask internally, eliminating the need for `register_buffer('tril', ...)`.

### Performance Impact

For this small model (T=128), expect ~10-30% speedup from the batched operations alone. For longer sequences, Flash Attention provides dramatically larger improvements.

---

## 3. GELU Activation: Replacing ReLU

### What Changed

**Before (line 124):**
```python
nn.ReLU(),
```

**After (line 97):**
```python
nn.GELU(),
```

### Why This Matters

**ReLU (Rectified Linear Unit):**
```
ReLU(x) = max(0, x)
```
- Hard cutoff at zero: all negative inputs produce exactly zero gradient
- This "dying ReLU" problem means some neurons can permanently stop learning

**GELU (Gaussian Error Linear Unit):**
```
GELU(x) = x * Φ(x)    where Φ is the standard Gaussian CDF
```
- Smooth, non-monotonic function that gradually scales inputs
- Small negative values still get a small non-zero output
- Provides a form of stochastic regularization

GELU was introduced in 2016 (Hendrycks & Gimpel) and adopted by:
- **GPT-2** (2019) — OpenAI
- **BERT** (2018) — Google
- **GPT-3** (2020) — OpenAI
- Essentially all modern transformers since ~2018

ReLU is still fine for CNNs and simpler architectures, but GELU is the industry standard for transformers. The improvement is typically small but consistent (a few percent better validation loss).

---

## 4. Configuration Dataclass: Replacing Global Variables

### What Changed

**Before (lines 8-19):**
```python
batch_size = 32
block_size = 128
max_iters = 5000
eval_interval = 100
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 128
n_head = 8
n_layer = 8
dropout = 0.0
```

**After (lines 17-39):**
```python
@dataclass
class GPTConfig:
    batch_size: int = 32
    block_size: int = 128
    max_iters: int = 5000
    eval_interval: int = 100
    eval_iters: int = 200
    learning_rate: float = 1e-3
    min_lr: float = 1e-4
    warmup_iters: int = 100
    weight_decay: float = 0.1
    n_embd: int = 128
    n_head: int = 8
    n_layer: int = 8
    dropout: float = 0.0
    vocab_size: int = 0
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    compile_model: bool = True
    use_amp: bool = True
```

### Why This Matters

**Problems with global variables:**

1. **Hidden dependencies:** When you read `class Head`, you can't tell from its signature what configuration it needs. You have to read the entire method body to discover it references `n_embd`, `block_size`, and `dropout` from global scope.

2. **Can't run two configs simultaneously:** Want to compare a 4-layer vs 8-layer model? With globals, you can't — there's only one `n_layer`. With a config object, you just create two: `config_small = GPTConfig(n_layer=4)` and `config_large = GPTConfig(n_layer=8)`.

3. **Testing is difficult:** Unit testing a class that reads globals requires modifying global state, which is error-prone and can leak between tests.

4. **No IDE support:** Your IDE can't autocomplete or type-check `n_embd` when it's just a bare global. With `config.n_embd`, you get autocomplete, type hints, and "go to definition."

**Why `@dataclass` specifically:**

Python's `@dataclass` decorator (from `dataclasses`, standard library since Python 3.7) auto-generates `__init__`, `__repr__`, and `__eq__` from the field definitions. You get a clean, typed configuration object with zero boilerplate. It's simpler than writing a custom class, more structured than a dictionary, and requires no external dependencies (unlike libraries like `pydantic` or `attrs`).

---

## 5. Class Renaming: Accurate Names

### What Changed

| Before | After | Why |
|--------|-------|-----|
| `BigramLanguageModel` | `GPTLanguageModel` | This is a full 8-layer transformer decoder, not a bigram model |
| `FeedFoward` | `FeedForward` | Typo fix (missing 'r') |
| `Head` + `MultiHeadAttention` | `CausalSelfAttention` | Standard name used in GPT-2, nanoGPT, and the broader community |

### Why This Matters

**BigramLanguageModel** is misleading because a bigram model only looks at the previous token to predict the next one. This model uses 128 tokens of context through multi-head self-attention across 8 transformer layers — it's a full GPT-style decoder. The name was a leftover from the tutorial, where Karpathy starts with an actual bigram model and incrementally adds attention, but never renames the class.

Good naming is not just cosmetic — when you return to this code in 6 months, or when someone else reads it, `GPTLanguageModel` immediately communicates what it is.

---

## 6. Removing Global Variable Dependencies

### What Changed

**Before — classes read from global scope:**
```python
class Head(nn.Module):
    def __init__(self, head_size):
        self.key = nn.Linear(n_embd, head_size, bias=False)  # n_embd is global
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))  # block_size is global
        self.dropout = nn.Dropout(dropout)  # dropout is global
```

**After — classes receive a config object:**
```python
class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.attn_dropout = config.dropout
```

### Why This Matters

This is closely related to Section 4 (Configuration Dataclass), but worth emphasizing separately because it affects every model class (`CausalSelfAttention`, `FeedForward`, `Block`, `GPTLanguageModel`) and both utility functions (`get_batch`, `estimate_loss`).

**Dependency injection** (passing `config` as a parameter) is a fundamental software engineering principle. Each class explicitly declares what it needs through its constructor signature. This makes the code:
- **Testable:** You can create a small config for fast unit tests
- **Reusable:** The classes can be imported and used in other scripts
- **Readable:** You can understand a class by reading just its constructor

Another subtle fix: the original `GPTLanguageModel.forward` used the global `device` for position embeddings:
```python
# Before: uses global 'device'
pos_emb = self.position_embedding_table(torch.arange(T, device=device))

# After: uses the device of the input tensor itself
pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
```

Using `idx.device` is more robust — the model automatically works on whatever device the input is on, without needing to know about a global `device` variable.

---

## 7. torch.compile: Automatic Kernel Fusion

### What Changed

**Before:** Not present.

**After (lines 253-256):**
```python
if config.compile_model and config.device == 'cuda':
    print("Compiling model with torch.compile()...")
    model = torch.compile(model)
```

### Why This Matters

`torch.compile()` was the headline feature of PyTorch 2.0 (March 2023). It analyzes your model's computation graph and automatically:

1. **Fuses kernels:** Instead of launching separate GPU kernels for LayerNorm → Linear → GELU → Linear → Dropout, it fuses them into fewer, larger kernels. Each kernel launch has overhead (~5-10 microseconds), so reducing launches directly speeds up training.

2. **Eliminates redundant memory reads/writes:** Without compilation, each operation reads its input from GPU memory and writes its output back. Fused kernels keep intermediate values in fast GPU registers.

3. **Applies algebraic simplifications:** The compiler can rearrange mathematically equivalent operations for better performance.

**Typical speedup: 10-30%** on modern GPUs, with zero code changes to the model itself.

**Why the guards:**
- `config.device == 'cuda'`: On CPU, `torch.compile` can sometimes be slower due to compilation overhead outweighing benefits for small models.
- `config.compile_model`: A flag to easily disable it, since `torch.compile` on Windows has historically had limited support and may cause errors in some environments.

---

## 8. Automatic Mixed Precision (AMP)

### What Changed

**Before:** All computation in float32.

**After (lines 258-263, 293-299):**
```python
# Setup
use_amp = config.use_amp and config.device == 'cuda'
amp_dtype = torch.float16
if use_amp and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
    amp_dtype = torch.bfloat16
scaler = torch.amp.GradScaler(enabled=use_amp and amp_dtype == torch.float16)

# Training step
with torch.amp.autocast(device_type=config.device, dtype=amp_dtype, enabled=use_amp):
    logits, loss = model(xb, yb)
optimizer.zero_grad(set_to_none=True)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### Why This Matters

**The problem:** Float32 uses 32 bits per number. Modern GPUs have special hardware (Tensor Cores) that can compute with 16-bit numbers at 2-8x the speed. But naively converting everything to float16 can cause numerical issues (underflow, overflow).

**The solution — AMP:** `torch.amp.autocast` automatically chooses which operations run in float16/bfloat16 and which stay in float32:
- **Matrix multiplications** (the bottleneck): Run in float16 — Tensor Cores handle these
- **Reductions** (softmax, layer norm, loss): Stay in float32 — these are sensitive to precision
- **Everything else:** Automatically decided by PyTorch

**float16 vs bfloat16:**

| | float16 | bfloat16 |
|--|---------|----------|
| Sign bits | 1 | 1 |
| Exponent bits | 5 | 8 |
| Mantissa bits | 10 | 7 |
| Range | ±65,504 | ±3.4 × 10³⁸ (same as float32) |
| Precision | Higher | Lower |
| Loss scaling needed? | Yes | No |
| GPU support | All modern GPUs | Ampere+ (RTX 30xx, A100, etc.) |

**bfloat16 is preferred** when available because it has the same exponent range as float32, meaning gradients won't underflow or overflow. This eliminates the need for `GradScaler` (loss scaling). That's why the code only enables `GradScaler` when using float16:

```python
scaler = torch.amp.GradScaler(enabled=use_amp and amp_dtype == torch.float16)
```

**`GradScaler` explained:** With float16, very small gradients can underflow to zero. `GradScaler` multiplies the loss by a large number before `.backward()`, making gradients larger. It then divides the gradients back down before the optimizer step. If gradients overflow (become inf/nan), it skips the optimizer step and reduces the scale factor. This all happens automatically.

**Typical speedup: ~2x** on GPUs with Tensor Cores, with roughly halved memory usage.

---

## 9. TF32 Precision

### What Changed

**Before:** Not present.

**After (lines 221-223):**
```python
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```

### Why This Matters

TF32 (TensorFloat-32) is an NVIDIA Ampere+ feature that uses 19-bit precision (8-bit exponent + 10-bit mantissa + 1-bit sign) for matrix multiplications, stored in 32-bit registers.

**Without TF32:** float32 matmuls use all 23 mantissa bits → full precision but slower.
**With TF32:** float32 matmuls internally use 10 mantissa bits → same speed as float16 matmuls, but with float32 dynamic range.

For deep learning, the reduced mantissa precision has no measurable impact on model quality. It's essentially a "free" 2-3x speedup for float32 operations on Ampere+ GPUs.

PyTorch disabled TF32 by default for reproducibility — these two lines explicitly opt in.

---

## 10. Cosine Learning Rate Schedule with Warmup

### What Changed

**Before:** Constant learning rate of `1e-3` for all 5000 iterations.

**After (lines 194-202):**
```python
def get_lr(step, config):
    """Cosine learning rate schedule with linear warmup."""
    if step < config.warmup_iters:  # First 100 steps: linear warmup
        return config.learning_rate * (step + 1) / config.warmup_iters
    if step >= config.max_iters:
        return config.min_lr
    # Cosine decay from learning_rate to min_lr
    decay_ratio = (step - config.warmup_iters) / (config.max_iters - config.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.min_lr + coeff * (config.learning_rate - config.min_lr)
```

The learning rate follows this curve:
```
LR
1e-3 |   /‾‾‾‾‾‾‾‾‾‾‾‾‾\
     |  /                  \
     | /                     \
1e-4 |/                        \________
     +-------|------------|------------>
     0      100          4500         5000
           warmup        cosine         min
```

### Why This Matters

**Why warmup (steps 0-100):**

At initialization, model weights are random. The gradients in the first few steps can be very large and point in misleading directions. A large learning rate at this stage can push the model into a bad region of the loss landscape that's hard to recover from.

Linear warmup starts with a tiny learning rate (`1e-3 * 1/100 = 1e-5`) and gradually increases to the full rate. This lets the model "find its bearings" before taking large steps.

**Why cosine decay (steps 100-5000):**

As training progresses, the model is closer to a good solution. Smaller steps help it settle into a sharper minimum rather than bouncing around it. Cosine decay provides a smooth, gradual reduction.

**Why not just constant LR?**

A constant LR can work (as the original code demonstrated), but cosine scheduling consistently produces better final loss values. It's the default in most modern training recipes (GPT-3, LLaMA, Chinchilla, etc.).

**Practical impact for this model:** Expect ~0.02-0.05 improvement in validation loss compared to constant LR.

---

## 11. Proper Weight Decay Configuration

### What Changed

**Before (line 207):**
```python
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
```

**After (lines 205-216):**
```python
def configure_optimizer(model, config):
    decay_params = [p for p in model.parameters() if p.dim() >= 2]
    nodecay_params = [p for p in model.parameters() if p.dim() < 2]
    param_groups = [
        {'params': decay_params, 'weight_decay': config.weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0},
    ]
    return torch.optim.AdamW(param_groups, lr=config.learning_rate)
```

### Why This Matters

**What is weight decay?**

Weight decay adds a penalty proportional to the magnitude of weights: `loss += weight_decay * sum(w²)`. This encourages smaller weights, which acts as regularization and prevents overfitting.

**Why not apply it to everything?**

- **Bias parameters** (1D, e.g., `nn.Linear(..., bias=True)`): These are small vectors that shift outputs. Penalizing them restricts the model's ability to fit basic offsets, which hurts more than it helps.

- **LayerNorm parameters** (1D — scale γ and shift β): These normalize activations. The scale parameter γ is initialized to 1.0 and the shift β to 0.0. Penalizing them pushes γ toward 0 (which would kill the signal) and has no benefit.

- **Weight matrices** (2D, e.g., `nn.Linear` weight, `nn.Embedding` weight): These are the large parameter matrices where overfitting happens. Weight decay here is beneficial.

**The `p.dim() >= 2` trick:**

Instead of checking parameter names (fragile, depends on naming conventions), we check the tensor's number of dimensions:
- Weight matrices: 2D (shape like `[out_features, in_features]`)
- Embedding tables: 2D (shape like `[vocab_size, n_embd]`)
- Biases: 1D (shape like `[out_features]`)
- LayerNorm γ, β: 1D (shape like `[n_embd]`)

This is the same pattern used in Andrej Karpathy's nanoGPT and Meta's LLaMA.

**Note:** The original code used the default `weight_decay=0.01` from AdamW. The new code uses `0.1`, which is the more common value in modern LLM training.

---

## 12. Model Checkpointing

### What Changed

**Before:** No saving. If training crashes at step 4999, you lose everything.

**After (lines 282-290):**
```python
if losses['val'] < best_val_loss:
    best_val_loss = losses['val']
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'step': step,
        'val_loss': best_val_loss,
    }, 'best_model.pt')
```

### Why This Matters

This saves the model whenever validation loss improves. The checkpoint includes:
- `model_state_dict`: All model weights
- `config`: The full configuration (so you know how to rebuild the model)
- `step`: Which training step this was (for logging/debugging)
- `val_loss`: The validation loss at this point

**Why save the best, not the last?**

Training loss generally keeps decreasing, but validation loss can start increasing (overfitting). Saving the checkpoint with the best validation loss gives you the most generalizable model.

**How to load it later:**
```python
checkpoint = torch.load('best_model.pt')
config = checkpoint['config']
model = GPTLanguageModel(config)
model.load_state_dict(checkpoint['model_state_dict'])
```

---

## 13. Variable Shadowing Fixes

### What Changed

#### A. `iter` → `step` (line 209 → 271)

**Before:**
```python
for iter in range(max_iters):
```

**After:**
```python
for step in range(config.max_iters):
```

`iter` is a Python built-in function (`iter()` creates an iterator from any iterable). Shadowing it means you can't use `iter()` inside the loop if you ever need to. While unlikely to cause a bug in this specific code, it's a bad habit.

#### B. `data` → `split_data` in get_batch (line 53 → 170)

**Before:**
```python
def get_batch(split):
    data = train_data if split == 'train' else val_data  # shadows global 'data'
```

**After:**
```python
def get_batch(split, config, train_data, val_data):
    split_data = train_data if split == 'train' else val_data  # clear, distinct name
```

The global `data` holds the full encoded dataset. Inside `get_batch`, the local `data` refers to either the train or val split. Using the same name makes the code confusing and could lead to bugs if someone later tries to reference the global `data` inside this function.

---

## 14. `if __name__ == '__main__':` Guard

### What Changed

**Before:** Training code runs at module import time (lines 200-227 are top-level).

**After (line 220):**
```python
if __name__ == '__main__':
    # All training and generation code here
```

### Why This Matters

Without this guard, `import Andrej_GPT_Test` would:
1. Read `input.txt` from disk
2. Create a 1.6M parameter model
3. Train for 5000 iterations
4. Generate text

That's clearly not what you want when importing. The `if __name__ == '__main__':` guard ensures training only runs when you execute the script directly (`python Andrej_GPT_Test.py`), not when importing it.

This means you can now do:
```python
from Andrej_GPT_Test import GPTLanguageModel, GPTConfig
# Use the model class in another script without triggering training
```

---

## 15. Minor Cleanups

### A. Removed `m = model.to(device)` Alias

**Before:**
```python
model = BigramLanguageModel()
m = model.to(device)                    # 'm' is an alias
print(sum(p.numel() for p in m.parameters()))  # uses 'm'
optimizer = torch.optim.AdamW(model.parameters())  # uses 'model'
print(decode(m.generate(...)))           # uses 'm' again
```

**After:**
```python
model = GPTLanguageModel(config).to(config.device)  # just 'model'
# 'model' used everywhere, consistently
```

Having two names (`model` and `m`) for the same object is confusing. Note that `.to(device)` modifies the model **in-place** and returns `self`, so `m` and `model` were always the same object — the alias was unnecessary.

### B. Clearer Shape Comments

**Before:**
```python
logits = self.lm_head(x) # (B, T, vocab_size), C = n_embd...WTF!!!
# ...
B, T, C = logits.shape # C here becomes vocab_size....WTF!!!
```

**After:**
```python
logits = self.lm_head(x)  # (B, T, vocab_size)
# ...
B, T, V = logits.shape  # V = vocab_size
```

The original author's confusion was understandable — `C` was being used for both `n_embd` and `vocab_size` in different places. The new code uses `V` for vocab_size, making the distinction clear.

### C. Informative Print Statements

**Before:**
```python
print(len(text))
print(vocab_size)
print(len(data), len(train_data), len(val_data))
```

**After:**
```python
print(f"data: {len(data):,} chars | train: {len(train_data):,} | val: {len(val_data):,} | vocab: {config.vocab_size}")
```

One formatted line with labels and thousand separators (`,`) is much more useful than three bare numbers.

### D. PyTorch Version Assertion

**After (line 13):**
```python
assert torch.__version__ >= '2.0', f"Requires PyTorch 2.0+, found {torch.__version__}"
```

The modernized code depends on PyTorch 2.0+ features (`F.scaled_dot_product_attention`, `torch.compile`, `torch.amp.autocast`). A clear error message at import time is better than a cryptic `AttributeError` deep in training.

---

## Summary: Impact on Model Quality

| Change | Expected Effect on Val Loss |
|--------|----------------------------|
| Fix attention scaling bug | Significant improvement (was handicapping attention) |
| GELU activation | Small improvement (~0.01-0.03) |
| Cosine LR with warmup | Moderate improvement (~0.02-0.05) |
| Proper weight decay | Small improvement |
| Flash Attention | Same quality, faster training |
| AMP | Same quality, ~2x faster, half memory |
| torch.compile | Same quality, ~10-30% faster |
| TF32 | Same quality, ~2-3x faster matmuls |

The attention scaling fix is the only change that meaningfully affects model quality. All other changes improve training speed, code quality, or provide minor quality gains through better optimization.

---

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Vaswani et al., 2017 (original Transformer paper, defines `1/sqrt(d_k)` scaling)
- [FlashAttention](https://arxiv.org/abs/2205.14135) — Dao et al., 2022 (the algorithm behind `F.scaled_dot_product_attention`)
- [GELU](https://arxiv.org/abs/1606.08415) — Hendrycks & Gimpel, 2016
- [nanoGPT](https://github.com/karpathy/nanoGPT) — Karpathy's cleaned-up GPT implementation (uses many of the same patterns applied here)
- [PyTorch 2.0 Release Notes](https://pytorch.org/blog/pytorch-2.0-release/) — torch.compile and scaled_dot_product_attention
