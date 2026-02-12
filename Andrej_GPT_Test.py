# Character-level GPT language model trained on Shakespeare text
# Based on Andrej Karpathy's "Let's build GPT" tutorial
# Original: 2023/02/25, 2023/03/01, 2023/03/07
# Modernized: 2026/02 — PyTorch 2.x (Flash Attention, torch.compile, AMP, GELU, etc.)

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

assert torch.__version__ >= '2.0', f"Requires PyTorch 2.0+, found {torch.__version__}"

# ----------------------------- Configuration ---------------------------------

@dataclass
class GPTConfig:
    # Data
    batch_size: int = 32        # how many independent sequences to process in parallel
    block_size: int = 128       # maximum context length for predictions
    # Training
    max_iters: int = 5000
    eval_interval: int = 100
    eval_iters: int = 200
    learning_rate: float = 1e-3
    min_lr: float = 1e-4        # cosine schedule minimum learning rate
    warmup_iters: int = 100     # linear warmup steps
    weight_decay: float = 0.1   # AdamW weight decay (applied only to weight matrices)
    # Model
    n_embd: int = 128           # embedding dimension
    n_head: int = 8             # number of attention heads
    n_layer: int = 8            # number of transformer blocks
    dropout: float = 0.0
    vocab_size: int = 0         # set after loading data
    # System
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    compile_model: bool = True  # use torch.compile (GPU only)
    use_amp: bool = True        # use automatic mixed precision (GPU only)

# --------------------------------- Model -------------------------------------

class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention using F.scaled_dot_product_attention.

    Replaces the original Head + MultiHeadAttention classes.
    All heads are computed in a single batched matmul (faster than a Python loop).
    F.scaled_dot_product_attention handles:
      - Correct scaling by 1/sqrt(head_size) (the original code used 1/sqrt(n_embd))
      - Causal masking (no need for manual tril buffer)
      - Dropout during training
      - Auto-selects Flash Attention kernel when available
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # Q, K, V projections for all heads, batched into one linear layer
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = config.dropout
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.shape
        # compute Q, K, V for all heads in one matmul
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        # reshape to (B, n_head, T, head_size)
        head_size = C // self.n_head
        q = q.view(B, T, self.n_head, head_size).transpose(1, 2)
        k = k.view(B, T, self.n_head, head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_size).transpose(1, 2)
        # efficient attention (scales by 1/sqrt(head_size), applies causal mask and dropout)
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=True,
        )
        # re-assemble all head outputs
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.resid_dropout(self.c_proj(out))
        return out


class FeedForward(nn.Module):
    """Feed-forward network with GELU activation (GPT-2/3 standard)."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer block: pre-norm architecture with residual connections."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.ffwd = FeedForward(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))   # pre-norm before attention
        x = x + self.ffwd(self.ln2(x))   # pre-norm before feed-forward
        return x


class GPTLanguageModel(nn.Module):
    """Character-level GPT language model."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)    # final layer norm
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)                                   # (B, T, n_embd)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device)) # (T, n_embd)
        x = tok_emb + pos_emb  # (B, T, n_embd)
        x = self.blocks(x)     # (B, T, n_embd)
        x = self.ln_f(x)       # (B, T, n_embd)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, V = logits.shape  # V = vocab_size
            logits = logits.view(B * T, V)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        """Auto-regressively generate max_new_tokens characters."""
        for _ in range(max_new_tokens):
            # crop context to the last block_size tokens (model's max context length)
            idx_cond = idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]                       # (B, vocab_size)
            probs = F.softmax(logits, dim=-1)               # (B, vocab_size)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat((idx, idx_next), dim=-1)        # (B, T+1)
        return idx

# ------------------------------ Utilities ------------------------------------

def get_batch(split, config, train_data, val_data):
    """Sample a random batch of (input, target) pairs."""
    split_data = train_data if split == 'train' else val_data
    ix = torch.randint(len(split_data) - config.block_size, (config.batch_size,))
    x = torch.stack([split_data[i:i + config.block_size] for i in ix])
    y = torch.stack([split_data[i + 1:i + config.block_size + 1] for i in ix])
    x, y = x.to(config.device), y.to(config.device)
    return x, y


@torch.no_grad()
def estimate_loss(model, config, train_data, val_data):
    """Estimate mean loss on train and val splits."""
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(config.eval_iters)
        for k in range(config.eval_iters):
            X, Y = get_batch(split, config, train_data, val_data)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def get_lr(step, config):
    """Cosine learning rate schedule with linear warmup."""
    if step < config.warmup_iters:
        return config.learning_rate * (step + 1) / config.warmup_iters
    if step >= config.max_iters:
        return config.min_lr
    decay_ratio = (step - config.warmup_iters) / (config.max_iters - config.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.min_lr + coeff * (config.learning_rate - config.min_lr)


def configure_optimizer(model, config):
    """Configure AdamW with weight decay only on 2D+ parameters (not biases/LayerNorm)."""
    decay_params = [p for p in model.parameters() if p.dim() >= 2]
    nodecay_params = [p for p in model.parameters() if p.dim() < 2]
    param_groups = [
        {'params': decay_params, 'weight_decay': config.weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0},
    ]
    num_decay = sum(p.numel() for p in decay_params)
    num_nodecay = sum(p.numel() for p in nodecay_params)
    print(f"weight decay params: {num_decay:,} | no decay params: {num_nodecay:,}")
    return torch.optim.AdamW(param_groups, lr=config.learning_rate)

# ------------------------------- Training ------------------------------------

if __name__ == '__main__':
    # Enable TF32 for faster matmuls on Ampere+ GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    config = GPTConfig()

    # Reproducibility
    torch.manual_seed(1337)

    # Load and tokenize data
    with open('input.txt', 'r', encoding='utf-8') as f:
        text = f.read()

    chars = sorted(list(set(text)))
    config.vocab_size = len(chars)

    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])

    # Train/val split (90/10)
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    print(f"data: {len(data):,} chars | train: {len(train_data):,} | val: {len(val_data):,} | vocab: {config.vocab_size}")

    # Create model
    model = GPTLanguageModel(config).to(config.device)
    print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")

    # torch.compile for automatic kernel fusion (Note: may not work on Windows)
    if config.compile_model and config.device == 'cuda':
        print("Compiling model with torch.compile()...")
        model = torch.compile(model)

    # Mixed precision setup
    use_amp = config.use_amp and config.device == 'cuda'
    amp_dtype = torch.float16
    if use_amp and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        amp_dtype = torch.bfloat16  # preferred: same dynamic range as fp32
    scaler = torch.amp.GradScaler(enabled=use_amp and amp_dtype == torch.float16)

    # Optimizer with proper weight decay groups
    optimizer = configure_optimizer(model, config)

    # Training loop
    best_val_loss = float('inf')

    for step in range(config.max_iters):
        # Update learning rate (cosine schedule with warmup)
        lr = get_lr(step, config)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Periodic evaluation
        if step % config.eval_interval == 0 or step == config.max_iters - 1:
            losses = estimate_loss(model, config, train_data, val_data)
            print(f"step {step:5d} | train loss {losses['train']:.4f} | val loss {losses['val']:.4f} | lr {lr:.2e}")

            # Save best checkpoint
            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'config': config,
                    'step': step,
                    'val_loss': best_val_loss,
                }, 'best_model.pt')

        # Forward/backward with AMP
        xb, yb = get_batch('train', config, train_data, val_data)
        with torch.amp.autocast(device_type=config.device, dtype=amp_dtype, enabled=use_amp):
            logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    # Generate from the trained model
    model.eval()
    context = torch.zeros((1, 1), dtype=torch.long, device=config.device)
    print(decode(model.generate(context, max_new_tokens=2000)[0].tolist()))
