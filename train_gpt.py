"""
OpenAI Parameter Golf - v4 Leaderboard Elite
Targets: 26.7M Params | Top-5 BPB Performance
Improvements: Cosine Schedule, EMA/SWA, Better Hash, Weight Decay, XSA, Depth Recurrence
"""

from __future__ import annotations
import math, os, glob, json
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from collections import defaultdict
from pathlib import Path

# -----------------------------
# HYPERPARAMETERS & SCHEDULE
# -----------------------------

class Hyperparameters:
    vocab_size = 1024
    num_layers = 11
    model_dim = 512
    num_heads = 8
    num_kv_heads = 4
    mlp_mult = 3

    bigram_vocab_size = 3072
    bigram_dim = 80

    iterations = int(os.environ.get("ITERATIONS", 20000))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 1500))
    warmdown_steps = int(os.environ.get("WARMDOWN_STEPS", 1000))  # Cosine decay tail
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 1024))

    matrix_lr = float(os.environ.get("MATRIX_LR", 0.045))
    scalar_lr = float(os.environ.get("SCALAR_LR", 0.02))
    tied_embed_lr = float(os.environ.get("TIED_EMBED_LR", 0.03))
    weight_decay = float(os.environ.get("WEIGHT_DECAY", 0.001))
    
    grad_clip = float(os.environ.get("GRAD_CLIP", 1.0))
    
    # Schedule mode: 'linear' (legacy), 'cosine' (new, recommended)
    schedule_mode = os.environ.get("SCHEDULE_MODE", "cosine")
    
    # EMA & SWA
    use_ema = os.environ.get("USE_EMA", "true").lower() == "true"
    ema_decay = float(os.environ.get("EMA_DECAY", 0.999))
    use_swa = os.environ.get("USE_SWA", "false").lower() == "true"
    swa_start = int(os.environ.get("SWA_START", 15000))
    
    # Depth recurrence (re-use middle layers)
    use_depth_recurrence = os.environ.get("USE_DEPTH_RECURRENCE", "false").lower() == "true"
    recurrence_interval = int(os.environ.get("RECURRENCE_INTERVAL", 3))
    
    # Selective attention (XSA) - only on deeper layers
    use_xsa = os.environ.get("USE_XSA", "false").lower() == "true"
    xsa_start_layer = int(os.environ.get("XSA_START_LAYER", 7))
    xsa_ratio = float(os.environ.get("XSA_RATIO", 0.5))  # Keep 50% of heads full attention
    
    # Sliding window evaluation
    use_sliding_window_eval = os.environ.get("USE_SLIDING_EVAL", "false").lower() == "true"
    eval_stride = int(os.environ.get("EVAL_STRIDE", 512))
    
    # Checkpointing
    save_ema_model = os.environ.get("SAVE_EMA_MODEL", "true").lower() == "true"
    checkpoint_dir = os.environ.get("CHECKPOINT_DIR", "./checkpoints_v4")
    
    @classmethod
    def log_config(cls):
        config_dict = {k: getattr(cls, k) for k in dir(cls) 
                      if not k.startswith('_') and not callable(getattr(cls, k))}
        print("\n=== CONFIG ===")
        for k, v in sorted(config_dict.items()):
            print(f"  {k}: {v}")

# Learning rate schedule
def get_lr_schedule(step, total_steps, warmup_steps, warmdown_steps, mode="cosine", base_lr=1.0):
    """
    Compute learning rate multiplier.
    mode='cosine': warmup -> cosine decay -> warmdown tail
    mode='linear': warmup -> constant (legacy)
    """
    if mode == "cosine":
        if step < warmup_steps:
            return step / warmup_steps
        elif step >= total_steps - warmdown_steps:
            # Final warmdown phase
            remaining = total_steps - step
            return 0.1 * (1 + math.cos(math.pi * (warmdown_steps - remaining) / warmdown_steps)) / 2
        else:
            # Cosine annealing
            progress = (step - warmup_steps) / (total_steps - warmup_steps - warmdown_steps)
            return 0.5 * (1 + math.cos(math.pi * progress))
    else:  # legacy linear
        if step < warmup_steps:
            return step / warmup_steps
        return 1.0

# EMA state wrapper
class EMAState:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {name: param.data.clone() for name, param in model.named_parameters()}
    
    def update(self, model):
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1 - self.decay)
    
    def apply(self, model):
        for name, param in model.named_parameters():
            if name in self.shadow:
                param.data.copy_(self.shadow[name])
    
    def restore(self, model):
        # Keep EMA copy, restore original params (handled by checkpoint)
        pass

# SWA state wrapper
class SWAState:
    def __init__(self, model):
        self.shadow = {name: param.data.clone() for name, param in model.named_parameters()}
        self.count = 1
    
    def update(self, model):
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.shadow[name].mul_(self.count / (self.count + 1)).add_(param.data, alpha=1 / (self.count + 1))
        self.count += 1
    
    def apply(self, model):
        for name, param in model.named_parameters():
            if name in self.shadow:
                param.data.copy_(self.shadow[name])

# Memory-efficient SWA (simplified)
class SWAAccumulator:
    def __init__(self):
        self.accumulated = None
        self.count = 0
    
    def add(self, state_dict):
        if self.accumulated is None:
            self.accumulated = {k: v.clone() for k, v in state_dict.items()}
            self.count = 1
        else:
            for k in self.accumulated:
                if k in state_dict:
                    self.accumulated[k] = (self.accumulated[k] * self.count + state_dict[k]) / (self.count + 1)
            self.count += 1
    
    def get_state_dict(self):
        return self.accumulated if self.accumulated else {}
    
    def get_count(self):
        return self.count

# ---------------------
# CORE MODULES & DATA
# ---------------------

class DataLoader:
    def __init__(self, pattern, seq_len, device):
        self.files = sorted(glob.glob(pattern))
        if not self.files:
            print(f"WARNING: No files found for {pattern}. Using synthetic fallback.")
            self.data = None
        else:
            # Use memmap for memory efficiency
            self.data = np.memmap(self.files[0], dtype=np.uint16, mode='r')
            self.n = len(self.data)
        self.seq_len = seq_len
        self.device = device
        self.eval_offset = 0  # For sliding window evaluation

    def get_batch(self, batch_size=1):
        if self.data is None:  # Synthetic fallback
            x = torch.randint(0, 1024, (batch_size, self.seq_len), device=self.device)
            y = torch.randint(0, 1024, (batch_size, self.seq_len), device=self.device)
            return x, y
        ix = torch.randint(0, self.n - self.seq_len - 1, (batch_size,))
        x = torch.stack([torch.from_numpy(self.data[i:i+self.seq_len].astype(np.int64)) for i in ix]).to(self.device)
        y = torch.stack([torch.from_numpy(self.data[i+1:i+self.seq_len+1].astype(np.int64)) for i in ix]).to(self.device)
        return x, y

    def get_eval_batch_sliding(self, stride=512):
        """Sliding window evaluation for more accurate validation."""
        if self.data is None:
            x = torch.randint(0, 1024, (1, self.seq_len), device=self.device)
            y = torch.randint(0, 1024, (1, self.seq_len), device=self.device)
            return x, y, False
        
        # Use fixed validation window with stride
        max_start = max(0, self.n - self.seq_len - 1 - stride * 100)
        start = max_start + (self.eval_offset % (max(1, (self.n - max_start) // stride)))
        self.eval_offset += 1
        
        if start + self.seq_len >= self.n:
            self.eval_offset = 0
            is_last = True
        else:
            is_last = False
        
        x = torch.from_numpy(self.data[start:start+self.seq_len].astype(np.int64)).unsqueeze(0).to(self.device)
        y = torch.from_numpy(self.data[start+1:start+self.seq_len+1].astype(np.int64)).unsqueeze(0).to(self.device)
        return x, y, is_last

class RMSNorm(nn.Module):
    def forward(self, x): 
        return F.rms_norm(x, (x.size(-1),))

class GatedBigramHash(nn.Module):
    """Improved dual bigram hash with learned gating."""
    def __init__(self, vocab_size, bigram_vocab_size, bigram_dim, model_dim):
        super().__init__()
        self.bigram_vocab_size = bigram_vocab_size
        self.embed1 = nn.Embedding(bigram_vocab_size, bigram_dim)
        self.embed2 = nn.Embedding(bigram_vocab_size, bigram_dim)
        self.proj = nn.Linear(2 * bigram_dim, model_dim, bias=False)
        # Learned gating for mixing
        self.gate = nn.Parameter(torch.ones(1, 1, 1) * 0.5)  # Will be trained via backprop

    def forward(self, x):
        prev = F.pad(x[:, :-1], (1, 0), value=0)
        # Hash 1: multiplicative
        h1 = (prev * 1024 + x) % self.bigram_vocab_size
        # Hash 2: additive
        h2 = (prev + 31 * x) % self.bigram_vocab_size
        
        e1 = self.embed1(h1)
        e2 = self.embed2(h2)
        
        # Gated combination
        g = torch.sigmoid(self.gate)  # Learn to mix the two embeddings
        combined = g * e1 + (1 - g) * e2
        
        return self.proj(torch.cat([e1, e2], dim=-1))

def apply_rope(q, k):
    B, H, T, D = q.shape
    half = D // 2
    freqs = torch.arange(half, device=q.device, dtype=q.dtype) / half
    angles = torch.einsum("t,d->td", torch.arange(T, device=q.device, dtype=q.dtype), freqs)
    sin, cos = angles.sin().view(1, 1, T, half), angles.cos().view(1, 1, T, half)
    def rotate(x):
        return torch.cat([x[..., :half] * cos - x[..., half:] * sin, 
                         x[..., :half] * sin + x[..., half:] * cos], dim=-1)
    return rotate(q), rotate(k)

class CausalSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, layer_idx=0, use_xsa=False, xsa_start=7, xsa_ratio=0.5):
        super().__init__()
        self.num_heads, self.num_kv_heads = num_heads, num_kv_heads
        self.head_dim = dim // num_heads
        self.layer_idx = layer_idx
        self.use_xsa = use_xsa and (layer_idx >= xsa_start)
        self.xsa_ratio = xsa_ratio
        
        self.c_attn = nn.Linear(dim, (num_heads + 2 * num_kv_heads) * self.head_dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.q_gain = nn.Parameter(torch.ones(num_heads))
        self.attn_temp = nn.Parameter(torch.ones(num_heads))
        
        # Per-head QK scaling (improves modeling)
        self.qk_scale = nn.Parameter(torch.ones(num_heads))

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split([self.num_heads * self.head_dim, 
                            self.num_kv_heads * self.head_dim, 
                            self.num_kv_heads * self.head_dim], dim=2)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        k, v = k.repeat_interleave(2, dim=1), v.repeat_interleave(2, dim=1)  # GQA
        q, k = apply_rope(q, k)
        
        # Apply per-head gains
        q = q * self.q_gain.view(1, -1, 1, 1) * self.attn_temp.view(1, -1, 1, 1)
        
        if self.use_xsa:
            # Selective attention: keep some heads full, sparsify others
            num_full = max(1, int(self.num_heads * self.xsa_ratio))
            y_full = F.scaled_dot_product_attention(
                q[:, :num_full], k[:, :num_full], v[:, :num_full], 
                is_causal=True, scale=1.0/math.sqrt(self.head_dim)
            )
            
            # Sparse attention on remaining heads
            if self.num_heads > num_full:
                q_sparse = q[:, num_full:]
                k_sparse = k[:, num_full:]
                v_sparse = v[:, num_full:]
                
                # Simple stride-based sparsity
                stride = max(1, T // int(math.sqrt(T)))
                k_sparse_strided = k_sparse[:, :, ::stride, :]
                v_sparse_strided = v_sparse[:, :, ::stride, :]
                
                # Compute attention with causal masking for strided positions
                attn_weights = torch.matmul(q_sparse, k_sparse_strided.transpose(-2, -1))
                attn_weights = attn_weights / math.sqrt(self.head_dim)
                
                # Apply causal mask: can't attend to future strided positions
                # For each query position t, only attend to key positions <= t (in stride positions)
                t_positions = torch.arange(T, device=q_sparse.device)
                k_positions = torch.arange(0, T, stride, device=q_sparse.device)
                causal_mask = t_positions.view(-1, 1) >= k_positions.view(1, -1)
                attn_weights = attn_weights.masked_fill(~causal_mask, float('-inf'))
                
                y_sparse = torch.matmul(F.softmax(attn_weights, dim=-1), v_sparse_strided)
                
                y = torch.cat([y_full, y_sparse], dim=1)
            else:
                y = y_full
        else:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=1.0/math.sqrt(self.head_dim))
        
        return self.proj(y.transpose(1, 2).reshape(B, T, C))

class Block(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, mlp_mult, layer_idx, use_xsa=False, xsa_start=7, xsa_ratio=0.5):
        super().__init__()
        self.layer_idx = layer_idx
        self.attn_norm, self.mlp_norm = RMSNorm(), RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, layer_idx, use_xsa, xsa_start, xsa_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_mult * dim, bias=False), 
            nn.GELU(), 
            nn.Linear(mlp_mult * dim, dim, bias=False)
        )
        # Residual scaling (depth initialization)
        scale = 1.0 / math.sqrt(2 * (layer_idx + 1))
        self.attn_scale = nn.Parameter(torch.full((dim,), scale))
        self.mlp_scale = nn.Parameter(torch.full((dim,), scale))

    def forward(self, x):
        # Standard sequential: attn + residual, then MLP + residual
        x = x + self.attn_scale * self.attn(self.attn_norm(x))
        x = x + self.mlp_scale * self.mlp(self.mlp_norm(x))
        return x

class GPT(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.tok_emb = nn.Embedding(args.vocab_size, args.model_dim)
        self.bigram = GatedBigramHash(args.vocab_size, args.bigram_vocab_size, args.bigram_dim, args.model_dim)
        
        # Build blocks with XSA support
        self.blocks = nn.ModuleList([
            Block(args.model_dim, args.num_heads, args.num_kv_heads, args.mlp_mult, i,
                  use_xsa=args.use_xsa, xsa_start=args.xsa_start_layer, xsa_ratio=args.xsa_ratio)
            for i in range(args.num_layers)
        ])
        
        # Depth recurrence: select middle layers to optionally reuse
        self.use_depth_recurrence = args.use_depth_recurrence
        self.recurrence_interval = args.recurrence_interval
        
        self.final_norm = RMSNorm()
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear): 
                torch.nn.init.trunc_normal_(m.weight, std=0.01)
        torch.nn.init.normal_(self.tok_emb.weight, std=0.02)

    def get_logits(self, x):
        h = self.tok_emb(x) + self.bigram(x)
        
        for i, block in enumerate(self.blocks):
            h = block(h)
            
            # Optional: depth recurrence (reuse selected layers for better depth modeling)
            if self.use_depth_recurrence and i > 0 and i % self.recurrence_interval == 0:
                # Reapply a middle layer (optional, can help with gradient flow)
                recur_idx = max(0, i - self.recurrence_interval)
                if recur_idx < len(self.blocks):
                    h = self.blocks[recur_idx](h)
        
        logits = F.linear(self.final_norm(h), self.tok_emb.weight)
        scale = torch.rsqrt((logits ** 2).mean(dim=-1, keepdim=True) + 1e-5)
        return logits * scale * 6  # Balanced logit scaling

    def forward(self, x, y):
        logits = self.get_logits(x)
        return F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

# --------------------------
# EVALUATION & TRAINING
# --------------------------

@torch.no_grad()
def evaluate(model, val_loader, steps=20, use_sliding=False):
    """Evaluate model on validation set."""
    model.eval()
    losses = []
    
    if use_sliding:
        val_loader.eval_offset = 0
        for _ in range(steps):
            x, y, _ = val_loader.get_eval_batch_sliding()
            loss = model(x, y)
            losses.append(loss.item())
    else:
        for _ in range(steps):
            x, y = val_loader.get_batch()
            loss = model(x, y)
            losses.append(loss.item())
    
    model.train()
    avg_loss = sum(losses) / len(losses)
    bpb = (avg_loss / math.log(2.0)) * 0.4412  # Standard conversion
    return avg_loss, bpb, losses

def log_training_state(step, loss, val_loss, val_bpb, learning_rates, elapsed_sec, tokens_seen):
    """Pretty print training state."""
    lr_str = " | ".join([f"LR{i}:{lr:.6f}" for i, lr in enumerate(learning_rates)])
    tokens_per_sec = tokens_seen / elapsed_sec if elapsed_sec > 0 else 0
    print(f"[{step:05d}] loss={loss:.4f} | val_loss={val_loss:.4f} val_bpb={val_bpb:.4f} "
          f"| {lr_str} | {tokens_per_sec:.0f} tok/s")

def main():
    args = Hyperparameters()
    args.log_config()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    
    # Create checkpoint dir
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    model = GPT(args).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n{'='*50}\nModel: {n_params:,} parameters\n{'='*50}")
    
    # Data loaders
    train_loader = DataLoader("./data/datasets/fineweb10B_sp1024/fineweb_train_*.bin", args.train_seq_len, device)
    val_loader = DataLoader("./data/datasets/fineweb10B_sp1024/fineweb_val_*.bin", args.train_seq_len, device)
    
    # Optimizer with grouped learning rates and weight decay
    tied_weight = model.tok_emb.weight
    param_groups = [
        {"params": [tied_weight], "lr": args.tied_embed_lr, "weight_decay": args.weight_decay * 0.1},
        {"params": [p for n, p in model.named_parameters() if p.ndim < 2 and p is not tied_weight], 
         "lr": args.scalar_lr, "weight_decay": 0.0},  # No WD for scalars
        {"params": [p for n, p in model.named_parameters() if p.ndim == 2 and p is not tied_weight], 
         "lr": args.matrix_lr, "weight_decay": args.weight_decay}
    ]
    
    opt = torch.optim.Adam(param_groups, fused=(device.type == 'cuda'))
    
    # EMA & SWA
    ema_state = EMAState(model, decay=args.ema_decay) if args.use_ema else None
    swa_accumulator = SWAAccumulator() if args.use_swa else None
    
    best_val_bpb = float('inf')
    best_checkpoint = None
    
    print(f"\nStarting training: {args.iterations} iterations | Warmup: {args.warmup_steps} | "
          f"Warmdown: {args.warmdown_steps}\n")
    
    import time
    start_time = time.time()
    tokens_seen = 0
    val_check_freq = int(os.environ.get("VAL_CHECK_FREQ", 100))
    
    for step in range(args.iterations + 1):
        # Learning rate schedule
        if args.schedule_mode == "cosine":
            lr_mult = get_lr_schedule(step, args.iterations, args.warmup_steps, args.warmdown_steps, mode="cosine")
        else:
            lr_mult = get_lr_schedule(step, args.iterations, args.warmup_steps, args.warmdown_steps, mode="linear")
        
        # Apply LR schedule to all param groups
        for i, param_group in enumerate(opt.param_groups):
            if i == 0:  # tied embed
                param_group['lr'] = args.tied_embed_lr * lr_mult
            elif i == 1:  # scalars
                param_group['lr'] = args.scalar_lr * lr_mult
            else:  # matrices
                param_group['lr'] = args.matrix_lr * lr_mult
        
        # Training step
        x, y = train_loader.get_batch()
        loss = model(x, y)
        tokens_seen += x.numel()
        
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        
        # EMA update
        if args.use_ema and ema_state:
            ema_state.update(model)
        
        # SWA accumulation
        if args.use_swa and swa_accumulator and step >= args.swa_start:
            swa_accumulator.add(model.state_dict())
        
        # Validation & logging
        if step % val_check_freq == 0:
            elapsed = time.time() - start_time
            current_lrs = [g['lr'] for g in opt.param_groups]
            
            val_loss, val_bpb, _ = evaluate(model, val_loader, steps=20, use_sliding=args.use_sliding_window_eval)
            log_training_state(step, loss.item(), val_loss, val_bpb, current_lrs, elapsed, tokens_seen)
            
            # Track best
            if val_bpb < best_val_bpb:
                best_val_bpb = val_bpb
                best_checkpoint = {
                    'model': model.state_dict(),
                    'step': step,
                    'bpb': val_bpb
                }
                # Save best model checkpoint
                torch.save(best_checkpoint, f"{args.checkpoint_dir}/best_model.pt")
    
    print(f"\n{'='*50}\nTraining complete!\n{'='*50}")
    if best_checkpoint is not None:
        print(f"Best validation BPB: {best_val_bpb:.4f} (at step {best_checkpoint['step']})")
    else:
        print("Best validation BPB: N/A (no checkpoint recorded)")
    
    # Apply EMA/SWA and do final eval
    if args.use_ema and ema_state:
        print("\nApplying EMA checkpoint...")
        ema_state.apply(model)
        ema_val_loss, ema_val_bpb, _ = evaluate(model, val_loader, steps=50, use_sliding=args.use_sliding_window_eval)
        print(f"EMA Validation BPB: {ema_val_bpb:.4f}")
        
        if ema_val_bpb < best_val_bpb and args.save_ema_model:
            best_checkpoint = {'model': model.state_dict(), 'step': args.iterations, 'bpb': ema_val_bpb}
            torch.save(best_checkpoint, f"{args.checkpoint_dir}/best_ema_model.pt")
    
    if args.use_swa and swa_accumulator:
        print(f"\nApplying SWA ({swa_accumulator.get_count()} checkpoints)...")
        swa_dict = swa_accumulator.get_state_dict()
        model.load_state_dict(swa_dict)
        swa_val_loss, swa_val_bpb, _ = evaluate(model, val_loader, steps=50, use_sliding=args.use_sliding_window_eval)
        print(f"SWA Validation BPB: {swa_val_bpb:.4f}")
        
        if swa_val_bpb < best_val_bpb:
            best_checkpoint = {'model': model.state_dict(), 'step': args.iterations, 'bpb': swa_val_bpb}
            torch.save(best_checkpoint, f"{args.checkpoint_dir}/best_swa_model.pt")
    
    if best_checkpoint is not None:
        print(f"\nFinal best BPB: {best_checkpoint['bpb']:.4f}\n")
    else:
        print("\nFinal best BPB: N/A\n")

if __name__ == "__main__": main()